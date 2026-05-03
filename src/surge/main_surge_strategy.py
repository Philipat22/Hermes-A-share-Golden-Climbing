"""
主升浪策略 — 趋势识别 + 突破入场 + 动态止损

三层架构:
  Layer 1: 趋势筛选 — ADX + 均线排列 + 成交量
  Layer 2: 入场信号 — 平台突破 / N型反转 / VCP收缩
  Layer 3: 风控执行 — ATR止损 / 移动止损 / 时间止损 / 仓位管理

用法:
    from src.surge.main_surge_strategy import MainSurgeStrategy, compute_indicators
    # 1. 预计算指标
    df_indicators = compute_indicators(price_df)
    # 2. 创建策略实例
    strategy = MainSurgeStrategy(indicators=df_indicators, params={...})
    # 3. 每日调用 strategy.check_signals(date) 获取信号
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field


# ════════════════════════════════════════════════════════════
# 数据结构
# ════════════════════════════════════════════════════════════

@dataclass
class StrategyParams:
    """主升浪策略参数 — 全部可配置，支持网格搜索"""
    # Layer 1: 趋势筛选
    adx_threshold: float = 25.0          # ADX 阈值 (20/25/30)
    ma_alignment: str = "tight"           # 均线排列: tight(MA5>MA20>MA50) / loose(MA20>MA60)
    price_above_ma200: bool = True       # 价格是否需在 MA200 之上
    vol_ratio_threshold: float = 1.5     # 成交量相对20日均量倍数

    # Layer 2: 入场信号
    platform_min_days: int = 15          # 平台整理最少天数
    platform_max_amplitude: float = 0.15 # 平台最大振幅
    breakout_vol_ratio: float = 2.0      # 突破量比
    n_first_leg_min: float = 0.15        # N型第一波最小涨幅
    n_pullback_max: float = 0.04         # N型回调最大幅度
    vcp_window: int = 60                 # VCP 收缩观察窗口
    sector_min_peers: int = 3            # 板块共振最少同板块股数
    macd_confirm: bool = True            # 是否要求 MACD 金叉确认
    bb_squeeze_confirm: bool = True      # 是否要求布林带收窄确认

    # Layer 3: 风控
    stop_loss_pct: float = -0.05         # 硬止损百分比
    stop_loss_atr_mult: float = 2.0     # ATR 止损倍数 (取两者中更紧的)
    trailing_start_pct: float = 0.08    # 移动止损触发盈利百分比
    trailing_atr_mult: float = 3.0      # 移动止损 ATR 倍数
    time_stop_days: int = 20            # 时间止损 (交易日)
    max_position_pct: float = 0.20      # 单票最大仓位
    bear_regime_cash_pct: float = 0.50  # BEAR 市保留现金比例

    # 全局
    max_positions: int = 4              # 最大持仓数 (保守)
    min_price: float = 3.0              # 最低股价过滤
    max_price: float = 200.0            # 最高股价过滤
    min_hold_days: int = 5              # 最低持仓天数 (避免日内翻转)


DEFAULT_PARAMS = StrategyParams()


# ════════════════════════════════════════════════════════════
# 指标预计算
# ════════════════════════════════════════════════════════════

def compute_indicators(prices_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """从价格字典预计算全部趋势指标

    Args:
        prices_dict: {vt_symbol: DataFrame(columns=['date','open','high','low','close','volume'])}

    Returns:
        DataFrame with columns:
            vt_symbol, date, open, high, low, close, volume,
            ma5, ma20, ma50, ma60, ma200,
            adx, plus_di, minus_di,
            macd, macd_signal, macd_hist,
            bb_middle, bb_upper, bb_lower, bb_pct_b, bb_bandwidth,
            atr_14, atr_pct,
            vol_ma20, vol_ratio
    """
    dfs = []
    for sym, df in prices_dict.items():
        if df is None or len(df) < 200:
            continue

        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values

        # ── 均线 ──
        df['ma5'] = pd.Series(closes).rolling(5).mean().values
        df['ma20'] = pd.Series(closes).rolling(20).mean().values
        df['ma50'] = pd.Series(closes).rolling(50).mean().values
        df['ma60'] = pd.Series(closes).rolling(60).mean().values
        df['ma200'] = pd.Series(closes).rolling(200).mean().values

        # ── ADX ──
        adx, plus_di, minus_di = _compute_adx_np(highs, lows, closes, window=14)
        df['adx'] = adx
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        # ── MACD ──
        macd_line, macd_signal, macd_hist = _compute_macd_np(closes)
        df['macd'] = macd_line
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist

        # ── Bollinger Bands ──
        bb_mid, bb_up, bb_low = _compute_bb_np(closes, window=20, num_std=2.0)
        df['bb_middle'] = bb_mid
        df['bb_upper'] = bb_up
        df['bb_lower'] = bb_low
        df['bb_pct_b'] = (closes - bb_low) / (bb_up - bb_low)
        df['bb_bandwidth'] = (bb_up - bb_low) / bb_mid

        # ── ATR ──
        atr, atr_pct = _compute_atr_np(highs, lows, closes, window=14)
        df['atr_14'] = atr
        df['atr_pct'] = atr_pct

        # ── 成交量 ──
        df['vol_ma20'] = pd.Series(volumes).rolling(20).mean().values
        df['vol_ratio'] = volumes / df['vol_ma20'].values

        df['vt_symbol'] = sym
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    return result


# ════════════════════════════════════════════════════════════
# 指标计算 (纯 NumPy，不依赖 talib)
# ════════════════════════════════════════════════════════════

def _compute_adx_np(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """纯 NumPy ADX 计算"""
    n = len(high)
    adx = np.full(n, np.nan)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)

    if n < window + 1:
        return adx, plus_di, minus_di

    # True Range
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )

    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Wilder smoothing
    atr_smooth = np.zeros(n)
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)

    # Initial values (simple mean of first window)
    atr_smooth[window] = np.mean(tr[:window])
    plus_dm_smooth[window] = np.mean(plus_dm[:window])
    minus_dm_smooth[window] = np.mean(minus_dm[:window])

    alpha = 1.0 / window
    for i in range(window + 1, n):
        atr_smooth[i] = atr_smooth[i-1] + alpha * (tr[i-1] - atr_smooth[i-1])
        plus_dm_smooth[i] = plus_dm_smooth[i-1] + alpha * (plus_dm[i-1] - plus_dm_smooth[i-1])
        minus_dm_smooth[i] = minus_dm_smooth[i-1] + alpha * (minus_dm[i-1] - minus_dm_smooth[i-1])

    # +DI / -DI
    for i in range(window, n):
        if atr_smooth[i] > 0:
            plus_di[i] = (plus_dm_smooth[i] / atr_smooth[i]) * 100.0
            minus_di[i] = (minus_dm_smooth[i] / atr_smooth[i]) * 100.0

    # DX then ADX (Wilder smooth)
    dx = np.full(n, np.nan)
    for i in range(window, n):
        denom = plus_di[i] + minus_di[i]
        if denom > 0:
            dx[i] = (abs(plus_di[i] - minus_di[i]) / denom) * 100.0

    # ADX = Wilder smooth of DX
    if window * 2 - 1 < n:
        adx[window * 2 - 1] = np.nanmean(dx[window:window*2])
        for i in range(window * 2, n):
            if not np.isnan(dx[i]):
                adx[i] = adx[i-1] + alpha * (dx[i] - adx[i-1])
            else:
                adx[i] = adx[i-1]

    return adx, plus_di, minus_di


def _compute_macd_np(
    close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """纯 NumPy MACD 计算"""
    n = len(close)
    macd_line = np.full(n, np.nan)
    macd_signal = np.full(n, np.nan)
    macd_hist = np.full(n, np.nan)

    if n < slow:
        return macd_line, macd_signal, macd_hist

    # EMA
    ema_fast = _ema_np(close, fast)
    ema_slow = _ema_np(close, slow)

    macd_line = ema_fast - ema_slow
    macd_signal_raw = _ema_np(macd_line[slow-1:], signal)
    # Skip the first (signal-1) NaNs from signal EMA
    macd_signal_valid = macd_signal_raw[signal-1:]
    macd_signal = np.full(n, np.nan)
    macd_signal[slow + signal - 2:] = macd_signal_valid

    macd_hist = macd_line - macd_signal

    return macd_line, macd_signal, macd_hist


def _ema_np(data: np.ndarray, window: int) -> np.ndarray:
    """指数移动平均"""
    n = len(data)
    result = np.full(n, np.nan)
    if n < window:
        return result

    alpha = 2.0 / (window + 1)
    # 初始 SMA
    result[window - 1] = np.mean(data[:window])
    for i in range(window, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]

    return result


def _compute_bb_np(
    close: np.ndarray, window: int = 20, num_std: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands"""
    n = len(close)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    if n < window:
        return middle, upper, lower

    # Rolling mean and std
    s = pd.Series(close)
    middle = s.rolling(window).mean().values
    std = s.rolling(window).std().values
    upper = middle + num_std * std
    lower = middle - num_std * std

    return middle, upper, lower


def _compute_atr_np(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14
) -> tuple[np.ndarray, np.ndarray]:
    """ATR 及百分比"""
    n = len(high)
    atr = np.full(n, np.nan)
    atr_pct = np.full(n, np.nan)

    if n < window + 1:
        return atr, atr_pct

    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )

    # Rolling mean
    s = pd.Series(np.concatenate([[np.nan], tr]))
    atr = s.rolling(window, min_periods=window).mean().values

    atr_pct = atr / close * 100.0

    return atr, atr_pct


# ════════════════════════════════════════════════════════════
# 主策略引擎
# ════════════════════════════════════════════════════════════

@dataclass
class Position:
    """持仓记录"""
    symbol: str
    entry_date: str
    entry_price: float
    shares: int
    stop_loss_price: float          # 硬止损价
    highest_close: float = 0.0      # 持仓期间最高收盘价(移动止损用)
    trailing_active: bool = False   # 移动止损是否已激活
    trailing_stop_price: float = 0.0  # 移动止损价


@dataclass
class Signal:
    """交易信号"""
    symbol: str
    action: str                     # 'BUY' | 'SELL'
    reason: str                     # 原因描述
    score: float = 0.0              # 信号强度评分
    target_shares: int = 0          # 目标股数
    stop_loss_price: float = 0.0    # 建议止损价


class MainSurgeStrategy:
    """主升浪策略引擎

    不与 BacktestingEngine 耦合 — 纯信号生成。
    回测脚本负责将信号转化为实际交易。
    """

    def __init__(
        self,
        indicators: pd.DataFrame,
        params: StrategyParams = None,
        sector_map: dict[str, list[str]] = None,
    ):
        """
        Args:
            indicators: compute_indicators() 的输出
            params: 策略参数
            sector_map: 板块→股票映射 (用于板块共振)
        """
        self.params = params or DEFAULT_PARAMS
        self.sector_map = sector_map or {}

        # 建立快速查找索引: (date, symbol) → row
        self._indicator_index: dict[tuple, pd.Series] = {}
        for _, row in indicators.iterrows():
            date_key = str(row['date'])[:10]
            self._indicator_index[(date_key, row['vt_symbol'])] = row

        self.all_dates = sorted(set(str(d)[:10] for d in indicators['date']))
        self.all_symbols = sorted(indicators['vt_symbol'].unique())

        # 状态
        self.positions: dict[str, Position] = {}
        self.regime_cache: dict[str, str] = {}  # date → regime

    # ════════════════════════════════════════════════════
    # Layer 1: 趋势筛选
    # ════════════════════════════════════════════════════

    def _get_row(self, date: str, symbol: str) -> Optional[pd.Series]:
        return self._indicator_index.get((date, symbol))

    def _trend_filter(self, date: str, symbol: str) -> tuple[bool, str]:
        """检查 Layer 1 趋势条件

        Returns: (passed, reason)
        """
        p = self.params
        row = self._get_row(date, symbol)
        if row is None:
            return False, "no_data"

        close = row.get('close', 0)
        if pd.isna(close) or close <= 0:
            return False, "no_close"

        # 价格过滤
        if close < p.min_price or close > p.max_price:
            return False, f"price_filter({close:.1f})"

        # ADX 趋势强度
        adx = row.get('adx', 0)
        if pd.isna(adx) or adx < p.adx_threshold:
            return False, f"adx_low({adx:.1f})"

        # 方向确认: +DI > -DI
        plus_di = row.get('plus_di', 0)
        minus_di = row.get('minus_di', 0)
        if not pd.isna(plus_di) and not pd.isna(minus_di):
            if plus_di <= minus_di:
                return False, f"di_direction(+{plus_di:.1f} vs -{minus_di:.1f})"

        # 均线排列
        if p.ma_alignment == "tight":
            ma5 = row.get('ma5', 0)
            ma20 = row.get('ma20', 0)
            ma50 = row.get('ma50', 0)
            if pd.isna(ma5) or pd.isna(ma20) or pd.isna(ma50):
                return False, "no_ma"
            if not (close > ma5 > ma20 > ma50):
                return False, "ma_misalign"
        else:  # loose
            ma20 = row.get('ma20', 0)
            ma60 = row.get('ma60', 0)
            if pd.isna(ma20) or pd.isna(ma60):
                return False, "no_ma"
            if not (close > ma20 > ma60):
                return False, "ma_misalign"

        # MA200 过滤器
        if p.price_above_ma200:
            ma200 = row.get('ma200', 0)
            if pd.isna(ma200) or close <= ma200:
                return False, "below_ma200"

        # 成交量
        vol_ratio = row.get('vol_ratio', 0)
        if not pd.isna(vol_ratio) and vol_ratio < p.vol_ratio_threshold:
            return False, f"low_vol({vol_ratio:.1f}x)"

        return True, "ok"

    # ════════════════════════════════════════════════════
    # Layer 2: 入场信号
    # ════════════════════════════════════════════════════

    def _entry_signal(self, date: str, symbol: str) -> Optional[Signal]:
        """检测入场信号 — 需历史形态确认"""
        p = self.params
        row = self._get_row(date, symbol)
        if row is None:
            return None

        close = row['close']

        # ── MACD 金叉确认: 今日 hist > 0 且昨日 <= 0 ──
        if p.macd_confirm:
            macd_hist = row.get('macd_hist', 0)
            if pd.isna(macd_hist) or macd_hist <= 0:
                return None
            # 要求金叉（今日转正）
            prev_row = self._get_prev_row(date, symbol)
            if prev_row is not None:
                prev_hist = prev_row.get('macd_hist', 0)
                if not pd.isna(prev_hist) and prev_hist > 0:
                    return None  # 不是金叉，只是持续为正

        score = 0.0
        reasons = []

        # ── 信号1: 平台突破 (需确认横盘历史) ──
        if self._detect_platform_breakout(date, symbol, close):
            score += 0.40
            reasons.append("platform_breakout")

        # ── 信号2: N型突破 (需识别三波走势) ──
        if self._detect_n_pattern(date, symbol, close):
            score += 0.35
            reasons.append("n_pattern")

        # ── 信号3: VCP 收缩突破 (需多期波动率递减) ──
        if self._detect_vcp(date, symbol, close):
            score += 0.35
            reasons.append("vcp")

        # ── 信号4: 板块共振 ──
        sector_score = self._sector_resonance(date, symbol)
        if sector_score > 0:
            score += sector_score * 0.15
            reasons.append(f"sector({sector_score:.1f})")

        # 需要至少一个信号 AND 不是在已有持仓的情况下重复信号
        if score >= 0.30 and (score >= 0.60 or len(reasons) >= 2):
            return Signal(
                symbol=symbol,
                action='BUY',
                reason='+'.join(reasons),
                score=score,
            )

        return None

    def _get_prev_row(self, date: str, symbol: str, offset: int = 1) -> Optional[pd.Series]:
        """获取前 N 天的指标数据"""
        try:
            dt = pd.to_datetime(date)
            # 在 all_dates 中找前一个交易日
            idx = self.all_dates.index(date) if date in self.all_dates else -1
            if idx > offset - 1:
                prev_date = self.all_dates[idx - offset]
                return self._indicator_index.get((prev_date, symbol))
        except:
            pass
        return None

    def _detect_platform_breakout(self, date: str, symbol: str, current_close: float) -> bool:
        """检测平台突破 — 横盘整理后放量突破

        条件:
        1. 过去 platform_min_days 日内价格振幅 < platform_max_amplitude (横盘)
        2. 今日收盘突破平台高点
        3. 成交量放大 > breakout_vol_ratio
        """
        p = self.params
        row = self._get_row(date, symbol)
        if row is None:
            return False

        bb_pct_b = row.get('bb_pct_b', 0)
        vol_ratio = row.get('vol_ratio', 0)
        if pd.isna(bb_pct_b) or pd.isna(vol_ratio):
            return False

        # 今日必须接近/突破布林上轨
        if bb_pct_b < 0.75:
            return False

        # 成交量确认
        if vol_ratio < p.breakout_vol_ratio:
            return False

        # 检查过去 platform_min_days 是否横盘
        # 方法：检查 BB 带宽在过去 N 天是否持续收窄
        narrow_count = 0
        for offset in range(1, p.platform_min_days + 1):
            prev = self._get_prev_row(date, symbol, offset)
            if prev is not None:
                bw = prev.get('bb_bandwidth', 1)
                if not pd.isna(bw) and bw < 0.20:
                    narrow_count += 1

        # 至少有一半天数带宽收窄
        if narrow_count < p.platform_min_days // 2:
            return False

        return True

    def _detect_n_pattern(self, date: str, symbol: str, current_close: float) -> bool:
        """检测 N 型反转 — 涨→回调→再创新高

        条件:
        1. 20日前的价格比现在低 > n_first_leg_min (第一波涨)
        2. 期间有至少一次 5% 以上的回调
        3. 当前创 20 日新高
        """
        p = self.params
        row = self._get_row(date, symbol)
        if row is None:
            return False

        # 当前必须创新高
        ma5 = row.get('ma5', 0)
        if pd.isna(ma5) or current_close < ma5 * 1.01:
            return False

        # 检查 20 日前价格（第一波起点）
        prev20 = self._get_prev_row(date, symbol, 20)
        if prev20 is None:
            return False
        close_20d_ago = prev20.get('close', 0)
        if pd.isna(close_20d_ago) or close_20d_ago <= 0:
            return False

        first_leg_ret = current_close / close_20d_ago - 1
        if first_leg_ret < p.n_first_leg_min:
            return False

        # 检查期间是否有明显回调（从期间高点回落 > 5%）
        max_close = current_close
        had_pullback = False
        for offset in range(1, 20):
            prev = self._get_prev_row(date, symbol, offset)
            if prev is not None:
                c = prev.get('close', 0)
                if not pd.isna(c) and c > 0:
                    if c > max_close:
                        max_close = c
                    # 检查是否从局部高点回调超过 5%
                    if max_close > 0 and (max_close - c) / max_close > 0.05:
                        had_pullback = True

        return had_pullback and first_leg_ret >= p.n_first_leg_min

    def _detect_vcp(self, date: str, symbol: str, current_close: float) -> bool:
        """检测 VCP 波动收缩 — 波动率逐波递减后放量突破

        条件:
        1. 当前 BB 带宽 < 0.15 (收缩到位)
        2. 过去 60 天 BB 带宽总体递减趋势
        3. %B 在 0.5 以上（价格偏强）
        """
        p = self.params
        row = self._get_row(date, symbol)
        if row is None:
            return False

        bb_bw = row.get('bb_bandwidth', 1)
        bb_pct_b = row.get('bb_pct_b', 0)
        if pd.isna(bb_bw) or pd.isna(bb_pct_b):
            return False

        # 当前必须收缩
        if bb_bw > 0.15:
            return False

        # 价格不能太弱
        if bb_pct_b < 0.5:
            return False

        # 检查波动率递减趋势：30天前带宽 > 15天前带宽 > 当前带宽
        bw_30 = self._get_prev_row(date, symbol, 30)
        bw_15 = self._get_prev_row(date, symbol, 15)
        bw_30_val = bw_30.get('bb_bandwidth', 0) if bw_30 is not None else 0
        bw_15_val = bw_15.get('bb_bandwidth', 0) if bw_15 is not None else 0

        if pd.isna(bw_30_val) or pd.isna(bw_15_val):
            return False

        # 波动率递减
        if not (bw_30_val > bw_15_val > bb_bw):
            return False

        return True

    def _sector_resonance(self, date: str, symbol: str) -> float:
        """板块共振得分 — 同板块其他股票是否也触发信号"""
        if not self.sector_map:
            return 0.0

        # 查找该股票所属板块
        symbol_sectors = []
        for sector, symbols in self.sector_map.items():
            if symbol in symbols:
                symbol_sectors.append(sector)

        if not symbol_sectors:
            return 0.0

        # 计算同板块有多少股票也通过趋势筛选
        p = self.params
        total_peers = 0
        for sector in symbol_sectors:
            peers = self.sector_map.get(sector, [])
            for peer in peers:
                if peer == symbol:
                    continue
                passed, _ = self._trend_filter(date, peer)
                if passed:
                    total_peers += 1

        if total_peers >= p.sector_min_peers:
            return min(1.0, total_peers / 10.0)

        return 0.0

    # ════════════════════════════════════════════════════
    # Layer 3: 风控出场
    # ════════════════════════════════════════════════════

    def _exit_check(self, date: str, pos: Position) -> Optional[Signal]:
        """检查持仓是否需要退出

        Returns: Signal if exit triggered, None if hold
        """
        p = self.params
        row = self._get_row(date, pos.symbol)
        if row is None:
            return Signal(pos.symbol, 'SELL', 'no_data')

        close = row['close']
        if pd.isna(close):
            return None

        atr = row.get('atr_14', 0)

        # 计算持仓天数
        try:
            days_held = (pd.to_datetime(date) - pd.to_datetime(pos.entry_date)).days
        except:
            days_held = 999
        in_min_hold = days_held < p.min_hold_days

        # ── 硬止损 (任何时间触发) ──
        if close <= pos.stop_loss_price:
            return Signal(pos.symbol, 'SELL', f'hard_stop({close:.2f}<={pos.stop_loss_price:.2f})')

        # ── 更新最高价 ──
        if close > pos.highest_close:
            pos.highest_close = close

        # ── 移动止损 (仅在最低持有期后激活) ──
        if not in_min_hold:
            if not pos.trailing_active:
                profit_pct = close / pos.entry_price - 1
                if profit_pct >= p.trailing_start_pct:
                    pos.trailing_active = True
                    pos.trailing_stop_price = pos.highest_close * (1 - p.trailing_atr_mult * atr / close) if atr > 0 else pos.highest_close * 0.95
            else:
                new_stop = pos.highest_close * (1 - p.trailing_atr_mult * atr / close) if atr > 0 and close > 0 else pos.highest_close * 0.95
                if new_stop > pos.trailing_stop_price:
                    pos.trailing_stop_price = new_stop
                if close <= pos.trailing_stop_price:
                    return Signal(pos.symbol, 'SELL', f'trailing_stop({close:.2f}<={pos.trailing_stop_price:.2f})')

        # ── 时间止损 (仅在最低持有期后激活) ──
        if not in_min_hold:
            if days_held >= p.time_stop_days:
                profit_pct = (close / pos.entry_price - 1)
                if profit_pct < 0.02:
                    return Signal(pos.symbol, 'SELL', f'time_stop({days_held}d, {profit_pct:.1%})')

        # ── 趋势反转 (仅在最低持有期后激活) ──
        if not in_min_hold:
            passed, reason = self._trend_filter(date, pos.symbol)
            if not passed:
                return Signal(pos.symbol, 'SELL', f'trend_break({reason})')

        return None  # 持有

    # ════════════════════════════════════════════════════
    # 每日信号生成
    # ════════════════════════════════════════════════════

    def check_signals(
        self, date: str, regime: str = 'SIDEWAYS'
    ) -> tuple[list[Signal], list[Signal]]:
        """生成当日信号

        Args:
            date: YYYY-MM-DD
            regime: 市场状态 (BULL/BEAR/SIDEWAYS)

        Returns:
            (entry_signals, exit_signals)
        """
        p = self.params
        entries: list[Signal] = []
        exits: list[Signal] = []

        # ── 先检查出场 (不删除持仓，由调用方处理) ──
        for sym, pos in self.positions.items():
            exit_sig = self._exit_check(date, pos)
            if exit_sig:
                exits.append(exit_sig)

        # ── 入场 ──
        # BEAR 市限制入场
        max_new = p.max_positions - len(self.positions)
        if max_new <= 0:
            return entries, exits
        if regime == 'BEAR':
            max_new = max(1, int(max_new * (1 - p.bear_regime_cash_pct)))

        candidates = []
        for sym in self.all_symbols:
            if sym in self.positions:
                continue

            passed, _ = self._trend_filter(date, sym)
            if not passed:
                continue

            sig = self._entry_signal(date, sym)
            if sig:
                candidates.append(sig)

        # 按评分排序，取前 max_new 个
        candidates.sort(key=lambda s: s.score, reverse=True)
        entries = candidates[:max_new]

        return entries, exits

    def add_position(self, symbol: str, date: str, entry_price: float, shares: int):
        """记录入场"""
        row = self._get_row(date, symbol)
        atr = row.get('atr_14', 0) if row is not None else 0
        close = entry_price

        p = self.params
        # 止损价: min(-5%硬止损, ATR×2止损)
        hard_stop = close * (1 + p.stop_loss_pct)
        atr_stop = close * (1 - p.stop_loss_atr_mult * atr / close) if atr > 0 and close > 0 else hard_stop
        stop_price = min(hard_stop, atr_stop)  # 取更紧的

        self.positions[symbol] = Position(
            symbol=symbol,
            entry_date=date,
            entry_price=entry_price,
            shares=shares,
            stop_loss_price=stop_price,
            highest_close=close,
        )

    def remove_position(self, symbol: str):
        """记录出场"""
        self.positions.pop(symbol, None)
