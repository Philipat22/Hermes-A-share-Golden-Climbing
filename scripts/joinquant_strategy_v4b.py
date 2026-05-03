"""
JoinQuant A-Share Surge Strategy — V4b Adaptation
==================================================
基于本地AI Hedge Fund的V4b策略移植版。

核心逻辑:
1. 大盘择时 (CSI300 MA系统 → 5种regime)
2. 简化Alpha因子评分 (基于LightGBM top-40因子权重)
3. V4执行Guard (个股冷却期、连续止损降仓、波动率调门槛)
4. 5天持有期 + -5%止损
5. 每日14:50交易

移植说明:
- 本地使用LightGBM + 160个Alpha因子做评分
- 聚宽环境无法跑LGBM，改用加权简化因子评分
- 因子权重从本地模型feature importance提取
"""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from jqdata import *
from jqlib.technical_analysis import *

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

# ── 因子权重 (从 LightGBM model feature importance 提取, 归一化) ──
# 仅包含聚宽可计算的Alpha 158型时间序列因子
FACTOR_WEIGHTS = {
    # K线基础因子
    "klen":    0.0682,   # (high - low) / open
    "klow":    0.0050,   # (open - low) / open  (近似)
    # 波动率因子
    "std_10":  0.0602,   # std(close,10) / close
    "std_20":  0.0341,   # std(close,20) / close
    "std_30":  0.0301,   # std(close,30) / close
    "std_60":  0.0377,   # std(close,60) / close
    # 动量/反转因子
    "roc_30":  0.0460,   # close_{t-30} / close  (高值=近期下跌)
    "roc_60":  0.0383,   # close_{t-60} / close
    # 价格位置因子
    "min_60":  0.0397,   # min(low,60) / close
    "max_60":  0.0298,   # max(high,60) / close
    "min_5":   0.0288,   # min(low,5) / close
    "max_5":   0.0214,   # max(high,5) / close
    # 趋势强度因子 (线性回归R²)
    "rsqr_60": 0.0326,   # close 60天R² (高值=强趋势)
    "rsqr_30": 0.0297,   # close 30天R²
    # 斜率/贝塔因子
    "beta_60": 0.0300,   # slope(close,60) / close
    "resi_30": 0.0207,   # 线性回归残差/close
    # 极值位置因子
    "imax_60": 0.0178,   # argmax(high,60) / 60
    "imax_30": 0.0123,   # argmax(high,30) / 30
}
FACTOR_WEIGHTS_TOTAL = sum(FACTOR_WEIGHTS.values())

# ── Regime 参数 ──
REGIME_CONFIG = {
    'bull':        {'model': '5d', 'threshold': 0.30, 'max_pos': 5, 'capacity': 0.80},
    'sideways':    {'model': '5d', 'threshold': 0.40, 'max_pos': 5, 'capacity': 0.60},
    'bear':        {'model': '5d', 'threshold': 0.35, 'max_pos': 3, 'capacity': 0.30},
    'severe_bear': {'model': '5d', 'threshold': 0.45, 'max_pos': 2, 'capacity': 0.15},
    'recovery':    {'model': '5d', 'threshold': 0.50, 'max_pos': 0, 'capacity': 0.00},
}

# ── V4 Guard 参数 ──
COOLDOWN_DAYS = 20
MAX_CONSECUTIVE_STOP_DAYS = 3
VOLA_WINDOW = 5
VOLA_HOLD_WARN = 3
VOLA_HOLD_CRITICAL = 2
STOP_LOSS = -0.05
HOLDING_DAYS = 5
ON_HIGH_THRESHOLD = 0.10  # 波动调门槛增量

# ── 交易成本 ──
COMMISSION_RATE = 0.0003  # 万3
STAMP_TAX_RATE = 0.0005   # 万5印花税
MIN_COMMISSION = 5         # 最低佣金5元

# ── 交易时间 ──
TRADE_TIME = '14:50'

# ══════════════════════════════════════════════════════════════════════
# 股票池
# ══════════════════════════════════════════════════════════════════════

# 核心板块股票池 (约500只, 基于之前16板块 + 申万行业精选)
CORE_STOCKS = [
    # === 银行 (12) ===
    '601398.XSHG', '601939.XSHG', '601288.XSHG', '601988.XSHG', '600036.XSHG',
    '600016.XSHG', '601166.XSHG', '600000.XSHG', '002142.XSHE', '601009.XSHG',
    '600015.XSHG', '601818.XSHG',
    # === 非银金融 (8) ===
    '601318.XSHG', '601628.XSHG', '601601.XSHG', '601336.XSHG',
    '600030.XSHG', '601211.XSHG', '600837.XSHG', '000776.XSHE',
    # === 半导体 (15) ===
    '603501.XSHG', '002371.XSHE', '600703.XSHG', '688981.XSHG', '603986.XSHG',
    '300661.XSHE', '002049.XSHE', '600584.XSHG', '300782.XSHE', '688012.XSHG',
    '688008.XSHG', '300604.XSHE', '002156.XSHE', '300474.XSHE', '600460.XSHG',
    # === 消费电子 (8) ===
    '002475.XSHE', '601138.XSHG', '002241.XSHE', '300433.XSHE', '002600.XSHE',
    '002384.XSHE', '603160.XSHG', '002850.XSHE',
    # === 白酒/食品饮料 (10) ===
    '600519.XSHG', '000568.XSHE', '000858.XSHE', '002304.XSHE', '600809.XSHG',
    '000596.XSHE', '603369.XSHG', '600887.XSHG', '002714.XSHE', '300015.XSHE',
    # === 医药 (12) ===
    '600276.XSHG', '300760.XSHE', '002007.XSHE', '300122.XSHE', '000661.XSHE',
    '300529.XSHE', '300347.XSHE', '603259.XSHG', '600196.XSHG', '000538.XSHE',
    '300015.XSHE', '002821.XSHE',
    # === 新能源 (12) ===
    '300750.XSHE', '002594.XSHE', '300274.XSHE', '601012.XSHG', '600438.XSHG',
    '300124.XSHE', '002812.XSHE', '300763.XSHE', '688599.XSHG', '300450.XSHE',
    '002459.XSHE', '300014.XSHE',
    # === 军工 (8) ===
    '600760.XSHG', '600893.XSHG', '002179.XSHE', '600862.XSHG', '000768.XSHE',
    '600118.XSHG', '600185.XSHG', '600150.XSHG',
    # === 机械设备 (10) ===
    '600406.XSHG', '300124.XSHE', '601100.XSHG', '600031.XSHG', '000988.XSHE',
    '300457.XSHE', '002353.XSHE', '300308.XSHE', '601877.XSHG', '603338.XSHG',
    # === 通信/计算机 (12) ===
    '000063.XSHE', '600941.XSHG', '300502.XSHE', '002230.XSHE', '300451.XSHE',
    '600570.XSHG', '002410.XSHE', '300454.XSHE', '300624.XSHE', '688111.XSHG',
    '002916.XSHE', '300033.XSHE',
    # === 汽车 (8) ===
    '600104.XSHG', '000625.XSHE', '601238.XSHG', '600741.XSHG', '002920.XSHE',
    '601689.XSHG', '600660.XSHG', '601633.XSHG',
    # === 家电 (6) ===
    '000333.XSHE', '000651.XSHE', '002032.XSHE', '002242.XSHE', '000100.XSHE',
    '002508.XSHE',
    # === 房地产 (6) ===
    '600048.XSHG', '000002.XSHE', '001979.XSHE', '600383.XSHG', '600606.XSHG',
    '000069.XSHE',
    # === 化工/有色 (10) ===
    '600309.XSHG', '002601.XSHE', '600585.XSHG', '603260.XSHG', '600010.XSHG',
    '000830.XSHE', '600019.XSHG', '002460.XSHE', '601899.XSHG', '603993.XSHG',
    # === 公用事业/电力 (8) ===
    '600900.XSHG', '601985.XSHG', '600886.XSHG', '600011.XSHG', '600023.XSHG',
    '600025.XSHG', '600905.XSHG', '601868.XSHG',
    # === 航空/物流 (6) ===
    '601111.XSHG', '600029.XSHG', '600115.XSHG', '002352.XSHE', '601006.XSHG',
    '601919.XSHG',
]

# ══════════════════════════════════════════════════════════════════════
# INITIALIZE
# ══════════════════════════════════════════════════════════════════════

def initialize(context):
    """策略初始化"""
    # 设置股票池
    g.stocks = CORE_STOCKS
    g.index_code = '000300.XSHG'  # CSI300
    
    # 交易设置
    g.trade_time = TRADE_TIME
    
    # V4 Guard 状态
    g.stop_loss_cooldown = {}       # {symbol: cooldown_until_date}
    g.consecutive_stop_days = 0      # 连续止损天数
    g.recent_stop_hold_times = []    # 最近止损持股天数(用于波动率调门槛)
    g.entry_dates = {}              # {symbol: entry_date}
    g.sell_dates = {}               # {symbol: sell_date}
    g.entry_prices = {}             # {symbol: entry_price}
    
    # 每日运行
    run_daily(handle_daily, time=TRADE_TIME)


# ══════════════════════════════════════════════════════════════════════
# REGIME CLASSIFIER
# ══════════════════════════════════════════════════════════════════════

def classify_regime(context):
    """CSI300 MA大盘择时分类器"""
    df = get_price(g.index_code, 
                   end_date=context.current_dt.date(),
                   count=130, 
                   fields=['close'],
                   skip_paused=True, 
                   fq='pre')
    
    if df is None or len(df) < 60:
        return 'sideways'
    
    closes = df['close'].values
    c = closes[-1]
    
    ma20 = np.mean(closes[-20:])
    ma60 = np.mean(closes[-60:])
    ma120 = np.mean(closes[-120:]) if len(closes) >= 120 else ma60
    
    # 20日收益
    ret20 = c / closes[-21] - 1 if len(closes) >= 21 else 0
    
    if c < ma120 * 0.90:
        return 'severe_bear'
    if c < ma60 and c < ma120 and ret20 < -0.03:
        return 'bear'
    if c > ma20 > ma60 and ret20 > 0.03:
        return 'bull'
    if c > ma60 and ret20 > 0:
        return 'recovery'
    return 'sideways'


# ══════════════════════════════════════════════════════════════════════
# FACTOR COMPUTATION
# ══════════════════════════════════════════════════════════════════════

def compute_factors(stock, context, lookback=60):
    """计算单一股票的简化Alpha因子值"""
    df = get_price(stock,
                   end_date=context.current_dt.date(),
                   count=lookback + 10,  # 多取一些确保窗口完整
                   fields=['open', 'high', 'low', 'close', 'volume', 'money'],
                   skip_paused=True,
                   fq='pre')
    
    if df is None or len(df) < 30:
        return None
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    opn = df['open'].values
    volume = df['volume'].values
    n = len(close)
    
    factors = {}
    
    # ── K线基础因子 ──
    klen_val = (high[-1] - low[-1]) / opn[-1] if opn[-1] != 0 else 0
    factors['klen'] = klen_val
    
    klow_val = (opn[-1] - low[-1]) / opn[-1] if opn[-1] != 0 else 0
    factors['klow'] = klow_val
    
    # ── 波动率因子 (std_N) ──
    for N in [10, 20, 30, 60]:
        if n >= N + 1:
            std_val = np.std(close[-N:]) / close[-1]
        else:
            std_val = 0
        factors[f'std_{N}'] = std_val
    
    # ── 动量/反转因子 (roc_N = close_{t-N} / close_t) ──
    for N in [30, 60]:
        if n >= N + 1:
            roc_val = close[-N-1] / close[-1]
        else:
            roc_val = 1.0
        factors[f'roc_{N}'] = roc_val
    
    # ── 价格位置因子 ──
    for N in [5, 60]:
        if n >= N + 1:
            min_val = np.min(low[-N:]) / close[-1]
            max_val = np.max(high[-N:]) / close[-1]
        else:
            min_val = close[-1] / close[-1]
            max_val = close[-1] / close[-1]
        factors[f'min_{N}'] = min_val
        factors[f'max_{N}'] = max_val
    
    # ── 趋势强度 (线性回归R²) ──
    for N in [30, 60]:
        if n >= N + 2:
            x = np.arange(N)
            y = close[-N:]
            A = np.vstack([x, np.ones(N)]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
            fitted = slope * x + intercept
            ss_res = np.sum((y - fitted) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            rsqr = ss_res / ss_tot if ss_tot > 0 else 0  # R² (越小越好,拟合优度)
            rsqr = 1 - rsqr  # 转为标准R²
            beta_val = slope / close[-1]
            
            residuals = y - fitted
            resi_val = np.mean(residuals[-3:]) / close[-1] if N >= 3 else residuals[-1] / close[-1]
        else:
            rsqr = 0.5
            beta_val = 0
            resi_val = 0
        factors[f'rsqr_{N}'] = max(0, min(1, rsqr))
        factors[f'beta_{N}'] = beta_val
        factors[f'resi_{N}'] = resi_val
    
    # ── 极值位置因子 ──
    for N in [30, 60]:
        if n >= N + 1:
            window_high = high[-N:]
            argmax = np.argmax(window_high)
            imax_val = argmax / N
        else:
            imax_val = 0.5
        factors[f'imax_{N}'] = imax_val
    
    return factors


def compute_stock_score(factors, regime):
    """计算加权因子评分"""
    if factors is None:
        return 0.0
    
    score = 0.0
    total_weight = 0.0
    
    for factor_name, weight in FACTOR_WEIGHTS.items():
        if factor_name in factors:
            val = factors[factor_name]
            
            # ── 因子方向调整 (与模型预测方向对齐) ──
            # std_N: 高波动=机会(均值回归)
            if factor_name.startswith('std_'):
                val = val * 10  # 放大到可比量纲
                # 归一化: 截断到[0,2]范围
                val = min(val, 2.0)
            
            # roc_N: 高=近期跌(均值回归买点)
            elif factor_name.startswith('roc_'):
                # roc范围约0.7-1.3, 转为[-0.3, +0.3] 
                val = (val - 1.0) * 3
                # 方向: 高roc(近期下跌)=高分 → 直接取
    
            # min_N / max_N: 接近近期低点=抄底机会
            elif factor_name.startswith('min_'):
                val = (1.0 - val)  # min越小=越靠近低点=高分
                val = val * 2
            elif factor_name.startswith('max_'):
                val = (1.0 - val)  # max越小=越没到高点=还有空间
            
            # rsqr_N: 趋势越弱越好(均值回归)
            elif factor_name.startswith('rsqr_'):
                val = 1.0 - val  # 弱趋势=高分
            
            # imax_N: 近期高点越远越好
            elif factor_name.startswith('imax_'):
                val = 1.0 - val
            
            # beta: 正beta与大盘同向
            elif factor_name.startswith('beta_'):
                val = -val  # 负beta=与大盘反向(均值回归机会)
                val = np.clip(val, -2, 2)
            
            # klen/klow: 大K线=大波动=机会
            elif factor_name == 'klen':
                val = val * 5  # 放大klen值(通常在0.01-0.05)
                val = min(val, 1.5)
            
            elif factor_name == 'klow':
                val = val * 5
                val = min(val, 1.5)
            
            # resi_30: 残差
            elif factor_name.startswith('resi_'):
                val = val * 20
                val = np.clip(val, -2, 2)
            
            score += weight * val
            total_weight += weight
    
    # 归一化到[0, 1]区间
    if total_weight > 0:
        score = score / total_weight
    
    # sigmoid挤压到(0,1)
    score = 1.0 / (1.0 + np.exp(-score * 2))
    
    return score


# ══════════════════════════════════════════════════════════════════════
# TRADING LOGIC
# ══════════════════════════════════════════════════════════════════════

def handle_daily(context):
    """每日交易主逻辑 (14:50执行)"""
    current_date = context.current_dt.date()
    log.info(f"[{current_date}] === Daily Run ===")
    
    # 1. 判定大盘regime
    regime = classify_regime(context)
    log.info(f"  Regime: {regime}")
    cfg = REGIME_CONFIG.get(regime, REGIME_CONFIG['sideways'])
    threshold = cfg['threshold']
    max_pos = cfg['max_pos']
    capacity = cfg['capacity']
    
    # 2. V4 Guard: 波动率调门槛
    if len(g.recent_stop_hold_times) >= VOLA_WINDOW:
        avg_hold = np.mean(g.recent_stop_hold_times[-VOLA_WINDOW:])
        if avg_hold < VOLA_HOLD_CRITICAL:
            threshold += ON_HIGH_THRESHOLD * 2
            log.info(f"  Vola guard (crit): threshold +{ON_HIGH_THRESHOLD*2:.2f} → {threshold:.2f}")
        elif avg_hold < VOLA_HOLD_WARN:
            threshold += ON_HIGH_THRESHOLD
            log.info(f"  Vola guard (warn): threshold +{ON_HIGH_THRESHOLD:.2f} → {threshold:.2f}")
    
    # 3. V4 Guard: 连续止损降仓
    effective_max_pos = max_pos
    effective_capacity = capacity
    if g.consecutive_stop_days >= 5:
        effective_max_pos = max(1, max_pos - 2)
        effective_capacity = capacity * 0.5
        log.info(f"  Consecutive guard (severe): max_pos {max_pos}→{effective_max_pos}, cap {capacity}→{effective_capacity}")
    elif g.consecutive_stop_days >= MAX_CONSECUTIVE_STOP_DAYS:
        effective_max_pos = max(1, max_pos - 1)
        effective_capacity = capacity * 0.75
        log.info(f"  Consecutive guard: max_pos {max_pos}→{effective_max_pos}, cap {capacity}→{effective_capacity}")
    
    # 4. 清理过期冷却期
    g.stop_loss_cooldown = {
        k: v for k, v in g.stop_loss_cooldown.items() 
        if v >= current_date
    }
    
    # 5. 处理持仓 → 卖出
    today_had_stop = False
    today_winners = 0
    
    for stock in list(context.portfolio.positions.keys()):
        position = context.portfolio.positions[stock]
        entry_price = g.entry_prices.get(stock)
        entry_date = g.entry_dates.get(stock)
        
        if entry_price is None or entry_date is None:
            continue
        
        # 检查止损
        current_price = position.price
        dd = current_price / entry_price - 1
        
        if dd <= STOP_LOSS:
            # 止损卖出
            order_target_value(stock, 0)
            sell_date = current_date
            hold_days = (sell_date - entry_date).days if entry_date else 0
            
            log.info(f"  STOP {stock}: entry={entry_price:.2f}, exit={current_price:.2f}, "
                     f"ret={dd:.2%}, hold={hold_days}d")
            
            # 冷却期
            g.stop_loss_cooldown[stock] = current_date + pd.Timedelta(days=COOLDOWN_DAYS)
            today_had_stop = True
            g.recent_stop_hold_times.append(hold_days)
            if len(g.recent_stop_hold_times) > VOLA_WINDOW * 2:
                g.recent_stop_hold_times = g.recent_stop_hold_times[-VOLA_WINDOW * 2:]
            
        elif entry_date and (current_date - entry_date).days >= HOLDING_DAYS:
            # 到期卖出
            order_target_value(stock, 0)
            ret = current_price / entry_price - 1
            log.info(f"  MATURED {stock}: entry={entry_price:.2f}, exit={current_price:.2f}, ret={ret:.2%}")
            if ret > 0:
                today_winners += 1
    
    # 6. 更新连续止损计数
    if today_had_stop:
        g.consecutive_stop_days += 1
    elif today_winners > 0:
        g.consecutive_stop_days = 0
    
    # 7. 如果recovery或severe_bear, 不开新仓
    if max_pos == 0:
        log.info(f"  No new positions in {regime} regime")
        return
    
    # 8. 选股 → 买入
    open_slots = max(0, effective_max_pos - len(context.portfolio.positions))
    if open_slots <= 0:
        log.info(f"  No open slots ({len(context.portfolio.positions)}/{effective_max_pos})")
        return
    
    # 计算可用资金
    available = context.portfolio.available_cash * effective_capacity
    per_stock = available / open_slots if open_slots > 0 else 0
    
    # 过滤: 排除已有持仓、冷却期内的股票
    candidates = []
    positions = set(context.portfolio.positions.keys())
    
    for stock in g.stocks:
        if stock in positions or stock in g.stop_loss_cooldown:
            continue
        # 检查是否可交易
        try:
            df = get_price(stock, end_date=current_date, count=2, 
                          fields=['close', 'paused'], skip_paused=True)
            if df is None or len(df) < 1:
                continue
            current_price = df['close'].iloc[-1]
            if current_price <= 0:
                continue
        except:
            continue
        
        factors = compute_factors(stock, context)
        score = compute_stock_score(factors, regime)
        candidates.append((stock, score, current_price))
    
    if not candidates:
        log.info("  No candidates")
        return
    
    # 排序取top
    candidates.sort(key=lambda x: -x[1])
    selected = [(s, p, sc) for s, sc, p in candidates if sc >= threshold][:open_slots]
    
    log.info(f"  Candidates: {len(candidates)} (threshold={threshold:.2f})")
    
    for stock, score, price in selected[:open_slots]:
        # 100股整数倍
        shares = int(per_stock / (price * 100)) * 100
        if shares < 100:
            continue
        
        order(stock, shares)
        g.entry_dates[stock] = current_date
        g.sell_dates[stock] = current_date + pd.Timedelta(days=HOLDING_DAYS)
        g.entry_prices[stock] = price
        
        log.info(f"  BUY {stock}: price={price:.2f}, shares={shares}, score={score:.3f}, "
                 f"regime={regime}, threshold={threshold:.2f}")
    
    # 打印持仓状态
    log.info(f"  Positions: {len(context.portfolio.positions)}/{effective_max_pos}")
    log.info(f"  Portfolio: {context.portfolio.portfolio_value:.0f}")
