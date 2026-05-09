"""
19位投资大师规则引擎 v2.0 — 完整版(量价+基本面+消息面)

每位大师输出: {vote: bool, score: 0-100, reason: str, group: str}

参数: 0 — 所有阈值来自大师方法论,非回测优化
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


# ══════════════════════════════════════════════════════
#  工具
# ══════════════════════════════════════════════════════

def _get_fund(df: pd.DataFrame, fund: Optional[dict]) -> dict:
    """提取基本面数据,无数据时返回默认值

    字段来源:
      fundamentals_daily: pe, pb, pe_ttm, total_mv
      fina_indicator:     roe_yearly(优先), roe, roa, roic, grossprofit_margin, netprofit_margin,
                          or_yoy, op_yoy, netprofit_yoy, equity_yoy, bps_yoy, basic_eps_yoy,
                          debt_to_assets, debt_to_eqt, current_ratio, quick_ratio,
                          q_ocf_to_sales, profit_dedt, eps, bps, ocfps, assets_turn
    兼容旧 roe_data.pkl: roe 字段(季度)自动年化
    """
    def _f(key, default=-999):
        val = fund.get(key) if fund else None
        if val is None or (isinstance(val, float) and np.isnan(val)): return default
        try: return float(val)
        except: return default

    if fund is None:
        return {
            'pe': 999, 'pb': 999, 'pe_ttm': 999, 'pb_ttm': 999, 'mcap': 0,
            'roe': -999, 'roa': -999, 'roe_yearly': -999, 'roic': -999,
            'gross_margin': -1, 'net_margin': -1,
            'or_yoy': -999, 'op_yoy': -999, 'netprofit_yoy': -999,
            'equity_yoy': -999, 'bps_yoy': -999, 'basic_eps_yoy': -999,
            'debt_to_assets': -1, 'debt_to_eqt': -1, 'current_ratio': -1, 'quick_ratio': -1,
            'q_ocf_to_sales': -999, 'profit_dedt': -999,
            'eps': -999, 'bps': -999, 'ocfps': -999, 'assets_turn': -1,
        }

    # ROE: 优先 roe_yearly(已年化), 回退 roe(季度×4)
    roe_annual = _f('roe_yearly')
    if roe_annual <= -900:
        roe_raw = _f('roe')
        roe_annual = roe_raw * 4 if -900 < roe_raw < 900 else -999

    return {
        'pe': _f('pe', 999) if _f('pe', 999) < 999 else _f('pe_ttm', 999),
        'pb': _f('pb', 999) if _f('pb', 999) < 999 else _f('pb_ttm', 999),
        'pe_ttm': _f('pe_ttm', 999) if _f('pe_ttm', 999) < 999 else _f('pe', 999),
        'pb_ttm': _f('pb_ttm', 999) if _f('pb_ttm', 999) < 999 else _f('pb', 999),
        'mcap': _f('total_mv', 0) if _f('total_mv', 0) > 0 else _f('circ_mv', 0),
        # 盈利
        'roe': float(roe_annual) if roe_annual > -900 else -999,
        'roa': _f('roa'), 'roe_yearly': _f('roe_yearly'), 'roic': _f('roic'),
        'gross_margin': _f('grossprofit_margin', -1), 'net_margin': _f('netprofit_margin', -1),
        # 成长 (YoY %)
        'or_yoy': _f('or_yoy'), 'op_yoy': _f('op_yoy'),
        'netprofit_yoy': _f('netprofit_yoy'), 'equity_yoy': _f('equity_yoy'),
        'bps_yoy': _f('bps_yoy'), 'basic_eps_yoy': _f('basic_eps_yoy'),
        # 财务健康
        'debt_to_assets': _f('debt_to_assets', -1), 'debt_to_eqt': _f('debt_to_eqt', -1),
        'current_ratio': _f('current_ratio', -1), 'quick_ratio': _f('quick_ratio', -1),
        # 盈利质量
        'q_ocf_to_sales': _f('q_ocf_to_sales'), 'profit_dedt': _f('profit_dedt'),
        # 每股
        'eps': _f('eps'), 'bps': _f('bps'), 'ocfps': _f('ocfps'),
        # 效率
        'assets_turn': _f('assets_turn', -1),
    }


def _vol_arr(df: pd.DataFrame) -> np.ndarray:
    if 'vol' in df.columns: return df['vol'].values
    if 'volume' in df.columns: return df['volume'].values
    return np.ones(len(df))


# ══════════════════════════════════════════════════════
#  动量派 (4)
# ══════════════════════════════════════════════════════

def master_oneil(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """O'Neil CANSLIM: MA排列 + 相对强度 + 量确认 + 盈利增长"""
    closes = df['close'].values; vols = _vol_arr(df)
    if len(closes) < 120: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '动量'}
    
    c = closes[-1]; ma50 = np.mean(closes[-50:]); ma200 = np.mean(closes[-120:])
    rs_60d = c / closes[-60] - 1
    vol_expand = np.mean(vols[-10:]) / np.mean(vols[-50:])
    f = _get_fund(df, fund)
    
    score = 0; reasons = []
    if c > ma50 > ma200: score += 30; reasons.append('MA多头排列')
    if rs_60d > 0.10: score += 20; reasons.append('60日RS>10%')
    if vol_expand > 1.3: score += 20; reasons.append('放量')
    if c > ma50 * 1.02: score += 15; reasons.append('突破MA50')
    if f['pe_ttm'] < 50 and f['pe_ttm'] > 0: score += 15; reasons.append(f'PE{f["pe_ttm"]:.0f}(合理)')
    
    vote = score >= 60
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '动量'}


def master_livermore(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Livermore: 关键点突破 + 量确认 + 最小阻力"""
    closes = df['close'].values; highs = df['high'].values; vols = _vol_arr(df)
    if len(closes) < 60: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '动量'}
    
    c = closes[-1]; ma20 = np.mean(closes[-20:])
    resistance = np.max(highs[-40:-20]) if len(highs) >= 40 else np.max(highs)
    breakthrough = c > resistance * 0.98
    vol_ratio = vols[-1] / np.mean(vols[-20:])
    trend_20d = c / closes[-20] - 1
    
    score = 0; reasons = []
    if breakthrough: score += 35; reasons.append('接近关键阻力')
    if vol_ratio > 1.5: score += 30; reasons.append('放量突破')
    if trend_20d > 0.05: score += 20; reasons.append('最小阻力向上')
    if c > ma20: score += 15
    
    vote = score >= 65
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '动量'}


def master_darvas(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Darvas: 箱体突破"""
    closes = df['close'].values; highs = df['high'].values; lows = df['low'].values
    vols = _vol_arr(df)
    if len(closes) < 40: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '动量'}
    
    c = closes[-1]
    box_high = np.max(highs[-20:-1]); box_low = np.min(lows[-20:-1])
    box_range = (box_high - box_low) / box_low if box_low > 0 else 1
    breakthrough = c > box_high
    vol_ratio = vols[-1] / np.mean(vols[-20:])
    
    score = 0; reasons = []
    if box_range < 0.20: score += 20; reasons.append(f'窄幅箱体({box_range*100:.0f}%)')
    if breakthrough: score += 35; reasons.append('突破箱顶')
    if vol_ratio > 1.5: score += 30; reasons.append('放量')
    if c > np.mean(closes[-50:]): score += 15
    
    vote = score >= 60 and breakthrough
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '动量'}


def master_minervini(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Minervini SEPA: 趋势模板 + 收缩 + 放量"""
    closes = df['close'].values; highs = df['high'].values; lows = df['low'].values
    vols = _vol_arr(df)
    if len(closes) < 200: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '动量'}
    
    c = closes[-1]
    ma20 = np.mean(closes[-20:]); ma50 = np.mean(closes[-50:])
    ma150 = np.mean(closes[-150:]); ma200 = np.mean(closes[-200:])
    trend_ok = c > ma20 > ma50 > ma150 > ma200
    
    recent_range = np.mean([(highs[i]-lows[i])/closes[i] for i in range(-5, 0)])
    avg_range = np.mean([(highs[i]-lows[i])/closes[i] for i in range(-20, -5)])
    contraction = recent_range < avg_range * 0.8
    
    vol_ratio = vols[-1] / np.mean(vols[-50:])
    rs_60d = c / closes[-60] - 1
    
    score = 0; reasons = []
    if trend_ok: score += 40; reasons.append('趋势模板通过')
    if contraction: score += 25; reasons.append('振幅收缩')
    if vol_ratio > 1.3: score += 20; reasons.append('放量')
    if rs_60d > 0.15: score += 15; reasons.append('强势股')
    
    vote = score >= 70 and trend_ok
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '动量'}


# ══════════════════════════════════════════════════════
#  价值派 (3) — 基本面驱动
# ══════════════════════════════════════════════════════

def master_buffett(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Buffett: 好公司+好价格 — PE<20 + PB<3 + ROE>15% + 低负债 + 趋势企稳 + 深度回撤"""
    closes = df['close'].values; vols = _vol_arr(df)
    if len(closes) < 200: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '价值'}
    
    c = closes[-1]; ma200 = np.mean(closes[-200:])
    low_52w = np.min(closes[-200:]); high_52w = np.max(closes[-200:])
    discount = (high_52w - c) / high_52w
    f = _get_fund(df, fund)
    
    score = 0; reasons = []
    
    # 估值面 (A股: PE对收益无预测力, 降权)
    if 0 < f['pe_ttm'] < 30: score += 8; reasons.append(f'PE{f["pe_ttm"]:.0f}(不贵)')
    if 0 < f['pb_ttm'] < 3: score += 8; reasons.append(f'PB{f["pb_ttm"]:.1f}(低)')
    
    # 质量面 (保留: 造假检测仍有价值)
    if f['roe'] > 15: score += 15; reasons.append(f'ROE{f["roe"]:.0f}%(优)')
    elif f['roe'] > 10: score += 8; reasons.append(f'ROE{f["roe"]:.0f}%(良)')
    if 0 < f['debt_to_assets'] < 50: score += 8; reasons.append(f'负债率{f["debt_to_assets"]:.0f}%(低)')
    if f['q_ocf_to_sales'] > 0.1: score += 8; reasons.append('现金流真实')
    
    # 价格面 (加量价 — A股核心因子)
    if discount > 0.30: score += 12; reasons.append(f'深度回撤{discount*100:.0f}%')
    elif discount > 0.15: score += 6; reasons.append(f'回撤{discount*100:.0f}%')
    if c > ma200 * 0.90: score += 8; reasons.append('趋势企稳')
    # 量能确认 — A股最强信号之一
    vol_ratio = np.mean(vols[-5:]) / np.mean(vols[-60:]) if len(vols)>=60 else 1
    if vol_ratio > 1.5: score += 12; reasons.append('放量吸筹')
    # 超跌反弹 — A股最强因子
    ret_60d = c / closes[-60] - 1 if len(closes)>=60 else 0
    if ret_60d < -0.20: score += 12; reasons.append(f'超跌反弹({ret_60d:.0%})')
    
    # 质量+价格+量能共振
    if f['roe'] > 10 and discount > 0.15 and vol_ratio > 1.3:
        score += 12; reasons.append('质量+价格+资金共振')
    
    vote = score >= 55 and (f['pe_ttm'] < 999 or discount > 0.25)
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '价值'}


def master_graham(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Graham: 烟蒂 — PE<15 + PB<1.5 + 低负债 + 盈利 + 深度折价"""
    closes = df['close'].values; vols = _vol_arr(df)
    if len(closes) < 120: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '价值'}
    
    c = closes[-1]; high_120d = np.max(closes[-120:])
    discount = (high_120d - c) / high_120d
    f = _get_fund(df, fund)
    vol_ratio = np.mean(vols[-5:]) / np.mean(vols[-60:])
    
    score = 0; reasons = []
    
    # 烟蒂标准: 极低估值
    if 0 < f['pe_ttm'] < 15: score += 25; reasons.append(f'PE{f["pe_ttm"]:.0f}(烟蒂)')
    elif 0 < f['pe_ttm'] < 25: score += 12
    if 0 < f['pb_ttm'] < 1.5: score += 20; reasons.append(f'PB{f["pb_ttm"]:.1f}(破净附近)')
    
    # 质量门槛 (Graham: 负债率<50% + 流动比率>2 + 真实盈利)
    if 0 < f['debt_to_assets'] < 50: score += 12; reasons.append(f'负债率{f["debt_to_assets"]:.0f}%(安全)')
    if f['current_ratio'] > 2.0: score += 15; reasons.append(f'流动比率{f["current_ratio"]:.1f}(格雷厄姆标准)')
    elif f['current_ratio'] > 1.5: score += 8
    if f['roe'] > 0 and f['net_margin'] > 0:
        score += 10; reasons.append(f'盈利持续(ROE{f["roe"]:.0f}%)')
    if f['q_ocf_to_sales'] > 0: score += 10; reasons.append('现金流正向')
    if f['profit_dedt'] > 0: score += 8; reasons.append('扣非盈利')
    
    # 深度折价
    if discount > 0.40: score += 20; reasons.append(f'深度折价{discount*100:.0f}%')
    elif discount > 0.25: score += 12
    
    # 底部放量
    if vol_ratio > 1.5 and c < high_120d * 0.75: score += 12; reasons.append('底部放量')
    
    vote = score >= 55 and discount > 0.20
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '价值'}


def master_templeton(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Templeton: 极度悲观 — 连续下跌+止跌+估值合理 + 财务可存活 + 资产效率"""
    closes = df['close'].values; vols = _vol_arr(df)
    if len(closes) < 60: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '价值'}
    
    c = closes[-1]; f = _get_fund(df, fund)
    down_streak = sum(1 for i in range(-10, -1) if closes[i] < closes[i-1])
    dropped_20d = c < closes[-20] * 0.90
    yang = c > df['open'].values[-1] if 'open' in df.columns else True
    vol_surge = vols[-1] > np.mean(vols[-20:]) * 1.5
    
    score = 0; reasons = []
    if down_streak >= 5: score += 20; reasons.append(f'连跌{down_streak}日')
    if dropped_20d: score += 22; reasons.append('深度回调')
    if yang and vol_surge: score += 22; reasons.append('放量止跌')
    if 0 < f['pe_ttm'] < 30: score += 15; reasons.append(f'PE{f["pe_ttm"]:.0f}(不贵)')
    
    # 困境反转需要: 不会破产 + 有反转基础
    if f['current_ratio'] > 1.5: score += 12; reasons.append(f'流动比率{f["current_ratio"]:.1f}(不会破产)')
    if f['q_ocf_to_sales'] > 0: score += 10; reasons.append('经营现金流正')
    if f['assets_turn'] > 0.3: score += 8; reasons.append('资产有效运转')
    
    vote = score >= 55
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '价值'}


# ══════════════════════════════════════════════════════
#  成长派 (3) — 基本面+动量
# ══════════════════════════════════════════════════════

def master_lynch(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Lynch: 合理价格成长 — PE合理 + ROE质量 + 净利率 + 趋势向上 + 未过度拉升"""
    closes = df['close'].values; vols = _vol_arr(df)
    if len(closes) < 120: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '成长'}
    
    c = closes[-1]; ma50 = np.mean(closes[-50:]); ma200 = np.mean(closes[-200:])
    ret_60d = c / closes[-60] - 1; ret_20d = c / closes[-20] - 1
    f = _get_fund(df, fund)
    
    # PEG代理: PE vs 60日涨幅(价格代理盈利增长)
    peg_proxy = f['pe_ttm'] / (ret_60d * 100 + 1) if ret_60d > 0 and f['pe_ttm'] > 0 else 999
    # Lynch质量: ROE/PE = 质量/价格的性价比
    roe_pe_ratio = f['roe'] / f['pe_ttm'] if f['roe'] > 0 and f['pe_ttm'] > 0 else 0
    
    score = 0; reasons = []
    if c > ma50 > ma200: score += 20; reasons.append('上升趋势')
    if 0.05 < ret_20d < 0.25: score += 15; reasons.append('温和上涨')
    if ret_60d < 0.50: score += 12; reasons.append('未过度拉升')
    if 0 < f['pe_ttm'] < 40: score += 10; reasons.append(f'PE{f["pe_ttm"]:.0f}')
    if peg_proxy < 2: score += 15; reasons.append(f'PEG≈{peg_proxy:.1f}(合理)')
    
    # Lynch质量维度 (ROE + 成长 + 现金流)
    if f['roe'] > 15: score += 12; reasons.append(f'ROE{f["roe"]:.0f}%(优质)')
    elif f['roe'] > 8: score += 6; reasons.append(f'ROE{f["roe"]:.0f}%')
    if f['net_margin'] > 10: score += 10; reasons.append(f'净利率{f["net_margin"]:.0f}%(高)')
    elif f['net_margin'] > 5: score += 5; reasons.append(f'净利率{f["net_margin"]:.0f}%')
    # ROE/PE > 0.3 = 性价比突出
    if roe_pe_ratio > 0.3: score += 8; reasons.append(f'性价比优秀(ROE/PE={roe_pe_ratio:.2f})')
    # Lynch最喜欢的: 营收增长 + 利润增长 + 现金流真实
    if f['or_yoy'] > 15: score += 15; reasons.append(f'营收+{f["or_yoy"]:.0f}%YoY(真成长)')
    elif f['or_yoy'] > 5: score += 8; reasons.append(f'营收+{f["or_yoy"]:.0f}%YoY')
    if f['netprofit_yoy'] > 10: score += 12; reasons.append(f'利润+{f["netprofit_yoy"]:.0f}%YoY')
    if f['q_ocf_to_sales'] > 0.05: score += 10; reasons.append('现金流验证成长')
    
    vote = score >= 55
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '成长'}


def master_fisher(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Fisher: 成长股 — 营收增长>10% + 利润增长 + 高利润率 + 合理估值 + 量价配合"""
    closes = df['close'].values; highs = df['high'].values; vols = _vol_arr(df)
    if len(closes) < 120: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '成长'}
    
    c = closes[-1]; f = _get_fund(df, fund)
    ret_60d = c / closes[-60] - 1
    price_up = closes[-1] > closes[-5]
    vol_up = vols[-1] > np.mean(vols[-10:])
    
    score = 0; reasons = []
    
    # 费雪核心: 营收增长 + 利润增长 (不是价格创新高)
    if f['or_yoy'] > 20: score += 30; reasons.append(f'营收+{f["or_yoy"]:.0f}%YoY(高增长)')
    elif f['or_yoy'] > 10: score += 20; reasons.append(f'营收+{f["or_yoy"]:.0f}%YoY')
    if f['netprofit_yoy'] > 15: score += 25; reasons.append(f'利润+{f["netprofit_yoy"]:.0f}%YoY')
    elif f['netprofit_yoy'] > 5: score += 15
    
    # 利润率 (高利润率 = 护城河)
    if f['gross_margin'] > 40: score += 15; reasons.append(f'毛利率{f["gross_margin"]:.0f}%(护城河)')
    elif f['gross_margin'] > 25: score += 8
    if f['net_margin'] > 10: score += 10; reasons.append(f'净利率{f["net_margin"]:.0f}%(优质)')
    
    # 量价配合 (技术确认)
    if price_up and vol_up: score += 15; reasons.append('量价配合')
    if ret_60d > 0.05: score += 10; reasons.append('中期向上')
    
    # 估值可接受 (费雪不买太贵的)
    if 0 < f['pe_ttm'] < 60: score += 10; reasons.append('估值可接受')
    
    # 有基本面数据才投票 (费雪是纯基本面大师)
    has_fundamentals = f['or_yoy'] > -900 or f['netprofit_yoy'] > -900 or f['gross_margin'] > 0
    
    vote = score >= 60 and has_fundamentals
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '成长'}


def master_wood(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Wood: 颠覆性创新 — 高波动+放量+突破 + 高毛利率(定价权) + 正净利率(商业验证)"""
    closes = df['close'].values; highs = df['high'].values; lows = df['low'].values
    vols = _vol_arr(df)
    if len(closes) < 60: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '成长'}
    
    c = closes[-1]; ret_5d = c / closes[-5] - 1
    range_60d = (np.max(highs[-60:]) - np.min(lows[-60:])) / np.mean(closes[-60:])
    vol_expand = np.mean(vols[-10:]) / np.mean(vols[-30:]) > 1.5
    
    score = 0; reasons = []
    if range_60d > 0.30: score += 25; reasons.append('高波动')
    if vol_expand: score += 22; reasons.append('放量')
    if ret_5d > 0.05: score += 20; reasons.append('短期突破')
    if c > np.mean(closes[-20:]): score += 18
    
    # 质量维度 (毛利率=定价权, 净利率=商业模式验证)
    f = _get_fund(df, fund)
    if f['gross_margin'] > 40: score += 15; reasons.append(f'毛利率{f["gross_margin"]:.0f}%(高定价权)')
    elif f['gross_margin'] > 25: score += 8; reasons.append(f'毛利率{f["gross_margin"]:.0f}%')
    if f['net_margin'] > 0: score += 10; reasons.append('盈利验证')
    
    vote = score >= 60
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '成长'}


# ══════════════════════════════════════════════════════
#  技术派 (3)
# ══════════════════════════════════════════════════════

def master_wyckoff(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Wyckoff: 回调缩量(吸筹) + 反弹放量"""
    closes = df['close'].values; vols = _vol_arr(df)
    if len(closes) < 60: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '技术'}
    
    c = closes[-1]
    ret_5d = c / closes[-5] - 1
    vol_5d = np.mean(vols[-5:]) / np.mean(vols[-20:])
    prev_up = closes[-10] > closes[-20]
    
    score = 0; reasons = []
    if -0.05 < ret_5d < 0.03 and vol_5d < 0.8:
        score += 40; reasons.append('回调缩量(吸筹)')
    if prev_up: score += 20; reasons.append('前期放量上涨')
    if c > np.mean(closes[-50:]): score += 20; reasons.append('中期趋势完好')
    
    vote = score >= 50
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '技术'}


def master_elder(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Elder: 三重滤网 — 长趋势↑ + 中回调 + 短突破"""
    closes = df['close'].values; vols = _vol_arr(df)
    if len(closes) < 120: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '技术'}
    
    c = closes[-1]; ma200 = np.mean(closes[-200:])
    tide_up = c > ma200
    
    if len(closes) >= 15:
        deltas = np.diff(closes[-15:])
        g = deltas.copy(); g[g<0]=0; l = -deltas.copy(); l[l<0]=0
        rsi = 100-100/(1+np.mean(g)/np.mean(l)) if np.mean(l)>0 else 100
    else: rsi = 50
    
    yang = c > df['open'].values[-1] if 'open' in df.columns else True
    vol_ok = vols[-1] > np.mean(vols[-10:])
    
    score = 0; reasons = []
    if tide_up: score += 30; reasons.append('长趋势向上')
    if 35 < rsi < 65: score += 25; reasons.append(f'中周期回调(RSI{rsi:.0f})')
    if yang and vol_ok: score += 25; reasons.append('短期突破')
    
    vote = score >= 60 and tide_up
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '技术'}


def master_raschke(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Raschke: 超卖反弹 或 窄幅突破"""
    closes = df['close'].values; highs = df['high'].values; lows = df['low'].values
    vols = _vol_arr(df)
    if len(closes) < 40: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '技术'}
    
    c = closes[-1]; ret_5d = c / closes[-5] - 1
    yang = c > df['open'].values[-1] if 'open' in df.columns else True
    range_10d = (np.max(highs[-10:-1]) - np.min(lows[-10:-1])) / np.mean(closes[-10:-1])
    
    score = 0; reasons = []
    if ret_5d < -0.05 and yang and vols[-1] > np.mean(vols[-20:]):
        score += 50; reasons.append('超卖放量反弹')
    elif range_10d < 0.08 and c > np.max(highs[-10:-1]):
        score += 50; reasons.append('窄幅突破')
    if c > np.mean(closes[-20:]): score += 20
    
    vote = score >= 55
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '技术'}


# ══════════════════════════════════════════════════════
#  宏观/量化派 (6) — 简化
# ══════════════════════════════════════════════════════

def master_soros(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Soros: 趋势加速 + 量能加速"""
    closes = df['close'].values; vols = _vol_arr(df)
    if len(closes) < 40: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '宏观'}
    c = closes[-1]
    ret_10d = c/closes[-10]-1; ret_20d = c/closes[-20]-1
    vol_accel = np.mean(vols[-5:]) > np.mean(vols[-15:-5])
    score = 0; reasons = []
    if ret_10d > 0.05 and ret_20d > 0.08: score += 35; reasons.append('趋势加速')
    if vol_accel: score += 30; reasons.append('量能加速')
    if c > np.mean(closes[-50:]): score += 20
    vote = score >= 55
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '宏观'}


def master_druckenmiller(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Druckenmiller: 放量突破 + 确定性"""
    closes = df['close'].values; vols = _vol_arr(df)
    if len(closes) < 40: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '宏观'}
    c = closes[-1]; ma20 = np.mean(closes[-20:])
    vol_ratio = vols[-1]/np.mean(vols[-20:]); ret_5d = c/closes[-5]-1
    score = 0; reasons = []
    if c > ma20 and vol_ratio > 1.8: score += 40; reasons.append('放量突破')
    if ret_5d > 0.03: score += 25
    if c > np.mean(closes[-60:]): score += 20
    vote = score >= 60
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '宏观'}


def master_dalio(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Dalio: 稳健上升 + 低波动"""
    closes = df['close'].values
    if len(closes) < 120: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '宏观'}
    c = closes[-1]; ma200 = np.mean(closes[-200:])
    ret_60d = c/closes[-60]-1; vol_60d = np.std(closes[-60:])/np.mean(closes[-60:])
    score = 0; reasons = []
    if c > ma200 and 0 < ret_60d < 0.40: score += 40; reasons.append('稳健上升')
    if vol_60d < 0.15: score += 30; reasons.append('低波动')
    if ret_60d > 0.10: score += 15
    vote = score >= 55
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '宏观'}


def master_simons(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Simons: 统计异常 — 超卖+异常量 或 突破+异常量"""
    closes = df['close'].values; vols = _vol_arr(df)
    if len(closes) < 30: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '量化'}
    c = closes[-1]; ret_3d = c/closes[-3]-1; ret_10d = c/closes[-10]-1
    vol_spike = vols[-1] > np.mean(vols[-20:])*2
    score = 0; reasons = []
    if ret_3d < -0.04 and vol_spike: score += 40; reasons.append('超卖+异常量')
    elif ret_3d > 0.04 and vol_spike: score += 35; reasons.append('突破+异常量')
    if -0.02 < ret_10d < 0.10: score += 20
    vote = score >= 50
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '量化'}


def master_griffin(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Griffin: 波动率扩张 + 方向向上"""
    closes = df['close'].values; highs = df['high'].values; lows = df['low'].values
    if len(closes) < 40: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '量化'}
    c = closes[-1]
    range_5d = np.mean([(highs[i]-lows[i])/closes[i] for i in range(-5,0)])
    range_20d = np.mean([(highs[i]-lows[i])/closes[i] for i in range(-20,-5)])
    score = 0; reasons = []
    if range_5d > range_20d * 1.5: score += 40; reasons.append('波动率扩张')
    if c > np.mean(closes[-20:]): score += 30; reasons.append('方向向上')
    vote = score >= 60
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '量化'}


def master_asness(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Asness: 质量价值 — 中低位+动量向上 + ROIC高 + 利润增长 + 低杠杆"""
    closes = df['close'].values
    if len(closes) < 120: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '量化'}
    c = closes[-1]; low_120 = np.min(closes[-120:]); high_120 = np.max(closes[-120:])
    value = (c - low_120)/(high_120 - low_120) if high_120 > low_120 else 0.5
    momentum = c/closes[-60]-1
    f = _get_fund(df, fund)
    score = 0; reasons = []
    if 0.2 < value < 0.6: score += 20; reasons.append('中低位')
    if momentum > 0.08: score += 25; reasons.append('动量向上')
    if value < 0.4 and momentum > 0: score += 12; reasons.append('价值+动量共振')
    if 0 < f['pe_ttm'] < 40: score += 10; reasons.append(f'PE{f["pe_ttm"]:.0f}')
    # 质量因子 (Asness QMJ)
    if f['roic'] > 10: score += 15; reasons.append(f'ROIC{f["roic"]:.0f}%(高质量)')
    if f['netprofit_yoy'] > 5: score += 12; reasons.append('利润增长')
    if 0 < f['debt_to_eqt'] < 2.0: score += 10; reasons.append('低杠杆')
    vote = score >= 50
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '量化'}


def master_greenblatt(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Greenblatt: 神奇公式 — 高ROIC + 高盈利收益率 (PE低+质量好=性价比最高)"""
    closes = df['close'].values
    if len(closes) < 60: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '量化'}
    f = _get_fund(df, fund)
    c = closes[-1]
    
    score = 0; reasons = []
    
    # 盈利收益率 = 1/PE (简化版; Greenblatt用EBIT/EV, 这里用PE代理)
    earnings_yield = 100 / f['pe_ttm'] if f['pe_ttm'] > 0 else 0
    
    # ROIC (资本回报率 — Greenblatt核心)
    if f['roic'] > 20: score += 35; reasons.append(f'ROIC{f["roic"]:.0f}%(极高)')
    elif f['roic'] > 12: score += 25; reasons.append(f'ROIC{f["roic"]:.0f}%(高)')
    elif f['roic'] > 8: score += 15; reasons.append(f'ROIC{f["roic"]:.0f}%')
    
    # 盈利收益率: 高 = 便宜
    if earnings_yield > 8: score += 30; reasons.append(f'EY{earnings_yield:.0f}%(极便宜)')
    elif earnings_yield > 5: score += 20; reasons.append(f'EY{earnings_yield:.0f}%(便宜)')
    elif earnings_yield > 3: score += 10; reasons.append(f'EY{earnings_yield:.0f}%')
    
    # 成长验证 (好公司不会变差)
    if f['netprofit_yoy'] > 10: score += 15; reasons.append(f'利润+{f["netprofit_yoy"]:.0f}%YoY')
    if f['or_yoy'] > 5: score += 10; reasons.append('营收增长')
    
    # 现金流质量
    if f['q_ocf_to_sales'] > 0.05: score += 10; reasons.append('现金流正')
    
    # 价格趋势 (不要接飞刀)
    if c > np.mean(closes[-20:]): score += 10; reasons.append('短期止跌')
    
    has_fundamentals = f['roic'] > -900 or earnings_yield > 0
    
    vote = score >= 55 and has_fundamentals
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '量化'}


# ══════════════════════════════════════════════════════
#  风险/反对票 (3) — 检测危险信号, 投反对票
# ══════════════════════════════════════════════════════

def master_chanosh(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Chanosh: 做空大师 — 财务危险信号: 高负债+负现金流+利润下滑+流动性差
    ≥2项触发=反对票. 数据驱动阈值: A股负债率P90≈80%, 流动比率P10≈0.8"""
    f = _get_fund(df, fund)
    if f['pe_ttm'] >= 999 and f['roe'] <= -900:
        return {'vote': False, 'anti_vote': False, 'score': 0, 'reason': '无数据', 'group': '风险'}

    dangers = []
    score = 0
    if f['debt_to_assets'] > 80: dangers.append(f'负债率{f["debt_to_assets"]:.0f}%'); score += 25
    if f['q_ocf_to_sales'] < 0:  dangers.append('现金流负');           score += 25
    if f['netprofit_yoy'] < -20:  dangers.append(f'利润-{abs(f["netprofit_yoy"]):.0f}%YoY'); score += 25
    if f['current_ratio'] < 0.8:  dangers.append(f'流动比率{f["current_ratio"]:.2f}'); score += 25

    vote = score >= 50  # ≥2项触发
    return {'vote': vote, 'anti_vote': vote, 'score': score,
            'reason': '⚠'+'|⚠'.join(dangers) if dangers else '无危险信号',
            'group': '风险'}


# 科技行业 — PE不适用 (数据证明科技股PE>80与PE<30前视收益无差异)
# 但软件服务除外 (PE>80+动量=-3.1%)
TECH_PE_EXEMPT = {'通信设备','互联网','半导体','元器件','IT设备','电器仪表','电脑设备'}

def master_klarman(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Klarman: 安全边际 — 贵且不赚钱: PE>100 & ROIC<0 & 现金流严重为负
    A股ROIC中位数仅1.3%, 用ROIC<0(真亏钱)而非<5%
    科技行业(通信设备/互联网/半导体等): 不查PE, 只查ROIC+现金流
    数据: 科技股PE>80前视+2.89% vs PE<30前视+2.32%, PE无预测力"""
    f = _get_fund(df, fund)
    industry = fund.get('industry', '') if fund else ''
    is_tech_exempt = industry in TECH_PE_EXEMPT
    
    if f['pe_ttm'] >= 999 or f['roic'] <= -900:
        return {'vote': False, 'anti_vote': False, 'score': 0, 'reason': '无数据', 'group': '风险'}

    dangers = []; score = 0
    
    if is_tech_exempt:
        # 科技股: 不查PE, 只关心是不是真亏钱+现金流
        if f['roic'] < 0:     dangers.append(f'ROIC负({f["roic"]:.1f}%)'); score += 45
        if f['q_ocf_to_sales'] < -0.2: dangers.append('严重现金流问题'); score += 35
        # 科技股也需要利润验证(没有利润但有故事 → 不反对)
        # ROIC<0 and 现金流<0 → 连故事都撑不住
    else:
        # 传统行业和软件服务: 查PE
        if f['pe_ttm'] > 100: dangers.append(f'PE虚高({f["pe_ttm"]:.0f})'); score += 35
        if f['roic'] < 0:     dangers.append(f'ROIC负({f["roic"]:.1f}%)'); score += 35
        if f['q_ocf_to_sales'] < -0.2: dangers.append('严重现金流问题'); score += 30

    vote = score >= 65  # 需要贵+差同时触发
    return {'vote': vote, 'anti_vote': vote, 'score': score,
            'reason': '⚠'+'|⚠'.join(dangers) if dangers else '估值可接受',
            'group': '风险'}


def master_marks(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Marks: 周期顶部 — PE高+利润下滑+低ROE = 周期顶部信号
    纯量价顶部检测(RSI>75+量缩)在静态筛选层不可用, 用财务代理"""
    closes = df['close'].values
    if len(closes) < 60:
        return {'vote': False, 'anti_vote': False, 'score': 0, 'reason': '数据不足', 'group': '风险'}

    f = _get_fund(df, fund)
    c = closes[-1]
    high_52w = np.max(closes[-200:]) if len(closes) >= 200 else np.max(closes[-len(closes):])
    near_high = c > high_52w * 0.95  # 接近52周高位

    dangers = []; score = 0
    if f['pe_ttm'] > 80 and f['pe_ttm'] < 999: dangers.append(f'PE{f["pe_ttm"]:.0f}(高)'); score += 25
    if f['netprofit_yoy'] < -10: dangers.append(f'利润下滑{abs(f["netprofit_yoy"]):.0f}%'); score += 25
    if f['roe'] < 5 and f['roe'] > -900: dangers.append(f'ROE低({f["roe"]:.0f}%)'); score += 25
    if near_high: dangers.append('接近52周高位'); score += 25

    vote = score >= 50  # ≥2项
    return {'vote': vote, 'anti_vote': vote, 'score': score,
            'reason': '⚠'+'|⚠'.join(dangers) if dangers else '非顶部',
            'group': '风险'}


# ══════════════════════════════════════════════════════
#  行业/板块 (2) — 申万行业动量+轮动
# ══════════════════════════════════════════════════════

# 模块级行业上下文 (在 run_all_masters 前设置)
_sector_ctx: dict[str, dict] = {}

def set_sector_context(ctx: dict[str, dict]):
    """设置行业上下文: {industry: {ret_20d_median, ret_5d_median, ...}}"""
    global _sector_ctx
    _sector_ctx = ctx


def master_sector_momentum(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Sector Momentum: 同申万行业近20日涨幅中位数>3% → 顺板块趋势"""
    industry = fund.get('industry', '') if fund else ''
    ctx = _sector_ctx.get(industry, {})

    if not ctx or not industry:
        return {'vote': False, 'anti_vote': False, 'score': 0, 'reason': '无行业数据', 'group': '行业'}

    ret_20d = ctx.get('ret_20d_median', 0)
    ret_5d = ctx.get('ret_5d_median', 0)
    n_stocks = ctx.get('n_stocks', 0)
    score = 0; reasons = []

    if ret_20d > 5:        score += 35; reasons.append(f'{industry}板块+{ret_20d*100:.0f}%(强)')
    elif ret_20d > 3:      score += 25; reasons.append(f'{industry}板块+{ret_20d*100:.0f}%')
    elif ret_20d > 0:      score += 10; reasons.append(f'{industry}板块微涨')
    if ret_5d > 0.02:      score += 20; reasons.append('近5日加速')
    if n_stocks >= 10:     score += 10; reasons.append(f'{n_stocks}只同行')

    vote = score >= 35
    return {'vote': vote, 'anti_vote': False, 'score': score,
            'reason': '|'.join(reasons) if reasons else f'{industry}板块平淡',
            'group': '行业'}


def master_sector_rotation(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Sector Rotation: 板块轮动进场 — 5日超额收益加速(5日收益>20日收益/4)"""
    industry = fund.get('industry', '') if fund else ''
    ctx = _sector_ctx.get(industry, {})

    if not ctx or not industry:
        return {'vote': False, 'anti_vote': False, 'score': 0, 'reason': '无行业数据', 'group': '行业'}

    ret_5d = ctx.get('ret_5d_median', 0)
    ret_20d = ctx.get('ret_20d_median', 0)
    n_stocks = ctx.get('n_stocks', 0)

    # 轮动检测: 近期加速
    accel = ret_5d > ret_20d / 4 if ret_20d > 0 else ret_5d > 0.01
    # 超额: 相对全市场
    market_ret = _sector_ctx.get('__market__', {}).get('ret_20d_median', 0)
    excess = ret_20d - market_ret

    score = 0; reasons = []
    if accel:              score += 35; reasons.append('近5日加速(轮动进场)')
    if excess > 0.03:      score += 30; reasons.append(f'超额+{excess*100:.0f}%')
    elif excess > 0:       score += 15; reasons.append(f'微超额+{excess*100:.0f}%')
    if n_stocks >= 10:     score += 10; reasons.append(f'{n_stocks}只同行')

    vote = score >= 30 and accel
    return {'vote': vote, 'anti_vote': False, 'score': score,
            'reason': '|'.join(reasons) if reasons else f'{industry}无轮动信号',
            'group': '行业'}


def master_emerging_theme(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Emerging Theme: 板块扩散加速器 — ≥3只放量暴涨+市场向好 → 跟风股补涨
    
    数据驱动规则:
      - 同申万行业≥3只20日>30%且平均量比≥1.5
      - CSI300 20日>0 (市场不跌)
      - 当前股票未暴涨(ret_20d≤0.3, 不追龙头)
      - 反对票不受影响(亏损造假照样拦)
    """
    industry = fund.get('industry', '') if fund else ''
    ctx = _sector_ctx.get(industry, {})
    mkt = _sector_ctx.get('__market__', {})
    
    if not ctx or not industry:
        return {'vote': False, 'anti_vote': False, 'score': 0, 'reason': '无行业数据', 'group': '板块动量'}
    
    is_emerging = ctx.get('is_emerging', False)
    # 数据驱动: 板块扩散只在极端市有效(大涨>5%或大跌<-2%)
    is_extreme = mkt.get('is_extreme', False)
    market_ret = mkt.get('ret_20d_median', 0)
    surge_count = ctx.get('surge_count', 0)
    avg_vol = ctx.get('avg_surge_vol', 0)
    n_stocks = ctx.get('n_stocks', 0)
    
    # 检查当前股票是否已经暴涨(不追龙头)
    closes = df['close'].values
    if len(closes) >= 20:
        ret_20d = closes[-1] / closes[-20] - 1
        is_surged = ret_20d > 0.3
    else:
        is_surged = False
    
    score = 0; reasons = []
    
    if is_emerging and is_extreme and not is_surged:
        score += 40; reasons.append(f'{industry}板块{surge_count}只暴涨(量比{avg_vol:.1f})')
        regime_label = '恐慌扩散' if market_ret < 0 else '强势扩散'
        score += 20; reasons.append(f'极端市{regime_label}(CSI300 {market_ret:+.1%})')
        score += 15; reasons.append('跟风补涨(未暴涨)')
        if n_stocks >= 10: score += 10; reasons.append(f'{n_stocks}只同行')
    elif is_emerging and not is_extreme:
        reasons.append(f'市场横盘({market_ret:+.1%}), 板块扩散不可靠')
    elif is_emerging and is_surged:
        reasons.append(f'已是暴涨龙头({ret_20d*100:.0f}%), 不追')
    elif not is_emerging:
        if surge_count > 0:
            reasons.append(f'仅{surge_count}只暴涨(需≥3)')
        else:
            reasons.append('板块无暴涨')
    
    vote = score >= 50
    return {'vote': vote, 'anti_vote': False, 'score': score,
            'reason': '|'.join(reasons) if reasons else '无Emerging Theme',
            'group': '板块动量'}


def master_mean_reversion(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Mean Reversion: A股最强因子 — 60日暴跌>20%+放量+止跌 → 超跌反弹
    
    数据: 60日暴跌<-20% → 前视+1.55%超额, 胜率54%, A股单因子最强
    逻辑: A股散户追涨杀跌→过度抛售→均值回归"""
    closes = df['close'].values; vols = _vol_arr(df)
    if len(closes) < 60: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '价格动量'}
    
    c = closes[-1]; ret_60d = c/closes[-60]-1
    ret_5d = c/closes[-5]-1 if len(closes)>=5 else 0
    vol_ratio = np.mean(vols[-5:])/np.mean(vols[-20:])
    
    score = 0; reasons = []
    if ret_60d < -0.30: score += 35; reasons.append(f'暴跌{ret_60d:.0%}(深度超卖)')
    elif ret_60d < -0.20: score += 25; reasons.append(f'超跌{ret_60d:.0%}')
    if vol_ratio > 1.5: score += 25; reasons.append('放量抄底')
    elif vol_ratio > 1.2: score += 15; reasons.append('量能恢复')
    if ret_5d > -0.02: score += 20; reasons.append('短期止跌')
    if c > np.mean(closes[-20:]): score += 15; reasons.append('站上20日均线')
    
    vote = score >= 45 and ret_60d < -0.15
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '价格动量'}


def master_volume_breakout(df: pd.DataFrame, fund: Optional[dict] = None) -> dict:
    """Volume Breakout: 量能爆发 — 巨量+方向向上 → 资金入场信号
    
    数据: 放量≥1.5x → 前视+0.79%超额, A股第二强单因子"""
    closes = df['close'].values; vols = _vol_arr(df)
    if len(closes) < 40: return {'vote': False, 'score': 0, 'reason': '数据不足', 'group': '价格动量'}
    
    c = closes[-1]
    vol_5d = np.mean(vols[-5:])/np.mean(vols[-20:])
    vol_20d = np.mean(vols[-20:])/np.mean(vols[-60:])
    ret_5d = c/closes[-5]-1 if len(closes)>=5 else 0
    
    score = 0; reasons = []
    if vol_5d > 2.0: score += 40; reasons.append(f'5日巨量({vol_5d:.1f}x)')
    elif vol_5d > 1.5: score += 25; reasons.append(f'5日放量({vol_5d:.1f}x)')
    if vol_20d > 1.3: score += 20; reasons.append('持续放量')
    if ret_5d > 0: score += 20; reasons.append('方向向上')
    if c > np.mean(closes[-20:]): score += 15; reasons.append('中期企稳')
    
    vote = score >= 45
    return {'vote': vote, 'score': score, 'reason': '|'.join(reasons), 'group': '价格动量'}


# ══════════════════════════════════════════════════════
#  注册表
# ══════════════════════════════════════════════════════

MASTERS = {
    'ONeil':      master_oneil,
    'Livermore':  master_livermore,
    'Darvas':     master_darvas,
    'Minervini':  master_minervini,
    'Buffett':    master_buffett,
    'Graham':     master_graham,
    'Templeton':  master_templeton,
    'Lynch':      master_lynch,
    'Fisher':     master_fisher,
    'Wood':       master_wood,
    'Wyckoff':    master_wyckoff,
    'Elder':      master_elder,
    'Raschke':    master_raschke,
    'Soros':      master_soros,
    'Druckenmiller': master_druckenmiller,
    'Dalio':      master_dalio,
    'Simons':     master_simons,
    'Griffin':    master_griffin,
    'Asness':     master_asness,
    'Greenblatt': master_greenblatt,
    'Chanosh':    master_chanosh,
    'Klarman':    master_klarman,
    'Marks':      master_marks,
    'Sector_Mom':  master_sector_momentum,
    'Sector_Rot':  master_sector_rotation,
    'Emerging':    master_emerging_theme,
    'MeanRev':     master_mean_reversion,
    'VolBreak':    master_volume_breakout,
}

MASTER_GROUPS = {
    'ONeil':'动量','Livermore':'动量','Darvas':'动量','Minervini':'动量',
    'Buffett':'价值','Graham':'价值','Templeton':'价值',
    'Lynch':'成长','Fisher':'成长','Wood':'成长',
    'Wyckoff':'技术','Elder':'技术','Raschke':'技术',
    'Soros':'宏观','Druckenmiller':'宏观','Dalio':'宏观',
    'Simons':'量化','Griffin':'量化','Asness':'量化','Greenblatt':'量化',
    'Chanosh':'风险','Klarman':'风险','Marks':'风险',
    'Sector_Mom':'行业','Sector_Rot':'行业',
    'Emerging':'板块动量',
    'MeanRev':'价格动量',
    'VolBreak':'价格动量',
}


# ══════════════════════════════════════════════════════
#  信号簇 — 基于投票相关性聚类的真实独立视角
# ══════════════════════════════════════════════════════

# 从300只股票投票数据中计算出的自然信号簇 (相关性>0.4)
SIGNAL_CLUSTERS = {
    'momentum_trend':   ['ONeil', 'Livermore', 'Lynch', 'Soros', 'Dalio'],      # 趋势动量块
    'breakout':         ['Darvas'],                                               # 箱体突破(独立)
    'sepa_template':    ['Minervini'],                                            # SEPA模板(独立,更严)
    'value_deep':       ['Buffett'],                                              # 深度价值
    'value_cigar':      ['Graham'],                                               # 烟蒂价值
    'contrarian':       ['Templeton'],                                            # 逆向投资
    'growth_quality':   ['Fisher'],                                               # 成长质量
    'growth_disrupt':   ['Wood'],                                                 # 颠覆创新
    'tech_wyckoff':     ['Wyckoff'],                                              # 吸筹识别
    'tech_elder':       ['Elder'],                                                # 三重滤网
    'tech_raschke':     ['Raschke'],                                              # 均值回归
    'liquidity_stat':   ['Druckenmiller', 'Simons'],                             # 流动性和统计
    'quant_vol':        ['Griffin'],                                              # 波动率
    'quant_value_mom':  ['Asness'],                                               # 质量价值+动量
    'quant_magic':      ['Greenblatt'],                                           # 神奇公式
    'risk_financial':   ['Chanosh', 'Klarman'],                                  # 财务风险(反对票)
    'risk_cyclical':    ['Marks'],                                                # 周期顶部(反对票)
    'sector_momentum':  ['Sector_Mom'],                                           # 板块动量
    'sector_rotation':  ['Sector_Rot'],                                           # 板块轮动
    'emerging_theme':   ['Emerging'],                                             # 板块扩散加速器
    'mean_reversion':   ['MeanRev'],                                              # 超跌反弹 (A股最强)
    'volume_breakout':  ['VolBreak'],                                             # 量能爆发
}

# 每个簇的角色: bullish(顺趋势), contrarian(逆趋势), neutral, risk(反对)
CLUSTER_ROLE = {
    'momentum_trend':   'bullish',
    'breakout':         'bullish',
    'sepa_template':    'bullish',
    'value_deep':       'contrarian',
    'value_cigar':      'contrarian',
    'contrarian':       'contrarian',
    'growth_quality':   'bullish',
    'growth_disrupt':   'bullish',
    'tech_wyckoff':     'neutral',
    'tech_elder':       'neutral',
    'tech_raschke':     'neutral',
    'liquidity_stat':   'bullish',
    'quant_vol':        'neutral',
    'quant_value_mom':  'neutral',
    'quant_magic':      'neutral',
    'risk_financial':   'risk',
    'risk_cyclical':    'risk',
    'sector_momentum':  'bullish',
    'sector_rotation':  'bullish',
    'emerging_theme':   'bullish',
    'mean_reversion':   'contrarian',
    'volume_breakout':  'bullish',
}


def run_all_masters(df: pd.DataFrame, fundamentals: Optional[dict] = None) -> list[dict]:
    """运行全部19位大师"""
    results = []
    for name, fn in MASTERS.items():
        try:
            r = fn(df, fundamentals)
            r['name'] = name
            r['group'] = MASTER_GROUPS[name]
            results.append(r)
        except Exception as e:
            results.append({'name': name, 'group': MASTER_GROUPS[name],
                           'vote': False, 'score': 0, 'reason': f'error:{e}'})
    return results


def get_consensus(results: list[dict], min_clusters: int = 3,
                  require_contrarian: bool = False,
                  master_weights: dict[str, float] = None) -> dict:
    """大师共识计算 v3 — 信号簇 + 反对票扣分 + 加权投票

    正向簇: bullish / neutral / contrarian 角色的簇(≥1票)
    风险簇: risk 角色的簇(反对票) — 从净正向簇扣除

    高确信条件:
      - 净正向簇 ≥ min_clusters
      - 至少1个bullish簇
    """
    votes = [r for r in results if r.get('vote')]
    voter_names = set(r['name'] for r in votes)
    weights = master_weights or {}

    # 加权计算每个信号簇的有效性
    positive_clusters = set()   # bullish/neutral/contrarian
    risk_clusters = set()       # 反对票
    cluster_voters = {}
    cluster_weighted_score = {}
    
    for cluster_name, members in SIGNAL_CLUSTERS.items():
        cluster_vote = [m for m in members if m in voter_names]
        if cluster_vote:
            w_score = sum(weights.get(m, 1.0) for m in cluster_vote)
            cluster_weighted_score[cluster_name] = w_score
            if w_score >= 1.0:
                role = CLUSTER_ROLE.get(cluster_name, '?')
                if role == 'risk':
                    risk_clusters.add(cluster_name)
                else:
                    positive_clusters.add(cluster_name)
                cluster_voters[cluster_name] = cluster_vote

    # 净正向簇 = 正向簇 - 风险簇
    net_clusters = len(positive_clusters) - len(risk_clusters)
    all_voted_clusters = positive_clusters | risk_clusters

    has_bullish = any(CLUSTER_ROLE.get(c) == 'bullish' for c in positive_clusters)
    has_contrarian = any(CLUSTER_ROLE.get(c) == 'contrarian' for c in positive_clusters)
    has_emerging = 'emerging_theme' in positive_clusters
    # 反对票: 需要≥2位风险大师同时反对才生效 (避免单一条件误杀)
    has_risk = len(risk_clusters) >= 2
    raw_groups = set(r['group'] for r in votes)

    # Emerging Theme: 门槛从3降到2 (板块扩散信号本身已验证有效)
    effective_min = 2 if has_emerging else min_clusters
    
    consensus = net_clusters >= effective_min
    cross_view = has_bullish and has_contrarian if require_contrarian else (net_clusters >= effective_min)

    return {
        'total_votes': len(votes),
        'raw_groups': len(raw_groups),
        'positive_clusters': len(positive_clusters),
        'risk_clusters': len(risk_clusters),
        'net_clusters': net_clusters,
        'clusters_voted': len(all_voted_clusters),
        'cluster_names': sorted(all_voted_clusters),
        'cluster_roles': {c: CLUSTER_ROLE.get(c, '?') for c in all_voted_clusters},
        'cluster_scores': cluster_weighted_score,
        'has_bullish': has_bullish,
        'has_contrarian': has_contrarian,
        'has_emerging': has_emerging,
        'has_risk': has_risk,
        'consensus': consensus,
        'cross_view': cross_view,
        'high_confidence': consensus and cross_view and has_bullish,
        'voters': [(r['name'], r['group'], r['score']) for r in votes],
        'cluster_voters': cluster_voters,
        'all_scores': [(r['name'], r['group'], r['score'], r.get('vote', False)) for r in results],
    }


def get_rating(consensus: dict) -> str:
    """数据驱动评级 v5 — A股适配版
    
    全池1903只×26期回测发现: 票数越多,收益越差
      ≥13票: -2.33% 胜率32% → 拥挤共识, 跑输
       4-7票: 打平市场 → 最优分歧区
    
    逻辑: A股共识=拥挤,适度分歧=信息差
    """
    votes = consensus.get('total_votes', 0)
    has_risk = consensus.get('has_risk', False)
    
    if has_risk or votes < 4:
        return ''
    if 4 <= votes <= 7:
        return '★★★★'
    elif 8 <= votes <= 11:
        return '★★★'
    else:  # votes >= 12
        return '★★'
