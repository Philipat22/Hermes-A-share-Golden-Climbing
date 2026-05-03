#!/usr/bin/env python3
"""
主升浪策略回测脚本

用法:
    python scripts/backtest_main_surge.py

输出:
    quant_archive/YYYY-MM/main_surge_equity.csv  — 每日权益曲线
    quant_archive/YYYY-MM/main_surge_trades.csv  — 逐笔交易记录
    quant_archive/YYYY-MM/main_surge_summary.json — 汇总统计
"""
import os, sys, json, warnings, pickle, gc, time
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 项目路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.surge.main_surge_strategy import (
    MainSurgeStrategy, StrategyParams, compute_indicators, Signal
)
from src.surge.regime_classifier import RegimeClassifier
from src.utils.sector_map import SECTOR_INDUSTRY_MAP

# ══════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════

PRICE_PICKLE = os.path.join(ROOT, 'data', 'cache', 'backtest_prices_extended.pkl')
OUTPUT_DIR = os.path.join(ROOT, 'quant_archive', datetime.now().strftime('%Y-%m'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 回测周期 (与 V2r2 同期，方便对比)
START_DATE = '2022-01-01'
END_DATE = '2025-06-30'

# 资金与成本
INITIAL_CASH = 1_000_000.0
COMMISSION_RATE = 0.0003      # 万3
STAMP_DUTY = 0.0005           # 万5 卖
SLIPPAGE = 0.001              # 0.1% 滑点

# A-Share
LOT_SIZE = 100                # 1手=100股
LIMIT_UP_PCT = 0.10
LIMIT_DOWN_PCT = -0.10

# ══════════════════════════════════════════════════════
# 回测引擎
# ══════════════════════════════════════════════════════

class BacktestResult:
    """回测结果收集器"""
    def __init__(self):
        self.equity_curve: list[dict] = []
        self.trades: list[dict] = []
        self.daily_positions: list[dict] = []

    def add_equity(self, date, cash, holdings, total, daily_ret=0.0, num_pos=0, regime=''):
        self.equity_curve.append({
            'date': date, 'cash': cash, 'holdings': holdings,
            'value': total, 'ret': daily_ret, 'num_positions': num_pos,
            'regime': regime,
        })

    def add_trade(self, trade: dict):
        self.trades.append(trade)

    def to_dataframes(self):
        eq = pd.DataFrame(self.equity_curve)
        tr = pd.DataFrame(self.trades)
        return eq, tr


def run_backtest(
    prices_dict: dict[str, pd.DataFrame],
    params: StrategyParams,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> BacktestResult:
    """执行回测"""
    result = BacktestResult()

    # ── 1. 预计算指标 ──
    t0 = time.time()
    print(f"Computing indicators for {len(prices_dict)} stocks...")
    df_indicators = compute_indicators(prices_dict)
    print(f"  Done in {(time.time()-t0)/60:.1f} min, {len(df_indicators):,} rows")

    # ── 2. 初始化策略和状态分类 ──
    strategy = MainSurgeStrategy(df_indicators, params, SECTOR_INDUSTRY_MAP)
    rc = RegimeClassifier()

    # ── 3. 准备价格快速查找 ──
    price_lookup: dict[tuple, dict] = {}
    for sym, df in prices_dict.items():
        for _, row in df.iterrows():
            d = str(row['date'])[:10]
            price_lookup[(d, sym)] = {
                'open': row['open'], 'high': row['high'],
                'low': row['low'], 'close': row['close'],
                'volume': row.get('volume', 0),
            }

    # ── 4. 日循环 ──
    trading_dates = sorted(set(
        d for d in df_indicators['date']
        if start_date <= str(d)[:10] <= end_date
    ))
    trading_dates_str = [str(d)[:10] for d in trading_dates]

    print(f"Backtesting {len(trading_dates_str)} trading days ({trading_dates_str[0]} ~ {trading_dates_str[-1]})")

    cash = INITIAL_CASH
    prev_total = INITIAL_CASH
    all_trades: list[dict] = []

    for i, date in enumerate(trading_dates_str):
        if i % 60 == 0:
            print(f"  {date} ({i+1}/{len(trading_dates_str)})...")

        # ── 市场状态 ──
        try:
            regime_info = rc.classify(date=date, return_details=False)
            regime = regime_info.get('regime', 'UNKNOWN')
        except:
            regime = 'UNKNOWN'

        # ── 计算当前持仓市值 ──
        holdings_value = 0.0
        for sym, pos in strategy.positions.items():
            info = price_lookup.get((date, sym))
            if info:
                holdings_value += info['close'] * pos.shares
            else:
                # 股票停牌或无数据，用入场价
                holdings_value += pos.entry_price * pos.shares

        total_value = cash + holdings_value

        # ── 检查出场 ──
        entries, exits = strategy.check_signals(date, regime)

        for sig in exits:
            info = price_lookup.get((date, sig.symbol))
            if info is None:
                continue
            exit_price = info['close'] * (1 - SLIPPAGE)  # 卖出滑点

            pos = strategy.positions.get(sig.symbol)
            if pos is None:
                continue

            shares = pos.shares
            gross_proceeds = exit_price * shares
            # 成本: commission + stamp duty
            cost = gross_proceeds * (COMMISSION_RATE + STAMP_DUTY)
            net_proceeds = gross_proceeds - cost

            cash += net_proceeds
            strategy.remove_position(sig.symbol)

            trade_return = (net_proceeds / (pos.entry_price * pos.shares)) - 1
            all_trades.append({
                'symbol': sig.symbol,
                'entry_date': pos.entry_date,
                'exit_date': date,
                'entry_price': pos.entry_price,
                'exit_price': exit_price,
                'shares': shares,
                'gross_return': (gross_proceeds / (pos.entry_price * pos.shares)) - 1,
                'net_return': trade_return,
                'exit_reason': sig.reason,
                'days_held': (pd.to_datetime(date) - pd.to_datetime(pos.entry_date)).days,
                'regime': regime,
            })

        # ── 检查入场 ──
        for sig in entries:
            info = price_lookup.get((date, sig.symbol))
            if info is None:
                continue

            # 检查涨跌停
            if not _can_trade(info):
                continue

            entry_price = info['close'] * (1 + SLIPPAGE)  # 买入滑点

            # 仓位计算: 按波动率反比
            # 从指标查找 ATR%
            row = strategy._get_row(date, sig.symbol)
            atr_pct = row.get('atr_pct', 3.0) if row is not None else 3.0
            if pd.isna(atr_pct) or atr_pct <= 0:
                atr_pct = 3.0

            # 风险预算: 单笔最大亏损 = 净值的 2%
            risk_budget = total_value * 0.02
            stop_loss_pct_use = max(params.stop_loss_pct, -params.stop_loss_atr_mult * atr_pct / 100)
            per_share_risk = entry_price * abs(stop_loss_pct_use)

            if per_share_risk <= 0:
                continue

            target_shares = int(risk_budget / per_share_risk / LOT_SIZE) * LOT_SIZE
            max_position_shares = int(total_value * params.max_position_pct / entry_price / LOT_SIZE) * LOT_SIZE
            target_shares = min(target_shares, max_position_shares, LOT_SIZE * 100)  # 最少1手，最多100手
            target_shares = max(LOT_SIZE, target_shares)

            cost = entry_price * target_shares
            entry_cost = cost * COMMISSION_RATE
            total_cost = cost + entry_cost

            if total_cost > cash * 0.25:  # 不超现金25%
                target_shares = int(cash * 0.25 / (entry_price * (1 + COMMISSION_RATE)) / LOT_SIZE) * LOT_SIZE
                target_shares = max(LOT_SIZE, target_shares)
                total_cost = entry_price * target_shares * (1 + COMMISSION_RATE)

            if total_cost > cash:
                target_shares = int(cash * 0.9 / (entry_price * (1 + COMMISSION_RATE)) / LOT_SIZE) * LOT_SIZE
                target_shares = max(LOT_SIZE, target_shares)
                if target_shares * entry_price * (1 + COMMISSION_RATE) > cash:
                    continue
                total_cost = entry_price * target_shares * (1 + COMMISSION_RATE)

            cash -= total_cost
            strategy.add_position(sig.symbol, date, entry_price, target_shares)

        # ── 重新计算总市值 ──
        holdings_value = 0.0
        for sym, pos in strategy.positions.items():
            info = price_lookup.get((date, sym))
            if info:
                holdings_value += info['close'] * pos.shares
            else:
                holdings_value += pos.entry_price * pos.shares

        total_value = cash + holdings_value
        daily_ret = (total_value / prev_total) - 1 if prev_total > 0 else 0.0
        prev_total = total_value

        result.add_equity(
            date, cash, holdings_value, total_value,
            daily_ret=daily_ret,
            num_pos=len(strategy.positions),
            regime=regime,
        )

    # ── 最终清仓 ──
    last_date = trading_dates_str[-1] if trading_dates_str else end_date
    for sym in list(strategy.positions.keys()):
        pos = strategy.positions[sym]
        info = price_lookup.get((last_date, sym))
        exit_price = info['close'] * (1 - SLIPPAGE) if info else pos.entry_price
        gross = exit_price * pos.shares
        cost = gross * (COMMISSION_RATE + STAMP_DUTY)
        cash += gross - cost
        all_trades.append({
            'symbol': sym,
            'entry_date': pos.entry_date,
            'exit_date': last_date,
            'entry_price': pos.entry_price,
            'exit_price': exit_price,
            'shares': pos.shares,
            'gross_return': (gross / (pos.entry_price * pos.shares)) - 1,
            'net_return': ((gross - cost) / (pos.entry_price * pos.shares)) - 1,
            'exit_reason': 'force_close',
            'days_held': (pd.to_datetime(last_date) - pd.to_datetime(pos.entry_date)).days,
            'regime': 'force',
        })
        strategy.remove_position(sym)

    result.trades = all_trades
    eq_df, trade_df = result.to_dataframes()
    result.equity_df = eq_df
    result.trade_df = trade_df

    return result


def _can_trade(info: dict) -> bool:
    """检查是否可交易（非涨跌停）"""
    close = info.get('close', 0)
    high = info.get('high', close)
    low = info.get('low', close)
    if close <= 0 or high <= 0:
        return False
    # 涨停: high == close (简化判断)
    # 跌停: low == close
    if close >= high * 0.999:  # 涨停
        return False
    if close <= low * 1.001:   # 跌停
        return False
    return True


# ══════════════════════════════════════════════════════
# 统计
# ══════════════════════════════════════════════════════

def compute_stats(result: BacktestResult) -> dict:
    """计算汇总统计"""
    eq = result.equity_df
    trades = pd.DataFrame(result.trades) if len(result.trades) > 0 else pd.DataFrame()

    stats = {}

    if len(eq) == 0:
        return {'error': 'no_equity_data'}

    # 收益
    initial = eq['value'].iloc[0]
    final = eq['value'].iloc[-1]
    cum_ret = final / initial - 1

    # 年化
    days = len(eq)
    years = days / 240
    ann_ret = (1 + cum_ret) ** (1 / years) - 1 if years > 0 else 0

    # 夏普
    daily_rets = eq['ret'].dropna()
    ann_vol = daily_rets.std() * np.sqrt(240) if len(daily_rets) > 0 else 0
    sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0

    # 最大回撤
    cummax = eq['value'].cummax()
    dd = eq['value'] / cummax - 1
    max_dd = dd.min()
    max_dd_idx = dd.idxmin()
    max_dd_date = eq.loc[max_dd_idx, 'date'] if max_dd_idx < len(eq) else ''

    stats.update({
        'initial_cash': INITIAL_CASH,
        'final_value': round(final, 2),
        'cum_return': round(cum_ret, 6),
        'annual_return': round(ann_ret, 6),
        'annual_volatility': round(ann_vol, 6),
        'sharpe_ratio': round(sharpe, 4),
        'max_drawdown': round(max_dd, 6),
        'max_dd_date': str(max_dd_date),
        'trading_days': days,
    })

    # 交易统计
    if len(trades) > 0:
        win_mask = trades['net_return'] > 0
        stats.update({
            'total_trades': len(trades),
            'win_trades': int(win_mask.sum()),
            'loss_trades': int((~win_mask).sum()),
            'win_rate': round(float(win_mask.mean()), 4),
            'avg_net_return': round(float(trades['net_return'].mean()), 6),
            'avg_win': round(float(trades.loc[win_mask, 'net_return'].mean()), 6) if win_mask.any() else 0,
            'avg_loss': round(float(trades.loc[~win_mask, 'net_return'].mean()), 6) if (~win_mask).any() else 0,
            'avg_days_held': round(float(trades['days_held'].mean()), 1),
        })

        # Profit factor
        gross_win = trades.loc[win_mask, 'net_return'].sum() if win_mask.any() else 0
        gross_loss = abs(trades.loc[~win_mask, 'net_return'].sum()) if (~win_mask).any() else 0
        stats['profit_factor'] = round(gross_win / gross_loss, 4) if gross_loss > 0 else float('inf')

        # 按 regime 统计
        regime_stats = trades.groupby('regime').agg(
            trades=('net_return', 'count'),
            win_rate=('net_return', lambda x: (x > 0).mean()),
            avg_return=('net_return', 'mean'),
        ).round(4)
        stats['regime_breakdown'] = regime_stats.to_dict()

    return stats


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("主升浪策略回测")
    print("=" * 72)

    # ── 加载数据 ──
    print(f"\n[1] Loading price data from {PRICE_PICKLE}...")
    t0 = time.time()
    with open(PRICE_PICKLE, 'rb') as f:
        prices_dict = pickle.load(f)
    print(f"  {len(prices_dict)} stocks, {(time.time()-t0):.1f}s")

    # 过滤日期
    for sym in list(prices_dict.keys()):
        df = prices_dict[sym]
        df['date'] = pd.to_datetime(df['date'])
        prices_dict[sym] = df[(df['date'] >= START_DATE) & (df['date'] <= END_DATE)]

    # 去掉数据太短的股票
    min_dates = 0
    for sym in list(prices_dict.keys()):
        if len(prices_dict[sym]) < 200:
            min_dates += 1
            del prices_dict[sym]
    print(f"  Filtered: {min_dates} stocks with <200 data points removed")
    print(f"  Remaining: {len(prices_dict)} stocks")

    # ── 参数 ──
    params = StrategyParams(
        adx_threshold=25.0,
        ma_alignment="tight",
        price_above_ma200=True,
        vol_ratio_threshold=1.5,
        platform_min_days=15,
        platform_max_amplitude=0.15,
        breakout_vol_ratio=2.0,
        n_first_leg_min=0.15,
        n_pullback_max=0.04,
        vcp_window=60,
        sector_min_peers=3,
        macd_confirm=True,
        bb_squeeze_confirm=True,
        stop_loss_pct=-0.05,
        stop_loss_atr_mult=2.0,
        trailing_start_pct=0.08,
        trailing_atr_mult=3.0,
        time_stop_days=20,
        max_position_pct=0.20,
        bear_regime_cash_pct=0.50,
        max_positions=8,
        min_price=3.0,
        max_price=200.0,
    )

    print(f"\n[2] Strategy parameters:")
    for k, v in params.__dict__.items():
        print(f"  {k}: {v}")

    # ── 回测 ──
    print(f"\n[3] Running backtest...")
    result = run_backtest(prices_dict, params)

    # ── 统计 ──
    print(f"\n[4] Computing statistics...")
    stats = compute_stats(result)

    # ── 输出 ──
    print(f"\n{'='*72}")
    print(f"RESULTS")
    print(f"{'='*72}")
    for k, v in stats.items():
        if k == 'regime_breakdown':
            continue
        if isinstance(v, float):
            if abs(v) < 0.1:
                print(f"  {k}: {v:.4%}")
            else:
                print(f"  {k}: {v:,.2f}")
        else:
            print(f"  {k}: {v}")

    if 'regime_breakdown' in stats:
        print(f"\n  Regime breakdown:")
        for regime, data in stats['regime_breakdown'].items():
            print(f"    {regime}: trades={data.get('trades','?')}, "
                  f"win_rate={data.get('win_rate',0)*100:.1f}%, "
                  f"avg_ret={data.get('avg_return',0)*100:.2f}%")

    # ── 保存 ──
    equity_path = os.path.join(OUTPUT_DIR, 'main_surge_equity.csv')
    trades_path = os.path.join(OUTPUT_DIR, 'main_surge_trades.csv')
    summary_path = os.path.join(OUTPUT_DIR, 'main_surge_summary.json')

    result.equity_df.to_csv(equity_path, index=False)
    if len(result.trades) > 0:
        result.trade_df.to_csv(trades_path, index=False)

    with open(summary_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[5] Outputs saved:")
    print(f"  Equity: {equity_path}")
    print(f"  Trades: {trades_path}")
    print(f"  Summary: {summary_path}")

    return result, stats


if __name__ == '__main__':
    main()
