#!/usr/bin/env python3
"""
主升浪策略参数网格搜索

用法:
    python scripts/grid_search_main_surge.py

对 6 个关键参数各取 3 个值 = 729 个组合
输出: quant_archive/YYYY-MM/main_surge_grid_results.csv
"""
import os, sys, json, warnings, pickle, time, itertools
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.surge.main_surge_strategy import (
    MainSurgeStrategy, StrategyParams, compute_indicators
)
from src.surge.regime_classifier import RegimeClassifier
from src.utils.sector_map import SECTOR_INDUSTRY_MAP

PRICE_PICKLE = os.path.join(ROOT, 'data', 'cache', 'backtest_prices_extended.pkl')
OUTPUT_DIR = os.path.join(ROOT, 'quant_archive', datetime.now().strftime('%Y-%m'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-Sample: 2019-2021 (3年训练)
# Out-Sample: 2022-2025.6 (3.5年测试 — 与 V2r2 同期可比)
TRAIN_START = '2019-01-01'
TRAIN_END = '2021-12-31'
TEST_START = '2022-01-01'
TEST_END = '2025-06-30'

INITIAL_CASH = 1_000_000.0
COMMISSION_RATE = 0.0003
STAMP_DUTY = 0.0005
SLIPPAGE = 0.001
LOT_SIZE = 100

# ══════════════════════════════════════════════════════
# 参数网格 (各3个值)
# ══════════════════════════════════════════════════════

PARAM_GRID = {
    'adx_threshold': [20, 25, 30],
    'vol_ratio_threshold': [1.2, 1.5, 2.0],
    'breakout_vol_ratio': [1.5, 2.0, 2.5],
    'stop_loss_pct': [-0.03, -0.05, -0.08],
    'trailing_start_pct': [0.05, 0.08, 0.12],
    'max_position_pct': [0.15, 0.20, 0.25],
}

# ══════════════════════════════════════════════════════
# 轻量级回测 (只输出统计，不存详细记录)
# ══════════════════════════════════════════════════════

def quick_backtest(
    prices_dict: dict,
    df_indicators: pd.DataFrame,
    regime_cache: dict,
    params: StrategyParams,
    start_date: str,
    end_date: str,
) -> dict:
    """快速回测 — 只返回汇总统计"""

    strategy = MainSurgeStrategy(df_indicators, params, SECTOR_INDUSTRY_MAP)

    # 价格查找
    price_lookup: dict[tuple, dict] = {}
    for sym, df in prices_dict.items():
        for _, row in df.iterrows():
            d = str(row['date'])[:10]
            price_lookup[(d, sym)] = {
                'open': row['open'], 'high': row['high'],
                'low': row['low'], 'close': row['close'],
            }

    trading_dates = sorted(set(
        d for d in df_indicators['date']
        if start_date <= str(d)[:10] <= end_date
    ))
    trading_dates_str = [str(d)[:10] for d in trading_dates]

    cash = INITIAL_CASH
    prev_total = INITIAL_CASH
    equity_values = [INITIAL_CASH]
    all_net_rets = []

    for date in trading_dates_str:
        # 市值
        holdings_value = 0.0
        for sym, pos in strategy.positions.items():
            info = price_lookup.get((date, sym))
            if info:
                holdings_value += info['close'] * pos.shares
            else:
                holdings_value += pos.entry_price * pos.shares

        total = cash + holdings_value
        regime = regime_cache.get(date, 'UNKNOWN')

        # 出场
        entries, exits = strategy.check_signals(date, regime)
        for sig in exits:
            info = price_lookup.get((date, sig.symbol))
            if info is None:
                continue
            exit_price = info['close'] * (1 - SLIPPAGE)
            pos = strategy.positions.get(sig.symbol)
            if pos is None:
                continue
            gross = exit_price * pos.shares
            cost = gross * (COMMISSION_RATE + STAMP_DUTY)
            cash += gross - cost
            net_ret = (gross - cost) / (pos.entry_price * pos.shares) - 1
            all_net_rets.append(net_ret)
            strategy.remove_position(sig.symbol)

        # 入场
        for sig in entries:
            info = price_lookup.get((date, sig.symbol))
            if info is None:
                continue
            entry_price = info['close'] * (1 + SLIPPAGE)
            if entry_price <= 0:
                continue

            target_shares = int(total * params.max_position_pct / entry_price / LOT_SIZE) * LOT_SIZE
            target_shares = max(LOT_SIZE, min(target_shares, LOT_SIZE * 100))

            cost_total = entry_price * target_shares * (1 + COMMISSION_RATE)
            if cost_total > cash * 0.25:
                target_shares = int(cash * 0.25 / (entry_price * (1 + COMMISSION_RATE)) / LOT_SIZE) * LOT_SIZE
                target_shares = max(LOT_SIZE, target_shares)
                if target_shares * entry_price * (1 + COMMISSION_RATE) > cash:
                    continue

            cash -= entry_price * target_shares * (1 + COMMISSION_RATE)
            strategy.add_position(sig.symbol, date, entry_price, target_shares)

        # 重算
        holdings_value = 0.0
        for sym, pos in strategy.positions.items():
            info = price_lookup.get((date, sym))
            if info:
                holdings_value += info['close'] * pos.shares
            else:
                holdings_value += pos.entry_price * pos.shares

        total = cash + holdings_value
        equity_values.append(total)
        prev_total = total

    # 最终清仓
    last_date = trading_dates_str[-1] if trading_dates_str else end_date
    for sym in list(strategy.positions.keys()):
        pos = strategy.positions[sym]
        info = price_lookup.get((last_date, sym))
        exit_price = info['close'] * (1 - SLIPPAGE) if info else pos.entry_price
        gross = exit_price * pos.shares
        cost = gross * (COMMISSION_RATE + STAMP_DUTY)
        net_ret = (gross - cost) / (pos.entry_price * pos.shares) - 1
        all_net_rets.append(net_ret)

    # 统计
    equity = np.array(equity_values)
    cum_ret = equity[-1] / equity[0] - 1
    years = len(equity) / 240
    ann_ret = (1 + cum_ret) ** (1 / years) - 1 if years > 0 else 0

    daily_rets = np.diff(equity) / equity[:-1]
    ann_vol = np.std(daily_rets) * np.sqrt(240) if len(daily_rets) > 1 else 0
    sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0

    cummax = np.maximum.accumulate(equity)
    max_dd = np.min(equity[1:] / cummax[1:] - 1) if len(equity) > 1 else 0

    win_rate = np.mean([r > 0 for r in all_net_rets]) if all_net_rets else 0
    avg_ret = np.mean(all_net_rets) if all_net_rets else 0

    return {
        'cum_return': round(cum_ret, 6),
        'annual_return': round(ann_ret, 6),
        'sharpe': round(sharpe, 4),
        'max_dd': round(max_dd, 6),
        'total_trades': len(all_net_rets),
        'win_rate': round(win_rate, 4),
        'avg_trade_return': round(avg_ret, 6),
    }


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("主升浪参数网格搜索")
    print(f"Train: {TRAIN_START} ~ {TRAIN_END}")
    print(f"Test:  {TEST_START} ~ {TEST_END}")
    print("=" * 72)

    # ── 加载数据 ──
    print("\n[1] Loading data...")
    t0 = time.time()
    with open(PRICE_PICKLE, 'rb') as f:
        prices_dict = pickle.load(f)

    for sym in list(prices_dict.keys()):
        df = prices_dict[sym]
        df['date'] = pd.to_datetime(df['date'])

    train_prices = {}
    test_prices = {}
    for sym, df in prices_dict.items():
        t = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)]
        if len(t) >= 200:
            train_prices[sym] = t
        t2 = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)]
        if len(t2) >= 200:
            test_prices[sym] = t2

    print(f"  Train: {len(train_prices)} stocks")
    print(f"  Test:  {len(test_prices)} stocks")

    # ── 预计算指标 (只算一次，所有参数组合共享) ──
    print("\n[2] Computing indicators...")
    t0 = time.time()
    df_train_indicators = compute_indicators(train_prices)
    df_test_indicators = compute_indicators(test_prices)
    print(f"  Train: {len(df_train_indicators):,} rows")
    print(f"  Test:  {len(df_test_indicators):,} rows")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")

    # ── 市场状态缓存 ──
    print("\n[3] Loading regime cache...")
    try:
        rc = RegimeClassifier()
        bulk_train = rc.classify_bulk()
        regime_cache = {}
        for _, row in bulk_train.iterrows():
            d = str(row['trade_date'])[:10]
            regime_cache[d] = row['regime']
        print(f"  {len(regime_cache)} regime entries loaded")
    except:
        regime_cache = {}
        print("  WARNING: Regime classifier failed, using defaults")

    # ── 网格搜索 ──
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    total_combos = np.prod([len(v) for v in param_values])

    print(f"\n[4] Grid search: {total_combos} combinations...")
    print(f"  Params: {param_names}")

    results = []
    best_train = {'sharpe': -float('inf')}

    for i, combo in enumerate(itertools.product(*param_values)):
        param_dict = dict(zip(param_names, combo))

        # 构建参数
        params = StrategyParams(**param_dict)

        # 训练集回测 (选优)
        train_result = quick_backtest(
            train_prices, df_train_indicators, regime_cache,
            params, TRAIN_START, TRAIN_END
        )

        # 测试集回测 (验证)
        test_result = quick_backtest(
            test_prices, df_test_indicators, regime_cache,
            params, TEST_START, TEST_END
        )

        row = {
            **param_dict,
            'train_cum_return': train_result['cum_return'],
            'train_sharpe': train_result['sharpe'],
            'train_max_dd': train_result['max_dd'],
            'train_trades': train_result['total_trades'],
            'train_win_rate': train_result['win_rate'],
            'test_cum_return': test_result['cum_return'],
            'test_sharpe': test_result['sharpe'],
            'test_max_dd': test_result['max_dd'],
            'test_trades': test_result['total_trades'],
            'test_win_rate': test_result['win_rate'],
        }
        results.append(row)

        # 跟踪最佳训练集参数
        if train_result['sharpe'] > best_train['sharpe']:
            best_train = train_result
            best_train['params'] = param_dict

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{total_combos}... best train Sharpe: {best_train['sharpe']:.2f}")

    # ── 结果排序 ──
    df_results = pd.DataFrame(results)

    # 按训练集 Sharpe 排序
    df_results = df_results.sort_values('train_sharpe', ascending=False)

    print(f"\n{'='*72}")
    print(f"TOP 10 (by train Sharpe)")
    print(f"{'='*72}")
    cols_show = ['adx_threshold', 'vol_ratio_threshold', 'breakout_vol_ratio',
                 'stop_loss_pct', 'trailing_start_pct', 'max_position_pct',
                 'train_sharpe', 'train_max_dd', 'test_sharpe', 'test_max_dd']
    print(df_results[cols_show].head(10).to_string())

    # 按测试集 Sharpe 排序 (泛化能力)
    df_results_by_test = df_results.sort_values('test_sharpe', ascending=False)

    print(f"\n{'='*72}")
    print(f"TOP 10 (by test Sharpe — generalization)")
    print(f"{'='*72}")
    print(df_results_by_test[cols_show].head(10).to_string())

    # ── 过拟合检查 ──
    # 训练集 vs 测试集 Sharpe 散点关系
    train_sharpes = df_results['train_sharpe'].values
    test_sharpes = df_results['test_sharpe'].values
    correlation = np.corrcoef(train_sharpes, test_sharpes)[0, 1]
    print(f"\n  Train-test Sharpe correlation: {correlation:.3f}")
    if correlation < 0.3:
        print("  ✓ 无明显过拟合 (低相关 = 训练好≠测试好)")
    else:
        print("  ⚠ 可能存在过拟合 (高相关可能来自共同的市场beta)")

    # ── 保存 ──
    output_path = os.path.join(OUTPUT_DIR, 'main_surge_grid_results.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\n[5] Results saved: {output_path}")

    return df_results


if __name__ == '__main__':
    main()
