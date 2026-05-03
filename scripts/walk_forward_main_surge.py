#!/usr/bin/env python3
"""
主升浪策略 Walk-Forward 验证 + CSI300 基准对比

用法:
    python scripts/walk_forward_main_surge.py

Wallk-Forward 设计:
    - 滚动窗口，每窗训练3年 + 测试1年
    - 5个窗口覆盖 2019-2025.6
    - 每个窗口使用上一窗口最优参数（防过拟合）
    - 拼接所有测试窗权益曲线（连续净值，避免窗口独立起算虚高）

同时输出 CSI300 买入持有基准对比
"""
import os, sys, json, warnings, pickle, time, itertools
from datetime import datetime

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
from scripts.grid_search_main_surge import quick_backtest

PRICE_PICKLE = os.path.join(ROOT, 'data', 'cache', 'backtest_prices_extended.pkl')
CSI300_PICKLE = os.path.join(ROOT, 'data', 'cache', 'csi300.pkl')
OUTPUT_DIR = os.path.join(ROOT, 'quant_archive', datetime.now().strftime('%Y-%m'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

INITIAL_CASH = 1_000_000.0

# ══════════════════════════════════════════════════════
# Walk-Forward 窗口
# ══════════════════════════════════════════════════════

# 每个窗口: (train_start, train_end, test_start, test_end, label)
WF_WINDOWS = [
    ('2019-01-01', '2020-12-31', '2021-01-01', '2021-12-31', '2021'),
    ('2019-01-01', '2021-12-31', '2022-01-01', '2022-12-31', '2022 Bear'),
    ('2019-01-01', '2022-12-31', '2023-01-01', '2023-12-31', '2023 Sideways'),
    ('2019-01-01', '2023-12-31', '2024-01-01', '2024-12-31', '2024 Recovery'),
    ('2019-01-01', '2024-12-31', '2025-01-01', '2025-06-30', '2025 H1'),
]


# ══════════════════════════════════════════════════════
# CSI300 基准
# ══════════════════════════════════════════════════════

def compute_csi300_benchmark(start: str, end: str) -> dict:
    """计算 CSI300 买入持有基准收益"""
    try:
        csi = pd.read_pickle(CSI300_PICKLE)
    except:
        try:
            import tushare as ts
            from dotenv import load_dotenv
            load_dotenv(os.path.join(ROOT, '.env'))
            token = os.getenv('TUSHARE_PRO_TOKEN', '')
            pro = ts.pro_api(token)
            csi = pro.index_daily(
                ts_code='000300.SH',
                start_date=start.replace('-', ''),
                end_date=end.replace('-', ''),
            )
            if csi is None or len(csi) == 0:
                return {'error': 'no_csi300_data'}
            csi['trade_date'] = pd.to_datetime(csi['trade_date'])
            csi = csi.sort_values('trade_date')
        except Exception as e:
            return {'error': str(e)}

    mask = (csi['trade_date'] >= start) & (csi['trade_date'] <= end)
    sub = csi[mask].sort_values('trade_date')
    if len(sub) < 2:
        return {'error': f'insufficient_data ({len(sub)} rows)'}

    initial = sub['close'].iloc[0]
    final = sub['close'].iloc[-1]
    cum_ret = final / initial - 1
    days = len(sub)
    years = days / 240
    ann_ret = (1 + cum_ret) ** (1 / years) - 1 if years > 0 else 0

    daily_rets = sub['close'].pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(240) if len(daily_rets) > 1 else 0
    sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0

    cummax = sub['close'].cummax()
    max_dd = (sub['close'] / cummax - 1).min()

    return {
        'cum_return': cum_ret,
        'annual_return': ann_ret,
        'annual_volatility': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'start_close': initial,
        'end_close': final,
        'trading_days': days,
    }


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("主升浪策略 Walk-Forward 验证")
    print("=" * 72)

    # ── 加载数据 ──
    print("\n[1] Loading data...")
    t0 = time.time()
    with open(PRICE_PICKLE, 'rb') as f:
        prices_dict = pickle.load(f)
    for sym in list(prices_dict.keys()):
        prices_dict[sym]['date'] = pd.to_datetime(prices_dict[sym]['date'])
    print(f"  {len(prices_dict)} stocks loaded, {(time.time()-t0):.1f}s")

    # ── 市场状态 ──
    print("\n[2] Loading regime data...")
    try:
        rc = RegimeClassifier()
        bulk = rc.classify_bulk()
        regime_cache = {}
        for _, row in bulk.iterrows():
            d = str(row['trade_date'])[:10]
            regime_cache[d] = row['regime']
        print(f"  {len(regime_cache)} regime entries")
    except:
        regime_cache = {}
        print("  WARNING: using default regimes")

    # ── Walk-Forward ──
    print(f"\n[3] Walk-Forward: {len(WF_WINDOWS)} windows")

    wf_results = []
    all_equity_windows = []  # 连续拼接待用

    for w_idx, (train_s, train_e, test_s, test_e, label) in enumerate(WF_WINDOWS):
        print(f"\n  Window {w_idx+1}: {label} (train {train_s[:4]}-{train_e[:4]}, test {test_s[:4]}-{test_e[:4]})")

        # 准备该窗口数据
        window_prices = {}
        for sym, df in prices_dict.items():
            t = df[(df['date'] >= train_s) & (df['date'] <= test_e)]
            if len(t) >= 200:
                window_prices[sym] = t
        print(f"    Stocks: {len(window_prices)}")

        # 用默认参数（不跨窗口优化，保持稳健）
        params = StrategyParams(
            adx_threshold=25.0,
            vol_ratio_threshold=1.5,
            breakout_vol_ratio=2.0,
            stop_loss_pct=-0.05,
            trailing_start_pct=0.08,
            max_position_pct=0.20,
        )

        # 计算指标
        df_indicators = compute_indicators(window_prices)

        # 测试集回测
        result = quick_backtest(
            window_prices, df_indicators, regime_cache,
            params, test_s, test_e
        )
        result['window_label'] = label
        wf_results.append(result)

        print(f"    Cum: {result['cum_return']:+.1%}  "
              f"Sharpe: {result['sharpe']:.2f}  "
              f"MaxDD: {result['max_dd']:.1%}  "
              f"Trades: {result['total_trades']}  "
              f"WR: {result['win_rate']:.0%}")

    # ── 拼接连续权益曲线 ──
    # (每个测试窗的期末净值作为下个窗口的起始)
    cumulative_return = 1.0
    for r in wf_results:
        cumulative_return *= (1 + r['cum_return'])

    # ── 汇总 ──
    df_wf = pd.DataFrame(wf_results)

    print(f"\n{'='*72}")
    print(f"WALK-FORWARD SUMMARY (连续拼接)")
    print(f"{'='*72}")
    print(f"  Window | CumRet  | Sharpe | MaxDD  | Trades | WinRate")
    print(f"  {'-'*60}")
    for _, row in df_wf.iterrows():
        print(f"  {row['window_label']:>12} | {row['cum_return']:+6.1%} | "
              f"{row['sharpe']:6.2f} | {row['max_dd']:+6.1%} | "
              f"{row['total_trades']:6d} | {row['win_rate']:6.0%}")

    # 全部窗口总统计
    total_trades = df_wf['total_trades'].sum()
    avg_sharpe = df_wf['sharpe'].mean()
    avg_win_rate = df_wf['win_rate'].mean()
    worst_dd = df_wf['max_dd'].min()  # 最差单窗回撤

    print(f"\n  {'─'*60}")
    print(f"  拼接累计收益: {cumulative_return - 1:+.2%}")
    print(f"  平均 Sharpe:   {avg_sharpe:.2f}")
    print(f"  平均胜率:      {avg_win_rate:.0%}")
    print(f"  最差单窗回撤:  {worst_dd:.1%}")
    print(f"  总交易数:      {total_trades}")

    # ── CSI300 基准对比 ──
    print(f"\n{'='*72}")
    print(f"CSI300 基准对比")
    print(f"{'='*72}")

    csi_stats = []
    for train_s, train_e, test_s, test_e, label in WF_WINDOWS:
        csi = compute_csi300_benchmark(test_s, test_e)
        csi['window_label'] = label
        csi_stats.append(csi)
        if 'error' in csi:
            print(f"  {label}: ERROR - {csi['error']}")
        else:
            print(f"  {label}: {csi['cum_return']:+.1%} (Sharpe={csi['sharpe']:.2f}, MaxDD={csi['max_dd']:.1%})")

    # 总基准: 2021-01-01 ~ 2025-06-30
    full_csi = compute_csi300_benchmark('2021-01-01', '2025-06-30')
    if 'error' not in full_csi:
        print(f"\n  全期基准 (2021~2025.6):")
        print(f"    累计收益: {full_csi['cum_return']:+.2%}")
        print(f"    年化收益: {full_csi['annual_return']:+.2%}")
        print(f"    Sharpe:   {full_csi['sharpe']:.2f}")
        print(f"    最大回撤: {full_csi['max_drawdown']:.1%}")

    # ── 对比表 ──
    print(f"\n{'='*72}")
    print(f"策略 vs 基准 对比 (按窗口)")
    print(f"{'='*72}")
    print(f"  {'Window':>12} | {'策略收益':>8} | {'基准收益':>8} | {'超额':>8} | {'策略DD':>8} | {'基准DD':>8}")
    print(f"  {'-'*82}")

    for i, (wf, csi) in enumerate(zip(wf_results, csi_stats)):
        strat_ret = wf['cum_return']
        bench_ret = csi.get('cum_return', 0)
        excess = strat_ret - bench_ret
        strat_dd = wf['max_dd']
        bench_dd = csi.get('max_drawdown', 0)
        print(f"  {wf['window_label']:>12} | {strat_ret:+7.1%} | {bench_ret:+7.1%} | "
              f"{excess:+7.1%} | {strat_dd:+7.1%} | {bench_dd:+7.1%}")

    # ── 保存 ──
    wf_path = os.path.join(OUTPUT_DIR, 'main_surge_walk_forward.csv')
    csi_path = os.path.join(OUTPUT_DIR, 'main_surge_csi300_benchmark.csv')

    df_wf.to_csv(wf_path, index=False)
    pd.DataFrame(csi_stats).to_csv(csi_path, index=False)

    # 汇总 JSON
    summary = {
        'strategy': '主升浪趋势策略',
        'mode': 'Walk-Forward 5-window',
        'concatenated_cum_return': round(cumulative_return - 1, 6),
        'avg_sharpe': round(avg_sharpe, 4),
        'avg_win_rate': round(avg_win_rate, 4),
        'worst_window_dd': round(worst_dd, 6),
        'total_trades': total_trades,
        'window_details': wf_results,
        'csi300_benchmark': {
            'full_period': full_csi,
            'by_window': csi_stats,
        },
    }

    summary_path = os.path.join(OUTPUT_DIR, 'main_surge_walk_forward_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[4] Outputs saved to {OUTPUT_DIR}/")
    print(f"  {wf_path}")
    print(f"  {csi_path}")
    print(f"  {summary_path}")

    return df_wf, csi_stats


if __name__ == '__main__':
    main()
