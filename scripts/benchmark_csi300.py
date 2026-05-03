"""
CSI300 Buy & Hold Benchmark
比较 V2r2 ML 回测 vs 简单持有指数，回答核心问题：
是策略烂，还是市场烂？

用法: python scripts/benchmark_csi300.py
"""
import os, sys, warnings
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import tushare as ts

ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
load_dotenv(os.path.join(ROOT, '.env'))

pro = ts.pro_api(os.getenv('TUSHARE_PRO_TOKEN', ''))
TOKEN = os.getenv('TUSHARE_PRO_TOKEN', '')

# V2r2 回测同期: 2022-01-01 → 2026-04-29
START = '20220101'
END = '20260429'

print("=" * 60)
print("CSI300 Buy & Hold 基准回测")
print(f"周期: {START[:4]}-{START[4:6]}-{START[6:]} → {END[:4]}-{END[4:6]}-{END[6:]}")
print("=" * 60)

# ── 获取 CSI300 指数数据 ──
csi = pro.index_daily(ts_code='000300.SH', start_date=START, end_date=END)
if csi is None or len(csi) == 0:
    print("ERROR: 无法获取 CSI300 数据，请检查 Token 和网络")
    sys.exit(1)

csi['trade_date'] = pd.to_datetime(csi['trade_date'])
csi = csi.sort_values('trade_date').reset_index(drop=True)

# ── 基准收益 ──
initial = csi['close'].iloc[0]
final = csi['close'].iloc[-1]
total_ret = final / initial - 1
years = (csi['trade_date'].iloc[-1] - csi['trade_date'].iloc[0]).days / 365.25
annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

# 年化波动率
daily_ret = csi['close'].pct_change().dropna()
annual_vol = daily_ret.std() * np.sqrt(252) if len(daily_ret) > 0 else 0
sharpe = (annual_ret - 0.03) / annual_vol if annual_vol > 0 else 0

# 最大回撤
cummax = csi['close'].cummax()
drawdown = csi['close'] / cummax - 1
max_dd = drawdown.min()

print(f"\n  CSI300 基准:")
print(f"    起始: {initial:.0f} → 终值: {final:.0f}")
print(f"    累计收益: {total_ret:+.2%}")
print(f"    年化收益: {annual_ret:+.2%}")
print(f"    年化波动: {annual_vol:.1%}")
print(f"    夏普比率: {sharpe:.2f}")
print(f"    最大回撤: {max_dd:.2%} ({drawdown.idxmin().strftime('%Y-%m-%d') if hasattr(drawdown.idxmin(), 'strftime') else 'N/A'})")

# ── 分年统计 ──
csi['year'] = csi['trade_date'].dt.year
print(f"\n  分年收益:")
for yr in sorted(csi['year'].unique()):
    sub = csi[csi['year'] == yr]
    if len(sub) > 1:
        yr_ret = sub['close'].iloc[-1] / sub['close'].iloc[0] - 1
        marker = " ← V2r2崩盘年" if yr >= 2025 else ""
        print(f"    {yr}: {yr_ret:+.2%}{marker}")

# ── 与 V2r2 对比 ──
print(f"\n  ═══ 对比 ═══")
print(f"  指标           | CSI300基准    | V2r2 ML策略")
print(f"  ─────────────┼────────────┼────────────")
print(f"  累计收益      | {total_ret:+6.2%}      | {'-43.34%':>8}")
print(f"  最大回撤      | {max_dd:+6.2%}      | {'-92.02%':>8}")
print(f"  夏普比率      | {sharpe:6.2f}        | {'0.49':>6}")

if total_ret > -0.43:
    print(f"\n  ⚠️  结论: ML策略(-43%)跑输简单持有CSI300({total_ret:+.1%})")
    print(f"      差距: {total_ret - (-0.4334):+.1%} — 策略在帮倒忙")
else:
    print(f"\n  📊 结论: 市场整体下跌 {total_ret:+.1%}，策略-43%更差但差距不大")
    print(f"      策略只比基准多亏 {(-0.4334 - total_ret):.1%}")

# ── Regime分布 ──
csi['ma20'] = csi['close'].rolling(20).mean()
csi['ma60'] = csi['close'].rolling(60).mean()
csi['ma120'] = csi['close'].rolling(120).mean()
csi['ret20'] = csi['close'].pct_change(20)

def classify(row):
    c = row['close']; m20 = row['ma20']; m60 = row['ma60']; m120 = row['ma120']; r20 = row['ret20']
    if pd.isna(m60) or pd.isna(m120): return 'unknown'
    if c < m120 * 0.90: return 'severe_bear'
    if c < m60 and c < m120 and r20 < -0.03: return 'bear'
    if c > m20 > m60 and r20 > 0.03: return 'bull'
    if c > m60 and r20 > 0: return 'recovery'
    return 'sideways'

csi['regime'] = csi.apply(classify, axis=1)
print(f"\n  Regime分布 (2022-2026):")
for r in ['bull', 'sideways', 'bear', 'severe_bear', 'recovery']:
    cnt = (csi['regime'] == r).sum()
    pct = cnt / len(csi) * 100
    sub = csi[csi['regime'] == r]
    r_avg = sub['ret20'].mean() * 100 if len(sub) > 0 else 0
    print(f"    {r:12s}: {cnt:4d} 天 ({pct:5.1f}%)  平均20日收益: {r_avg:+.1f}%")

print("\nDone!")
