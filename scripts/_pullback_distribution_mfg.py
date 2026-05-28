"""牛回头 r120梯度 — 仅制造业赛道"""
import pickle, pandas as pd, numpy as np

BASE = r'D:\AIHedgeFund\ai-hedge-fund-main'

with open(BASE + r'\data\cache\prices_full.pkl', 'rb') as f:
    prices_dict = pickle.load(f)

sw = pd.read_pickle(BASE + r'\data\cache\sw_industry.pkl')
sector_map = {}
for _, row in sw.iterrows():
    sector_map[row['ts_code']] = row['industry']

# 高端制造赛道 (from previous scan: mean>5%, WR>55%)
MFG_SECTORS = {
    '纺织机械', '电器仪表', '专用机械', '铜', '铝', '纺织',
    '半导体', '通信设备', '钢加工', '特种钢', '机床制造', '机械基件',
    '工程机械', '电气设备', '矿物制品', '轻工机械', '汽车配件',
}

records = []

for code in prices_dict:
    if code.startswith('688') or code[:3] in ('300', '301'):
        continue
    sec = sector_map.get(code, '')
    if sec not in MFG_SECTORS:
        continue
    df = prices_dict[code].copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    df = df.dropna(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
    if len(df) < 390:
        continue
    c = df['close'].values.astype(float)
    n = len(c)
    for e in range(249, n - 90):
        ma60 = np.mean(c[e-59:e+1])
        if c[e] > ma60:
            continue
        lb = min(120, e)
        pi = e - lb + np.argmax(c[e-lb:e+1])
        dd = (c[e] / c[pi] - 1) * 100
        if dd > -20 or dd < -25:
            continue
        decline_speed = abs(dd) / (e - pi) if e > pi else 0
        if decline_speed < 0.5:
            continue
        r120 = (c[e] / c[e-120] - 1) * 100
        if r120 <= 20 or r120 > 50:
            continue
        r90 = (c[e+90] / c[e] - 1) * 100
        records.append({
            'r120': r120, 'r90': r90, 'sec': sec
        })

df = pd.DataFrame(records)
print(f"Total manufacturing signals: {len(df)}")
print(f"Overall: mean r90={df['r90'].mean():+.2f}% median={df['r90'].median():+.2f}% WR={(df['r90']>0).sum()/len(df)*100:.1f}%")

print(f"\n=== r120 Gradient (Manufacturing Only) ===")
print(f"{'r120区间':15s} {'n':>6s} {'均值':>8s} {'中位':>8s} {'胜率':>6s} {'>50%':>6s}")
print("-" * 50)
for lo, hi in [(20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50)]:
    sub = df[(df['r120'] >= lo) & (df['r120'] < hi)]
    ws = (sub['r90'] > 0).sum() / len(sub) * 100 if len(sub) > 0 else 0
    big = (sub['r90'] > 50).sum() / len(sub) * 100 if len(sub) > 0 else 0
    direction = "↑" if sub['r90'].median() > 0 else "↓"
    print(f"r120 [{lo:>2d}~{hi:>2d}%)  {len(sub):>6d}  {sub['r90'].mean():>+7.2f}%  {sub['r90'].median():>+7.2f}%  {ws:>5.0f}%  {big:>5.1f}%  {direction}")

# Also split by sector
print(f"\n=== By Sector (top 12) ===")
print(f"{'行业':12s} {'n':>6s} {'均值':>8s} {'中位':>8s} {'胜率':>6s} {'>50%':>6s}")
print("-" * 50)
sector_stats = []
for sec in sorted(df['sec'].unique()):
    sub = df[df['sec'] == sec]
    if len(sub) < 50:
        continue
    ws = (sub['r90'] > 0).sum() / len(sub) * 100
    big = (sub['r90'] > 50).sum() / len(sub) * 100
    sector_stats.append((sec, len(sub), sub['r90'].mean(), sub['r90'].median(), ws, big))

sector_stats.sort(key=lambda x: -x[2])  # sort by mean
for sec, n, mean, med, ws, big in sector_stats[:15]:
    print(f"{sec:<12s} {n:>6d}  {mean:>+7.2f}%  {med:>+7.2f}%  {ws:>5.0f}%  {big:>5.1f}%")
