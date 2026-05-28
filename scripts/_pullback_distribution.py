"""牛回头 - 收益分布 + 特征分析"""
import pickle, pandas as pd, numpy as np

BASE = r'D:\AIHedgeFund\ai-hedge-fund-main'

with open(BASE + r'\data\cache\prices_full.pkl', 'rb') as f:
    prices_dict = pickle.load(f)

sw = pd.read_pickle(BASE + r'\data\cache\sw_industry.pkl')
sector_map = {}
for _, row in sw.iterrows():
    sector_map[row['ts_code']] = row['industry']

records = []

for code in prices_dict:
    if code.startswith('688') or code[:3] in ('300', '301'):
        continue
    sec = sector_map.get(code, '')
    df = prices_dict[code].copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    df = df.dropna(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
    if len(df) < 390:  # need 120 lookback + 90 forward
        continue
    c = df['close'].values.astype(float)
    v = df['vol'].values.astype(float) if 'vol' in df.columns else np.ones(len(c))
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
        r30 = (c[e+30] / c[e] - 1) * 100
        r60 = (c[e+60] / c[e] - 1) * 100
        r90 = (c[e+90] / c[e] - 1) * 100
        days_high = e - pi
        vol20 = np.mean(v[max(0, e-20):e+1])
        vol_ratio = v[e] / vol20 if vol20 > 0 else 0
        records.append({
            'code': code, 'sec': sec, 'date': str(df['trade_date'].iloc[e])[:10],
            'r120': r120, 'dd': dd, 'speed': decline_speed,
            'days_high': days_high, 'vol_ratio': vol_ratio,
            'r30': r30, 'r60': r60, 'r90': r90
        })

df = pd.DataFrame(records)
print(f"Total signals: {len(df)}")

# === 1. Distribution ===
print(f"\n=== Return Distribution (90d) ===")
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  P{p:>2d}: {np.percentile(df['r90'], p):>+6.2f}%")

print(f"\n  Mean: {df['r90'].mean():+.2f}%")
print(f"  Median: {df['r90'].median():+.2f}%")
print(f"  Win Rate: {(df['r90']>0).sum()/len(df)*100:.1f}%")

# === 2. Buckets ===
print(f"\n=== Return Buckets (90d) ===")
bins = [(-100, -20), (-20, -10), (-10, 0), (0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 9999)]
for lo, hi in bins:
    cnt = ((df['r90'] >= lo) & (df['r90'] < hi)).sum()
    pct = cnt / len(df) * 100
    bar = "#" * int(pct)
    print(f"  [{lo:>4.0f}% ~ {hi:>4.0f}%): {cnt:>5d} ({pct:>5.1f}%) {bar}")

# === 3. Feature analysis: what predicts rocket? ===
print(f"\n=== Feature Split: Rocket(>30%) vs Rebound(0~10%) vs Loser(<0%) ===")
df['group'] = 'mid'
df.loc[df['r90'] > 30, 'group'] = 'rocket'
df.loc[(df['r90'] >= 0) & (df['r90'] <= 10), 'group'] = 'rebound'
df.loc[df['r90'] < 0, 'group'] = 'loser'

for feat in ['r120', 'dd', 'speed', 'days_high', 'vol_ratio']:
    print(f"\n  --- {feat} ---")
    for g in ['rocket', 'rebound', 'loser']:
        gdf = df[df['group'] == g]
        if len(gdf) == 0:
            continue
        vals = gdf[feat]
        print(f"  {g:>8s} (n={len(gdf):>4d}): mean={vals.mean():>+7.2f}  med={vals.median():>+7.2f}  std={vals.std():>6.2f}")

# === 4. r120 gradient within 20-50% ===
print(f"\n=== r120 Gradient ===")
for lo, hi in [(20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50)]:
    sub = df[(df['r120'] >= lo) & (df['r120'] < hi)]
    ws = (sub['r90'] > 0).sum() / len(sub) * 100 if len(sub) > 0 else 0
    print(f"  r120 [{lo}-{hi}%): n={len(sub):>4d}  median={sub['r90'].median():>+7.2f}%  mean={sub['r90'].mean():>+7.2f}%  WR={ws:.0f}%")

# === 5. DD gradient ===
print(f"\n=== DD Gradient ===")
for lo, hi in [(-25, -24), (-24, -23), (-23, -22), (-22, -21), (-21, -20)]:
    sub = df[(df['dd'] >= lo) & (df['dd'] < hi)]
    ws = (sub['r90'] > 0).sum() / len(sub) * 100 if len(sub) > 0 else 0
    print(f"  DD [{lo}~{hi}%): n={len(sub):>4d}  median={sub['r90'].median():>+7.2f}%  mean={sub['r90'].mean():>+7.2f}%  WR={ws:.0f}%")

# === 6. Days since high gradient ===
print(f"\n=== Days Since High Gradient ===")
for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 30), (30, 999)]:
    sub = df[(df['days_high'] >= lo) & (df['days_high'] < hi)]
    ws = (sub['r90'] > 0).sum() / len(sub) * 100 if len(sub) > 0 else 0
    print(f"  days [{lo:>3d}-{hi:>3d}): n={len(sub):>4d}  median={sub['r90'].median():>+7.2f}%  mean={sub['r90'].mean():>+7.2f}%  WR={ws:.0f}%")
