"""牛回头 全量回测 - Top赛道 + 收紧参数"""
import pickle, pandas as pd, numpy as np

BASE = r'D:\AIHedgeFund\ai-hedge-fund-main'

with open(BASE + r'\data\cache\prices_full.pkl', 'rb') as f:
    prices_dict = pickle.load(f)

sw = pd.read_pickle(BASE + r'\data\cache\sw_industry.pkl')
sector_map = {}
for _, row in sw.iterrows():
    sector_map[row['ts_code']] = row['industry']

# Top赛道 (90天均值>7% + n>100, from sector scan)
TOP_SECTORS = {
    '半导体', '纺织', '纺织机械', '专用机械', '轻工机械',
    '铝', '电器仪表', '矿物制品', '特种钢',
}

# 参数收紧
R120_LO, R120_HI = 30, 45       # 收紧r120
DD_LO, DD_HI = -25, -20
SPEED_MIN = 0.5
DAYS_HIGH_LO, DAYS_HIGH_HI = 10, 30  # 距高点10-30天

records = []

for code in prices_dict:
    if code.startswith('688') or code[:3] in ('300', '301'):
        continue
    sec = sector_map.get(code, '')
    if sec not in TOP_SECTORS:
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
        if dd > DD_HI or dd < DD_LO:
            continue
        days_high = e - pi
        decline_speed = abs(dd) / max(days_high, 1)
        if decline_speed < SPEED_MIN:
            continue
        r120 = (c[e] / c[e-120] - 1) * 100
        if r120 < R120_LO or r120 > R120_HI:
            continue
        if days_high < DAYS_HIGH_LO or days_high > DAYS_HIGH_HI:
            continue
        r90 = (c[e+90] / c[e] - 1) * 100
        r180 = (c[min(e+180, n-1)] / c[e] - 1) * 100 if e+180 < n else None
        r360 = (c[min(e+360, n-1)] / c[e] - 1) * 100 if e+360 < n else None
        records.append({
            'sec': sec, 'code': code,
            'date': str(df['trade_date'].iloc[e])[:7],  # year-month
            'year': int(str(df['trade_date'].iloc[e])[:4]),
            'r120': r120, 'dd': dd, 'speed': decline_speed,
            'r90': r90, 'r180': r180, 'r360': r360
        })

df = pd.DataFrame(records)
print(f"Total signals: {len(df)}")
print(f"Year range: {df['year'].min()}-{df['year'].max()}")

# Overall stats
print(f"\n=== Overall (all years) ===")
for period in ['r90', 'r180', 'r360']:
    sub = df.dropna(subset=[period])
    if len(sub) == 0:
        continue
    vals = sub[period]
    wr = (vals > 0).sum() / len(vals) * 100
    big = (vals > 50).sum() / len(vals) * 100
    double = (vals > 100).sum() / len(vals) * 100
    days = int(period[1:])
    annual = ((1 + vals.mean()/100) ** (252/days) - 1) * 100
    med_annual = ((1 + vals.median()/100) ** (252/days) - 1) * 100
    print(f"  {period}: n={len(sub):>5d}  mean={vals.mean():>+6.2f}%  med={vals.median():>+6.2f}%  "
          f"WR={wr:.0f}%  >50%={big:.1f}%  >100%={double:.1f}%  "
          f"年化(均值)={annual:+.0f}%  年化(中位)={med_annual:+.0f}%")

# By year
print(f"\n=== By Year (r90) ===")
sub = df.dropna(subset=['r90'])
print(f"{'Year':>6s} {'n':>5s} {'Mean':>7s} {'Med':>7s} {'WR':>5s} {'<0%':>6s} {'>30%':>6s} {'>50%':>6s}")
print("-" * 55)
for yr in sorted(sub['year'].unique()):
    ydf = sub[sub['year'] == yr]
    wr = (ydf['r90'] > 0).sum() / len(ydf) * 100
    neg = (ydf['r90'] < 0).sum() / len(ydf) * 100
    big30 = (ydf['r90'] > 30).sum() / len(ydf) * 100
    big50 = (ydf['r90'] > 50).sum() / len(ydf) * 100
    print(f" {yr:>4d}  {len(ydf):>5d}  {ydf['r90'].mean():>+6.2f}%  {ydf['r90'].median():>+6.2f}%  "
          f"{wr:>4.0f}%  {neg:>5.0f}%  {big30:>5.0f}%  {big50:>5.0f}%")

# By sector
print(f"\n=== By Sector (r90) ===")
print(f"{'Sector':12s} {'n':>5s} {'Mean':>7s} {'Med':>7s} {'WR':>5s} {'>50%':>6s}")
print("-" * 45)
for sec in sorted(sub['sec'].unique()):
    sdf = sub[sub['sec'] == sec]
    if len(sdf) < 10:
        continue
    wr = (sdf['r90'] > 0).sum() / len(sdf) * 100
    big = (sdf['r90'] > 50).sum() / len(sdf) * 100
    print(f" {sec:<11s} {len(sdf):>5d}  {sdf['r90'].mean():>+6.2f}%  {sdf['r90'].median():>+6.2f}%  {wr:>4.0f}%  {big:>5.0f}%")
