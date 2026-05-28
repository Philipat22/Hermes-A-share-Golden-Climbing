"""牛回头 - 金属/新能源/电力板块扫描"""
import pickle, pandas as pd, numpy as np

BASE = r'D:\AIHedgeFund\ai-hedge-fund-main'

with open(BASE + r'\data\cache\prices_full.pkl', 'rb') as f:
    prices_dict = pickle.load(f)

sw = pd.read_pickle(BASE + r'\data\cache\sw_industry.pkl')
sector_map = {}
for _, row in sw.iterrows():
    sector_map[row['ts_code']] = row['industry']

TARGET = ['小金属','铅锌','特种钢','钢加工','普钢','黄金',
          '电气设备','新型电力','火力发电','水力发电','矿物制品']
sector_results = {}

for code in prices_dict:
    if code.startswith('688') or code[:3] in ('300','301'):
        continue
    sec = sector_map.get(code, '')
    if sec not in TARGET:
        continue
    df = prices_dict[code].copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    df = df.dropna(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
    if len(df) < 300:
        continue
    c = df['close'].values.astype(float)
    n = len(c)
    for e in range(249, n - 45):
        ma60 = np.mean(c[e-59:e+1])
        if c[e] > ma60:
            continue
        lb = min(120, e)
        pi = e - lb + np.argmax(c[e-lb:e+1])
        dd = (c[e] / c[pi] - 1) * 100
        if dd > -20 or dd < -25:
            continue
        if abs(dd)/(e-pi) < 0.5:
            continue
        r120 = (c[e] / c[e-120] - 1) * 100
        if r120 <= 20 or r120 > 50:
            continue
        r30 = (c[min(e+30, n-1)] / c[e] - 1) * 100
        if sec not in sector_results:
            sector_results[sec] = []
        sector_results[sec].append(r30)

print(f"{'行业':12s} {'n':>5s} {'均值':>7s} {'中位':>7s} {'胜率':>6s} {'年化':>7s}")
print("-" * 50)

for sec in TARGET:
    if sec not in sector_results:
        print(f"{sec:<12s} 无数据")
        continue
    r = sector_results[sec]
    if len(r) < 10:
        print(f"{sec:<12s} {len(r):>5d} (样本不足)")
        continue
    r = np.array(r)
    wr = (r > 0).sum() / len(r) * 100
    annual = ((1 + np.mean(r)/100) ** (252/30) - 1) * 100
    marker = " ***" if np.mean(r) > 5 and wr > 55 else (" !!" if np.mean(r) > 3 else "")
    print(f"{sec:<12s} {len(r):>5d} {np.mean(r):>+6.1f}% {np.median(r):>+6.1f}% {wr:>5.0f}% {annual:>+6.0f}%{marker}")
