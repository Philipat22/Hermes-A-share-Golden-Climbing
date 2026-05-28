"""DD边界 + 信号密度 — 一次扫描, 多DD滤镜"""
import pickle, pandas as pd, numpy as np, time

BASE = r'D:\AIHedgeFund\ai-hedge-fund-main'
print("Loading...", flush=True)
with open(BASE + r'\data\cache\prices_full.pkl', 'rb') as f:
    prices_dict = pickle.load(f)
sw = pd.read_pickle(BASE + r'\data\cache\sw_industry.pkl')
sector_map = dict(zip(sw['ts_code'], sw['industry']))

MFG_17 = {'半导体', '纺织', '纺织机械', '专用机械', '轻工机械', '铝', '电器仪表', '矿物制品', '特种钢', '铜', '钢加工', '机床制造', '机械基件', '工程机械', '电气设备', '汽车配件', '通信设备'}

MAX_HOLD = 90
R120_LO, R120_HI = 20, 45

# Collect all signals once
records = []
t0 = time.time()
mfg_codes = [c for c in prices_dict if not c.startswith('688') and c[:3] not in ('300','301') and sector_map.get(c,'') in MFG_17]
print(f"Processing {len(mfg_codes)} MFG stocks...", flush=True)

for i, code in enumerate(mfg_codes):
    if i % 100 == 0:
        print(f"  {i}/{len(mfg_codes)} ({i/len(mfg_codes)*100:.0f}%)", flush=True)
    df = prices_dict[code].copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    df = df.dropna(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
    if len(df) < 390: continue
    c = df['close'].values.astype(float)
    n = len(c)
    ma60_arr = np.full(n, np.nan)
    high120_idx = np.full(n, -1, dtype=int)
    for e in range(249, n):
        ma60_arr[e] = np.mean(c[e-59:e+1])
        lb = min(120, e)
        high120_idx[e] = e - lb + np.argmax(c[e-lb:e+1])
    for e in range(249, n - MAX_HOLD):
        if c[e] >= ma60_arr[e]: continue
        pi = high120_idx[e]
        dd = (c[e]/c[pi]-1)*100
        if abs(dd)/max(e-pi,1) < 0.5: continue
        r120 = (c[e]/c[e-120]-1)*100
        if r120 < R120_LO or r120 > R120_HI: continue
        exit_day = MAX_HOLD
        for fwd in range(1, MAX_HOLD+1):
            if e+fwd >= n: exit_day = n-1-e; break
            if c[e+fwd] > ma60_arr[e+fwd]: exit_day = fwd; break
        if e+exit_day >= n: continue
        records.append({'dd': dd, 'ret': (c[e+exit_day]/c[e]-1)*100, 'days': exit_day, 'recovered': exit_day < MAX_HOLD})

df_all = pd.DataFrame(records)
print(f"Total signals: {len(df_all)} in {time.time()-t0:.0f}s\n", flush=True)

# Apply different DD filters to the same signals
dd_configs = [
    ("[-20,-16]", -20, -16),
    ("[-22,-18]", -22, -18),
    ("[-24,-20]", -24, -20),
    ("[-26,-22]", -26, -22),
    ("[-28,-24]", -28, -24),
    ("[-30,-26]", -30, -26),
]

print(f"{'DD':14s} {'n':>6s} {'中位':>7s} {'均值':>7s} {'WR':>5s} {'持有':>5s} {'恢复率':>5s} {'年化(中)':>8s} {'信号/年':>7s} {'N=3?':>5s}")
print("-" * 80)

for cfg_name, dd_lo, dd_hi in dd_configs:
    sub = df_all[(df_all['dd'] >= dd_lo) & (df_all['dd'] < dd_hi)]
    if len(sub) == 0: continue
    wr = (sub['ret']>0).sum()/len(sub)*100
    rec = sub['recovered'].sum()/len(sub)*100
    annual = ((1+sub['ret'].median()/100)**(252/sub['days'].median())-1)*100
    py = len(sub)/17
    enough = "YES" if py >= 20 else ("TIGHT" if py >= 10 else "NO")
    print(f"{cfg_name:<14s} {len(sub):>6d} {sub['ret'].median():>+6.2f}% {sub['ret'].mean():>+6.2f}% {wr:>4.0f}% {sub['days'].median():>5.0f}d {rec:>5.0f}% {annual:>+7.0f}% {py:>6.0f} {enough:>5s}")
