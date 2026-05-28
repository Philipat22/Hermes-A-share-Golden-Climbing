"""WFO参数确认: 深坑边界 + 信号密度"""
import pickle, pandas as pd, numpy as np

BASE = r'D:\AIHedgeFund\ai-hedge-fund-main'

with open(BASE + r'\data\cache\prices_full.pkl', 'rb') as f:
    prices_dict = pickle.load(f)
sw = pd.read_pickle(BASE + r'\data\cache\sw_industry.pkl')
sector_map = dict(zip(sw['ts_code'], sw['industry']))

MFG_17 = {
    '半导体', '纺织', '纺织机械', '专用机械', '轻工机械',
    '铝', '电器仪表', '矿物制品', '特种钢', '铜', '钢加工',
    '机床制造', '机械基件', '工程机械', '电气设备', '汽车配件', '通信设备',
}

MAX_HOLD = 90

# Test deeper DD ranges
# WFO chose [-26,-22]. What about deeper?
dd_configs = [
    ("[-20,-16]", -20, -16),
    ("[-22,-18]", -22, -18),   # our old shallow
    ("[-24,-20]", -24, -20),
    ("[-26,-22]", -26, -22),   # WFO choice
    ("[-28,-24]", -28, -24),   # deeper
    ("[-30,-26]", -30, -26),   # deeper still
]

r120_lo, r120_hi = 20, 45

print("=== DD Boundary Scan (r120 20-45%, 17 manufacturing) ===")
print(f"{'DD':14s} {'n':>6s} {'中位':>7s} {'均值':>7s} {'WR':>5s} {'持有':>5s} {'恢复率':>5s} {'年化(中)':>8s} {'信号/年':>7s} {'N=3够?':>7s}")
print("-" * 80)

for cfg_name, dd_lo, dd_hi in dd_configs:
    records = []
    for code in prices_dict:
        if code.startswith('688') or code[:3] in ('300', '301'):
            continue
        if sector_map.get(code,'') not in MFG_17:
            continue  # ONLY MFG_17
        df = prices_dict[code].copy()
        df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
        df = df.dropna(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
        if len(df) < 390:
            continue
        c = df['close'].values.astype(float)
        n = len(c)
        ma60_arr = np.full(n, np.nan)
        high120_idx = np.full(n, -1, dtype=int)
        for e in range(249, n):
            ma60_arr[e] = np.mean(c[e-59:e+1])
            lb = min(120, e)
            high120_idx[e] = e - lb + np.argmax(c[e-lb:e+1])
        
        for e in range(249, n - MAX_HOLD):
            if c[e] >= ma60_arr[e]:
                continue
            pi = high120_idx[e]
            dd = (c[e] / c[pi] - 1) * 100
            if dd > dd_hi or dd < dd_lo:
                continue
            if abs(dd) / max(e - pi, 1) < 0.5:
                continue
            r120 = (c[e] / c[e-120] - 1) * 100
            if r120 < r120_lo or r120 > r120_hi:
                continue
            
            exit_day = MAX_HOLD
            for fwd in range(1, MAX_HOLD + 1):
                if e+fwd >= n:
                    exit_day = n - 1 - e
                    break
                if c[e+fwd] > ma60_arr[e+fwd]:
                    exit_day = fwd
                    break
            if e + exit_day >= n:
                continue
            records.append({'ret': (c[e+exit_day]/c[e]-1)*100, 'days': exit_day, 'recovered': exit_day < MAX_HOLD})
    
    if records:
        df_r = pd.DataFrame(records)
        wr = (df_r['ret']>0).sum()/len(df_r)*100
        rec = df_r['recovered'].sum()/len(df_r)*100
        annual = ((1+df_r['ret'].median()/100)**(252/df_r['days'].median())-1)*100
        per_year = len(df_r)/17
        enough = "YES" if per_year >= 20 else ("TIGHT" if per_year >= 10 else "NO")
        print(f"{cfg_name:<14s} {len(df_r):>6d} {df_r['ret'].median():>+6.2f}% {df_r['ret'].mean():>+6.2f}% {wr:>4.0f}% {df_r['days'].median():>5.0f}d {rec:>5.0f}% {annual:>+7.0f}% {per_year:>6.0f} {enough:>7s}")
