"""Walk-forward 快速版 — 分批处理 + 进度输出"""
import pickle, pandas as pd, numpy as np, sys, time

BASE = r'D:\AIHedgeFund\ai-hedge-fund-main'

print("Loading data...", flush=True)
with open(BASE + r'\data\cache\prices_full.pkl', 'rb') as f:
    prices_dict = pickle.load(f)
sw = pd.read_pickle(BASE + r'\data\cache\sw_industry.pkl')
sector_map = dict(zip(sw['ts_code'], sw['industry']))

MFG_17 = {
    '半导体', '纺织', '纺织机械', '专用机械', '轻工机械',
    '铝', '电器仪表', '矿物制品', '特种钢', '铜', '钢加工',
    '机床制造', '机械基件', '工程机械', '电气设备', '汽车配件', '通信设备',
}

# Pre-filter to MFG stocks only
mfg_codes = [c for c in prices_dict 
             if not c.startswith('688') and c[:3] not in ('300','301')
             and sector_map.get(c,'') in MFG_17]
print(f"MFG stocks: {len(mfg_codes)}", flush=True)

MAX_HOLD = 90
records = []
t0 = time.time()

for i, code in enumerate(mfg_codes):
    if i % 100 == 0:
        elapsed = time.time() - t0
        pct = i / len(mfg_codes) * 100
        eta = elapsed / max(i, 1) * (len(mfg_codes) - i)
        print(f"  {i}/{len(mfg_codes)} ({pct:.0f}%) elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
    
    df = prices_dict[code].copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    df = df.dropna(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
    if len(df) < 390:
        continue
    
    c = df['close'].values.astype(float)
    n = len(c)
    
    # Pre-compute MA60 and 120d high index (vectorized)
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
        decline_speed = abs(dd) / max(e - pi, 1)
        if decline_speed < 0.5:
            continue
        r120 = (c[e] / c[e-120] - 1) * 100
        
        # MA60 exit
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
        
        year = int(str(df['trade_date'].iloc[e])[:4])
        records.append({
            'year': year, 'dd': dd, 'r120': r120,
            'speed': decline_speed, 'ret': (c[e+exit_day]/c[e]-1)*100,
            'days': exit_day
        })

df_all = pd.DataFrame(records)
print(f"\nTotal signals: {len(df_all)} in {time.time()-t0:.0f}s", flush=True)

# Walk-forward grid search
dd_grid = [(-26,-22), (-24,-20), (-22,-18), (-20,-16), (-18,-14)]
r120_grid = [(15,55), (20,50), (20,45), (25,50), (25,45), (30,50)]

print(f"\n=== Walk-Forward ===")
print(f"{'Year':>5s} {'FIX_n':>6s} {'FIX_med':>8s} {'FIX_WR':>6s} | {'WFO_n':>5s} {'WFO_med':>8s} {'WFO_WR':>6s} | Params")
print("-" * 90)

fixed_meds, wfo_meds = [], []
all_params_seen = []

for test_year in range(2009, 2026):
    train = df_all[df_all['year'] < test_year]
    test = df_all[df_all['year'] == test_year]
    
    if len(train) < 50 or len(test) < 5:
        continue
    
    # Fixed
    f_test = test[(test['dd']>=-22)&(test['dd']<=-18)&(test['r120']>=20)&(test['r120']<=50)&(test['speed']>=0.5)]
    f_med = f_test['ret'].median() if len(f_test)>0 else 0
    f_n = len(f_test)
    f_wr = (f_test['ret']>0).sum()/max(f_n,1)*100
    
    # WFO
    best_med, best_params = -999, None
    for dd_lo, dd_hi in dd_grid:
        for r120_lo, r120_hi in r120_grid:
            tr = train[(train['dd']>=dd_lo)&(train['dd']<=dd_hi)&(train['r120']>=r120_lo)&(train['r120']<=r120_hi)&(train['speed']>=0.5)]
            if len(tr) < 20: continue
            if tr['ret'].median() > best_med:
                best_med = tr['ret'].median()
                best_params = (dd_lo, dd_hi, r120_lo, r120_hi)
    
    if best_params:
        dd_lo, dd_hi, r120_lo, r120_hi = best_params
        w_test = test[(test['dd']>=dd_lo)&(test['dd']<=dd_hi)&(test['r120']>=r120_lo)&(test['r120']<=r120_hi)&(test['speed']>=0.5)]
        w_med = w_test['ret'].median() if len(w_test)>0 else 0
        w_n = len(w_test)
        w_wr = (w_test['ret']>0).sum()/max(w_n,1)*100
        all_params_seen.append(best_params)
    else:
        w_med, w_n, w_wr = 0, 0, 0
    
    params_str = f"DD{best_params[0]}~{best_params[1]} r{best_params[2]}~{best_params[3]}" if best_params else "N/A"
    print(f" {test_year:>4d} {f_n:>6d} {f_med:>+7.2f}% {f_wr:>5.0f}% | {w_n:>5d} {w_med:>+7.2f}% {w_wr:>5.0f}% | {params_str}")
    
    if f_n > 0: fixed_meds.append(f_med)
    if w_n > 0: wfo_meds.append(w_med)
    if best_params: all_params_seen.append(best_params)

print(f"\n=== Summary ===")
print(f"FIXED: avg_med={np.mean(fixed_meds):+.2f}%  WR={np.mean([f_wr for _ in range(len(fixed_meds))]):.0f}%")
print(f"WFO:   avg_med={np.mean(wfo_meds):+.2f}%  WR={np.mean([w_wr for _ in range(len(wfo_meds))]):.0f}%")

# Parameter drift
if all_params_seen:
    dd_los = [p[0] for p in all_params_seen]
    dd_his = [p[1] for p in all_params_seen]
    print(f"\nParameter drift:")
    print(f"  DD_lo:  min={min(dd_los)} max={max(dd_los)} mode={max(set(dd_los),key=dd_los.count)}")
    print(f"  DD_hi:  min={min(dd_his)} max={max(dd_his)} mode={max(set(dd_his),key=dd_his.count)}")
