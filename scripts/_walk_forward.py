"""牛回头 逐年Walk-Forward Optimization
每年用历史数据优化DD和r120, 在下一年验证
对比: 固定参数 vs 年度重优化"""
import pickle, pandas as pd, numpy as np

BASE = r'D:\AIHedgeFund\ai-hedge-fund-main'

with open(BASE + r'\data\cache\prices_full.pkl', 'rb') as f:
    prices_dict = pickle.load(f)

sw = pd.read_pickle(BASE + r'\data\cache\sw_industry.pkl')
sector_map = {}
for _, row in sw.iterrows():
    sector_map[row['ts_code']] = row['industry']

MFG_17 = {
    '半导体', '纺织', '纺织机械', '专用机械', '轻工机械',
    '铝', '电器仪表', '矿物制品', '特种钢', '铜', '钢加工',
    '机床制造', '机械基件', '工程机械', '电气设备', '汽车配件', '通信设备',
}

MAX_HOLD = 90

# Search grid
DD_OPTIONS = [
    (-26, -22), (-24, -20), (-22, -18), (-20, -16), (-18, -14),
]
R120_OPTIONS = [
    (15, 55), (20, 50), (20, 45), (25, 50), (25, 45), (30, 50),
]

# First, collect ALL signals with year tags
all_signals = []

for code in prices_dict:
    if code.startswith('688') or code[:3] in ('300', '301'):
        continue
    sec = sector_map.get(code, '')
    if sec not in MFG_17:
        continue
    df = prices_dict[code].copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    df = df.dropna(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
    if len(df) < 390:
        continue
    c = df['close'].values.astype(float)
    n = len(c)
    # Compute DD for all days (expensive but needed for grid search)
    for e in range(249, n - MAX_HOLD):
        entry_year = int(str(df['trade_date'].iloc[e])[:4])
        if entry_year < 2008:
            continue
        lb = min(120, e)
        pi = e - lb + np.argmax(c[e-lb:e+1])
        dd = (c[e] / c[pi] - 1) * 100
        decline_speed = abs(dd) / max(e - pi, 1)
        r120 = (c[e] / c[e-120] - 1) * 100
        ma60 = np.mean(c[e-59:e+1])
        
        # MA60 exit outcome
        exit_day = MAX_HOLD
        for fwd in range(1, MAX_HOLD + 1):
            if e+fwd >= n:
                exit_day = n - 1 - e
                break
            if c[e+fwd] > np.mean(c[e+fwd-59:e+fwd+1]):
                exit_day = fwd
                break
        if e + exit_day >= n:
            continue
        ret = (c[e+exit_day] / c[e] - 1) * 100
        
        all_signals.append({
            'year': entry_year, 'dd': dd, 'r120': r120,
            'ma60': ma60, 'close': c[e], 'speed': decline_speed,
            'ret': ret, 'days': exit_day, 'recovered': exit_day < MAX_HOLD,
            'sec': sec
        })

df_all = pd.DataFrame(all_signals)
print(f"Total signals collected: {len(df_all)}")
print(f"Year range: {df_all['year'].min()}-{df_all['year'].max()}")

# Walk-forward: for each test year, find best params on prior years
results_fixed = []   # fixed [-22,-18] [20,50]
results_wfo = []     # yearly optimized

for test_year in range(2009, 2026):
    train = df_all[df_all['year'] < test_year]
    test = df_all[df_all['year'] == test_year]
    
    if len(train) < 100 or len(test) < 10:
        continue
    
    # --- Fixed params ---
    fixed_signals = test[
        (test['dd'] >= -22) & (test['dd'] <= -18) &
        (test['r120'] >= 20) & (test['r120'] <= 50) &
        (test['close'] < test['ma60']) &
        (test['speed'] >= 0.5)
    ]
    fixed_med = fixed_signals['ret'].median() if len(fixed_signals) > 0 else None
    fixed_n = len(fixed_signals)
    fixed_wr = (fixed_signals['ret'] > 0).sum() / len(fixed_signals) * 100 if len(fixed_signals) > 0 else 0
    
    # --- WFO: find best (dd_lo,dd_hi, r120_lo,r120_hi) on train ---
    best_med = -999
    best_params = None
    for dd_lo, dd_hi in DD_OPTIONS:
        for r120_lo, r120_hi in R120_OPTIONS:
            train_filtered = train[
                (train['dd'] >= dd_lo) & (train['dd'] <= dd_hi) &
                (train['r120'] >= r120_lo) & (train['r120'] <= r120_hi) &
                (train['close'] < train['ma60']) &
                (train['speed'] >= 0.5)
            ]
            if len(train_filtered) < 30:
                continue
            med = train_filtered['ret'].median()
            if med > best_med:
                best_med = med
                best_params = (dd_lo, dd_hi, r120_lo, r120_hi)
    
    # Apply best params to test year
    if best_params:
        dd_lo, dd_hi, r120_lo, r120_hi = best_params
        wfo_signals = test[
            (test['dd'] >= dd_lo) & (test['dd'] <= dd_hi) &
            (test['r120'] >= r120_lo) & (test['r120'] <= r120_hi) &
            (test['close'] < test['ma60']) &
            (test['speed'] >= 0.5)
        ]
        wfo_med = wfo_signals['ret'].median() if len(wfo_signals) > 0 else None
        wfo_n = len(wfo_signals)
        wfo_wr = (wfo_signals['ret'] > 0).sum() / len(wfo_signals) * 100 if len(wfo_signals) > 0 else 0
    else:
        wfo_med = None
        wfo_n = 0
        wfo_wr = 0
        best_params = (None, None, None, None)
    
    results_fixed.append({'year': test_year, 'med': fixed_med, 'n': fixed_n, 'wr': fixed_wr})
    results_wfo.append({'year': test_year, 'med': wfo_med, 'n': wfo_n, 'wr': wfo_wr, 'params': best_params})

# Summary
print(f"\n=== Walk-Forward Optimization ===")
print(f"{'Year':>5s} {'FIXED_n':>7s} {'FIXED_med':>9s} {'FIXED_WR':>8s} | {'WFO_n':>6s} {'WFO_med':>8s} {'WFO_WR':>7s} | WFO_Params")
print("-" * 90)

fixed_meds = []
wfo_meds = []
fixed_wrs = []
wfo_wrs = []
fixed_total = []
wfo_total = []

for fr, wr in zip(results_fixed, results_wfo):
    fm = f"{fr['med']:>+6.2f}%" if fr['med'] is not None else "   N/A"
    wm = f"{wr['med']:>+6.2f}%" if wr['med'] is not None else "   N/A"
    fw = f"{fr['wr']:.0f}%" if fr['med'] is not None else "N/A"
    ww = f"{wr['wr']:.0f}%" if wr['med'] is not None else "N/A"
    params = f"DD{w_vals[0]}~{w_vals[1]} r{w_vals[2]}~{w_vals[3]}" if (w_vals := wr['params']) and w_vals[0] else "N/A"
    
    print(f" {fr['year']:>4d}  {fr['n']:>6d}  {fm:>9s}  {fw:>7s}  | {wr['n']:>5d}  {wm:>9s}  {ww:>7s}  | {params}")
    
    if fr['med'] is not None:
        fixed_meds.append(fr['med']); fixed_wrs.append(fr['wr'])
        fixed_total.extend([fr['med']] * fr['n'])
    if wr['med'] is not None:
        wfo_meds.append(wr['med']); wfo_wrs.append(wr['wr'])
        wfo_total.extend([wr['med']] * wr['n'])

print(f"\n=== Summary ===")
if fixed_meds:
    print(f"FIXED [-22,-18] [20,50]:  avg_med={np.mean(fixed_meds):+.2f}%  avg_WR={np.mean(fixed_wrs):.0f}%  total_med={np.median(fixed_total):+.2f}%")
if wfo_meds:
    print(f"WFO (yearly optimized):     avg_med={np.mean(wfo_meds):+.2f}%  avg_WR={np.mean(wfo_wrs):.0f}%  total_med={np.median(wfo_total):+.2f}%")
    
    # Parameter stability
    all_params = [wr['params'] for wr in results_wfo if wr['params'][0] is not None]
    dd_los = set(p[0] for p in all_params)
    dd_his = set(p[1] for p in all_params)
    r120_los = set(p[2] for p in all_params)
    r120_his = set(p[3] for p in all_params)
    print(f"\nParameter stability across years:")
    print(f"  DD_lo range: {sorted(dd_los)}")
    print(f"  DD_hi range: {sorted(dd_his)}")
    print(f"  r120_lo range: {sorted(r120_los)}")
    print(f"  r120_hi range: {sorted(r120_his)}")
