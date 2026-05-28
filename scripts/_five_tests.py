"""五大问题测试: 去半导体 + 价格止损 + 大盘闸门"""
import pickle, pandas as pd, numpy as np, time

BASE = r'D:\AIHedgeFund\ai-hedge-fund-main'

with open(BASE + r'\data\cache\prices_full.pkl', 'rb') as f:
    prices_dict = pickle.load(f)

sw = pd.read_pickle(BASE + r'\data\cache\sw_industry.pkl')
sector_map = dict(zip(sw['ts_code'], sw['industry']))

MFG_17 = {'半导体', '纺织', '纺织机械', '专用机械', '轻工机械', '铝', '电器仪表', '矿物制品', '特种钢', '铜', '钢加工', '机床制造', '机械基件', '工程机械', '电气设备', '汽车配件', '通信设备'}

# Load CSI300 for gate
csi = pd.read_pickle(BASE + r'\data\cache\csi300.pkl')
csi['trade_date'] = pd.to_datetime(csi['trade_date'], errors='coerce')
csi = csi.sort_values('trade_date').reset_index(drop=True)
csi_close = csi['close'].values.astype(float)
csi_ma60 = np.full(len(csi), np.nan)
for i in range(120, len(csi)):
    csi_ma60[i] = np.mean(csi_close[i-59:i+1])
csi_date_map = {str(d.date()): i for i, d in enumerate(csi['trade_date'])}

MAX_HOLD = 90
DD_LO, DD_HI = -28, -24
R120_LO, R120_HI = 20, 45

print("Collecting all signals...", flush=True)
records = []
t0 = time.time()
mfg_codes = [c for c in prices_dict if not c.startswith('688') and c[:3] not in ('300','301') and sector_map.get(c,'') in MFG_17]

for i, code in enumerate(mfg_codes):
    if i % 100 == 0:
        print(f"  {i}/{len(mfg_codes)}", flush=True)
    df = prices_dict[code].copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    df = df.dropna(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
    if len(df) < 390: continue
    sec = sector_map.get(code, '')
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
        if dd > DD_HI or dd < DD_LO: continue
        if abs(dd)/max(e-pi,1) < 0.5: continue
        r120 = (c[e]/c[e-120]-1)*100
        if r120 < R120_LO or r120 > R120_HI: continue
        
        # Check CSI300 gate
        entry_date_str = str(df['trade_date'].iloc[e].date())
        csi_idx = csi_date_map.get(entry_date_str, -1)
        csi_above_ma60 = True
        if csi_idx >= 120:
            csi_above_ma60 = csi_close[csi_idx] > csi_ma60[csi_idx]
        
        # Forward path (for stop-loss test, need daily path)
        path = []
        exit_day = MAX_HOLD
        stopped = False
        for fwd in range(1, MAX_HOLD + 1):
            if e+fwd >= n: exit_day = n-1-e; break
            ret_fwd = (c[e+fwd]/c[e]-1)*100
            path.append(ret_fwd)
            if ret_fwd <= 0.15:  # -15% stop
                exit_day = fwd
                stopped = True
                break
            if c[e+fwd] > ma60_arr[e+fwd]:
                exit_day = fwd
                break
        if e+exit_day >= n: continue
        
        records.append({
            'sec': sec, 'ret': (c[e+exit_day]/c[e]-1)*100,
            'days': exit_day, 'stopped': stopped, 'recovered': exit_day < MAX_HOLD and not stopped,
            'csi_above_ma60': csi_above_ma60,
            'path_max_loss': min(path) if path else 0,
        })

df_all = pd.DataFrame(records)
print(f"Total: {len(df_all)} signals in {time.time()-t0:.0f}s\n", flush=True)

# === Test 1: 去掉半导体 ===
print("=" * 50)
print("Test 1: 去掉半导体")
print("-" * 50)
all_semi = df_all[df_all['sec'] == '半导体']
no_semi = df_all[df_all['sec'] != '半导体']
for label, sub in [("全部17赛道", df_all), ("去掉半导体(16赛道)", no_semi), ("仅半导体", all_semi)]:
    wr = (sub['ret']>0).sum()/len(sub)*100
    print(f"  {label}: n={len(sub)}  med={sub['ret'].median():+.2f}%  mean={sub['ret'].mean():+.2f}%  WR={wr:.0f}%  days={sub['days'].median():.0f}d")

# === Test 2: 价格止损 ===
print(f"\n{'='*50}")
print("Test 2: 价格止损 (-15%)")
print("-" * 50)
stopped = df_all[df_all['stopped']]
natural = df_all[~df_all['stopped']]
print(f"  触发止损: n={len(stopped)} ({len(stopped)/len(df_all)*100:.1f}%)  med={stopped['ret'].median():+.2f}%")
print(f"  自然出场: n={len(natural)} ({len(natural)/len(df_all)*100:.1f}%)  med={natural['ret'].median():+.2f}%")
print(f"  整体: med={df_all['ret'].median():+.2f}%  WR={(df_all['ret']>0).sum()/len(df_all)*100:.0f}%")

# What if no stop?
# Same signals, but without -15% trigger → need to re-run
print(f"\n  (止损替代: 让所有票走完MA60/90天)")
# For comparison: the baseline without stop is same dataset but without stop filter
# Actually the 'stopped' column already captured the full path; non-stopped signals are the same
# The question is: what % of stopped signals would have recovered if not stopped?
stopped_would_recover = 0
# We can't easily reconstruct this from the current data
# But we can estimate: of the stopped signals, the ones where MA60 was eventually crossed
print(f"  止损信号中路径最大亏损中位: {df_all[df_all['stopped']]['path_max_loss'].median():+.2f}%")
print(f"  未止损信号中路径最大亏损中位: {df_all[~df_all['stopped']]['path_max_loss'].median():+.2f}%")

# === Test 3: CSI300 MA60闸门 ===
print(f"\n{'='*50}")
print("Test 3: CSI300 MA60闸门")
print("-" * 50)
csi_up = df_all[df_all['csi_above_ma60']]
csi_down = df_all[~df_all['csi_above_ma60']]
for label, sub in [("大盘MA60上方", csi_up), ("大盘MA60下方", csi_down), ("全部(无闸门)", df_all)]:
    wr = (sub['ret']>0).sum()/len(sub)*100
    print(f"  {label}: n={len(sub)}  med={sub['ret'].median():+.2f}%  WR={wr:.0f}%")
    if len(sub) > 0:
        py = len(sub)/17
        print(f"         信号/年={py:.0f}  {'够' if py>10 else '不够'}")
