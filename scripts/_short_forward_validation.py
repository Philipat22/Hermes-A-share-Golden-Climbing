"""短周期牛回头 前向验证: train(2008-2018) vs test(2019-2026)"""
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

# 浅坑 + MA60 90天上限
DD_LO, DD_HI = -22, -18
MAX_HOLD = 90
R120_LO, R120_HI = 20, 50

results_train = []
results_test = []

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
    
    for e in range(249, n - MAX_HOLD):
        # Determine train vs test
        entry_year = int(str(df['trade_date'].iloc[e])[:4])
        if entry_year < 2008:  # skip pre-2008
            continue
        
        ma60_e = np.mean(c[e-59:e+1])
        if c[e] > ma60_e:
            continue
        lb = min(120, e)
        pi = e - lb + np.argmax(c[e-lb:e+1])
        dd = (c[e] / c[pi] - 1) * 100
        if dd > DD_HI or dd < DD_LO:
            continue
        decline_speed = abs(dd) / max(e - pi, 1)
        if decline_speed < 0.5:
            continue
        r120 = (c[e] / c[e-120] - 1) * 100
        if r120 < R120_LO or r120 > R120_HI:
            continue
        
        exit_day = MAX_HOLD
        for fwd in range(1, MAX_HOLD + 1):
            if e+fwd >= n:
                exit_day = n - 1 - e
                break
            ma60_fwd = np.mean(c[e+fwd-59:e+fwd+1])
            if c[e+fwd] > ma60_fwd:
                exit_day = fwd
                break
        if e + exit_day >= n:
            continue
        ret = (c[e+exit_day] / c[e] - 1) * 100
        
        record = {'ret': ret, 'days': exit_day, 'recovered': exit_day < MAX_HOLD,
                   'year': entry_year, 'sec': sec}
        
        if entry_year <= 2018:
            results_train.append(record)
        else:
            results_test.append(record)

# Train stats
df_tr = pd.DataFrame(results_train)
df_te = pd.DataFrame(results_test)

for name, df_r in [("TRAIN 2008-2018", df_tr), ("TEST 2019-2026", df_te)]:
    if len(df_r) == 0:
        continue
    wr = (df_r['ret'] > 0).sum() / len(df_r) * 100
    rec_rate = df_r['recovered'].sum() / len(df_r) * 100
    annual_med = ((1 + df_r['ret'].median()/100) ** (252/df_r['days'].median()) - 1) * 100
    
    print(f"\n=== {name} ===")
    print(f"n={len(df_r)}  均值={df_r['ret'].mean():+.2f}%  中位={df_r['ret'].median():+.2f}%  "
          f"WR={wr:.0f}%  持有中位={df_r['days'].median():.0f}d  恢复率={rec_rate:.0f}%  年化(中)={annual_med:+.0f}%")
    
    # 分年
    print(f"  {'Year':>4s} {'n':>5s} {'中位':>7s} {'WR':>5s} {'Rec%':>5s}")
    for yr in sorted(df_r['year'].unique()):
        yd = df_r[df_r['year'] == yr]
        w = (yd['ret'] > 0).sum() / len(yd) * 100
        r = yd['recovered'].sum() / len(yd) * 100
        print(f"  {yr:>4d} {len(yd):>5d} {yd['ret'].median():>+6.2f}% {w:>4.0f}% {r:>4.0f}%")
    
    # 分行业
    print(f"\n  {'行业':12s} {'n':>5s} {'中位':>7s} {'WR':>5s}")
    for sec in sorted(df_r['sec'].unique()):
        sd = df_r[df_r['sec'] == sec]
        if len(sd) < 5:
            continue
        w = (sd['ret'] > 0).sum() / len(sd) * 100
        sym = "X" if sd['ret'].median() < 0 else "OK"
        print(f"  {sec:<12s} {len(sd):>5d} {sd['ret'].median():>+6.2f}% {w:>4.0f}%  {sym}")

# Summary comparison
print(f"\n=== TRAIN vs TEST ===")
metrics = [('n', 'count'), ('ret', 'median'), ('ret', 'mean'), ('ret', lambda x: (x>0).sum()/len(x)*100)]
for col, func in [('ret', 'median'), ('ret', 'mean'), ('ret', lambda x: (x>0).sum()/len(x)*100)]:
    tr_val = df_tr['ret'].median() if func == 'median' else (df_tr['ret'].mean() if func == 'mean' else (df_tr['ret']>0).sum()/len(df_tr)*100)
    te_val = df_te['ret'].median() if func == 'median' else (df_te['ret'].mean() if func == 'mean' else (df_te['ret']>0).sum()/len(df_te)*100)
    label = '中位' if func == 'median' else ('均值' if func == 'mean' else '胜率')
    print(f"  {label}: train {tr_val:+.2f} vs test {te_val:+.2f} = {'稳定' if abs(tr_val-te_val) < 2 else '⚠ 偏差' if abs(tr_val-te_val) < 5 else '✗ 崩了'}")
