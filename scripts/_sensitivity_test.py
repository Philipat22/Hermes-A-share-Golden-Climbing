"""短周期 参数敏感度测试 — 扰动DD和r120"""
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

# Test DD perturbations and r120 perturbations
dd_configs = [
    ("DD[-20,-16]", -20, -16),
    ("DD[-21,-17]", -21, -17),
    ("DD[-22,-18]", -22, -18),  # current
    ("DD[-23,-19]", -23, -19),
    ("DD[-24,-20]", -24, -20),
    ("DD[-25,-21]", -25, -21),
]

r120_configs = [
    ("r120[15,55]", 15, 55),
    ("r120[20,50]", 20, 50),   # current
    ("r120[20,45]", 20, 45),
    ("r120[25,50]", 25, 50),
    ("r120[25,45]", 25, 45),
    ("r120[30,50]", 30, 50),
]

print("=== DD Sensitivity (r120 fixed 20-50) ===")
print(f"{'Config':16s} {'n':>6s} {'中位':>7s} {'WR':>5s} {'持有':>5s} {'恢复率':>5s} {'年化(中)':>8s} {'OK?':>5s}")
print("-" * 65)

for cfg_name, dd_lo, dd_hi in dd_configs:
    records = []
    for code in list(prices_dict.keys())[:3000]:  # sample for speed
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
            ma60_e = np.mean(c[e-59:e+1])
            if c[e] > ma60_e:
                continue
            lb = min(120, e)
            pi = e - lb + np.argmax(c[e-lb:e+1])
            dd = (c[e] / c[pi] - 1) * 100
            if dd > dd_hi or dd < dd_lo:
                continue
            decline_speed = abs(dd) / max(e - pi, 1)
            if decline_speed < 0.5:
                continue
            r120 = (c[e] / c[e-120] - 1) * 100
            if r120 <= 20 or r120 > 50:
                continue
            exit_day = MAX_HOLD
            for fwd in range(1, MAX_HOLD + 1):
                if e+fwd >= n: break
                if c[e+fwd] > np.mean(c[e+fwd-59:e+fwd+1]):
                    exit_day = fwd; break
            if e + exit_day >= n: continue
            ret = (c[e+exit_day] / c[e] - 1) * 100
            records.append({'ret': ret, 'days': exit_day})
    
    if records:
        df_r = pd.DataFrame(records)
        wr = (df_r['ret'] > 0).sum() / len(df_r) * 100
        annual_med = ((1 + df_r['ret'].median()/100) ** (252/df_r['days'].median()) - 1) * 100
        status = "OK" if df_r['ret'].median() > 3 and wr > 65 else ("WARN" if df_r['ret'].median() > 0 else "DEAD")
        print(f"{cfg_name:<16s} {len(df_r):>6d} {df_r['ret'].median():>+6.2f}% {wr:>4.0f}% {df_r['days'].median():>5.0f}d "
              f"{(df_r['days']<MAX_HOLD).sum()/len(df_r)*100:>5.0f}% {annual_med:>+7.0f}% {status:>5s}")

print(f"\n=== r120 Sensitivity (DD fixed -22~-18) ===")
print(f"{'Config':16s} {'n':>6s} {'中位':>7s} {'WR':>5s} {'持有':>5s} {'恢复率':>5s} {'年化(中)':>8s} {'OK?':>5s}")
print("-" * 65)

for cfg_name, r120_lo, r120_hi in r120_configs:
    records = []
    for code in list(prices_dict.keys())[:3000]:
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
            ma60_e = np.mean(c[e-59:e+1])
            if c[e] > ma60_e:
                continue
            lb = min(120, e)
            pi = e - lb + np.argmax(c[e-lb:e+1])
            dd = (c[e] / c[pi] - 1) * 100
            if dd > -18 or dd < -22:
                continue
            decline_speed = abs(dd) / max(e - pi, 1)
            if decline_speed < 0.5:
                continue
            r120 = (c[e] / c[e-120] - 1) * 100
            if r120 < r120_lo or r120 > r120_hi:
                continue
            exit_day = MAX_HOLD
            for fwd in range(1, MAX_HOLD + 1):
                if e+fwd >= n: break
                if c[e+fwd] > np.mean(c[e+fwd-59:e+fwd+1]):
                    exit_day = fwd; break
            if e + exit_day >= n: continue
            ret = (c[e+exit_day] / c[e] - 1) * 100
            records.append({'ret': ret, 'days': exit_day})
    
    if records:
        df_r = pd.DataFrame(records)
        wr = (df_r['ret'] > 0).sum() / len(df_r) * 100
        annual_med = ((1 + df_r['ret'].median()/100) ** (252/df_r['days'].median()) - 1) * 100
        status = "OK" if df_r['ret'].median() > 3 and wr > 65 else ("WARN" if df_r['ret'].median() > 0 else "DEAD")
        print(f"{cfg_name:<16s} {len(df_r):>6d} {df_r['ret'].median():>+6.2f}% {wr:>4.0f}% {df_r['days'].median():>5.0f}d "
              f"{(df_r['days']<MAX_HOLD).sum()/len(df_r)*100:>5.0f}% {annual_med:>+7.0f}% {status:>5s}")
