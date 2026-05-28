"""牛回头 MA60出场诚实版 - 加90天/120天上限"""
import pickle, pandas as pd, numpy as np

BASE = r'D:\AIHedgeFund\ai-hedge-fund-main'

with open(BASE + r'\data\cache\prices_full.pkl', 'rb') as f:
    prices_dict = pickle.load(f)

sw = pd.read_pickle(BASE + r'\data\cache\sw_industry.pkl')
sector_map = {}
for _, row in sw.iterrows():
    sector_map[row['ts_code']] = row['industry']

TOP5 = {'半导体', '纺织', '特种钢', '矿物制品', '铝'}

# Test: MA60 exit with 90-day cap
# Also compare: fixed 45-day (C1 shallow DD)

for pool_name, pool_sectors, dd_lo, dd_hi, r120_lo, r120_hi in [
    ("TOP5_深坑", TOP5, -25, -20, 20, 50),
    ("TOP5_浅坑", TOP5, -22, -18, 20, 50),
    ("半导体+通信", {'半导体','通信设备'}, -25, -20, 20, 50),
]:
    for max_hold in [90, 120, 180]:
        records = []
        for code in prices_dict:
            if code.startswith('688') or code[:3] in ('300', '301'):
                continue
            sec = sector_map.get(code, '')
            if sec not in pool_sectors:
                continue
            df = prices_dict[code].copy()
            df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
            df = df.dropna(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
            if len(df) < 390:
                continue
            c = df['close'].values.astype(float)
            n = len(c)
            for e in range(249, n - max_hold):
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
                if r120 < r120_lo or r120 > r120_hi:
                    continue
                
                # Find MA60 recovery or hit cap
                exit_day = max_hold  # default: force sell at cap
                for fwd in range(1, max_hold + 1):
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
                recovered = exit_day < max_hold
                records.append({'ret': ret, 'days': exit_day, 'recovered': recovered})
        
        if not records:
            continue
        df_r = pd.DataFrame(records)
        wr = (df_r['ret'] > 0).sum() / len(df_r) * 100
        rec_rate = df_r['recovered'].sum() / len(df_r) * 100
        annual_med = ((1 + df_r['ret'].median()/100) ** (252/df_r['days'].median()) - 1) * 100
        
        # What happens to non-recovered?
        non_rec = df_r[~df_r['recovered']]
        rec = df_r[df_r['recovered']]
        
        print(f"  {pool_name} MA60+{max_hold}d上限: n={len(df_r):>5d}  ret_mean={df_r['ret'].mean():>+6.2f}%  "
              f"ret_med={df_r['ret'].median():>+6.2f}%  WR={wr:.0f}%  days_med={df_r['days'].median():.0f}d  "
              f"恢复率={rec_rate:.0f}%  年化(中)={annual_med:+.0f}%")
        if len(non_rec) > 0 and len(rec) > 0:
            print(f"        恢复组: n={len(rec):>4d}  ret_med={rec['ret'].median():>+6.2f}%  days_med={rec['days'].median():.0f}d")
            print(f"        未恢复: n={len(non_rec):>4d}  ret_med={non_rec['ret'].median():>+6.2f}%  days_med={non_rec['days'].median():.0f}d")
    print()
