"""牛回头 短周期扫描 - 寻找高频稳定配置"""
import pickle, pandas as pd, numpy as np

BASE = r'D:\AIHedgeFund\ai-hedge-fund-main'

with open(BASE + r'\data\cache\prices_full.pkl', 'rb') as f:
    prices_dict = pickle.load(f)

sw = pd.read_pickle(BASE + r'\data\cache\sw_industry.pkl')
sector_map = {}
for _, row in sw.iterrows():
    sector_map[row['ts_code']] = row['industry']

# 配置矩阵
configs = [
    # (name, sectors, r120_lo, r120_hi, dd_lo, dd_hi, days_hi_lo, days_hi_hi)
    # A组: 宽口径, 高频
    ("A1_宽20-50", 'MFG_ALL', 20, 50, -25, -20, 0, 999),
    ("A2_宽20-45", 'MFG_ALL', 20, 45, -25, -20, 0, 999),
    # B组: 收r120
    ("B1_30-45_全制造", 'MFG_ALL', 30, 45, -25, -20, 0, 999),
    # C组: 浅坑更快反弹?
    ("C1_浅坑_18-22", 'MFG_TOP5', 20, 50, -22, -18, 0, 999),
    ("C2_浅坑_15-20", 'MFG_TOP5', 20, 50, -20, -15, 0, 999),
    # D组: 距高点限制
    ("D1_10-30天", 'MFG_TOP5', 30, 45, -25, -20, 10, 30),
    ("D2_5-20天", 'MFG_TOP5', 30, 45, -25, -20, 5, 20),
]

MFG_ALL = {
    '半导体', '纺织', '纺织机械', '专用机械', '轻工机械',
    '铝', '电器仪表', '矿物制品', '特种钢', '铜', '钢加工',
    '机床制造', '机械基件', '工程机械', '电气设备', '汽车配件', '通信设备',
}

MFG_TOP5 = {'半导体', '纺织', '特种钢', '矿物制品', '铝'}

results = []

for cfg_name, sec_key, r120_lo, r120_hi, dd_lo, dd_hi, dh_lo, dh_hi in configs:
    sectors = MFG_ALL if sec_key == 'MFG_ALL' else MFG_TOP5
    
    r30_list, r45_list, r60_list, r90_list = [], [], [], []
    count = 0
    
    for code in prices_dict:
        if code.startswith('688') or code[:3] in ('300', '301'):
            continue
        sec = sector_map.get(code, '')
        if sec not in sectors:
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
            if dd > dd_hi or dd < dd_lo:
                continue
            days_high = e - pi
            if dh_lo > 0 and (days_high < dh_lo or days_high > dh_hi):
                continue
            decline_speed = abs(dd) / max(days_high, 1)
            if decline_speed < 0.5:
                continue
            r120 = (c[e] / c[e-120] - 1) * 100
            if r120 < r120_lo or r120 > r120_hi:
                continue
            
            r30_list.append((c[e+30] / c[e] - 1) * 100)
            r45_list.append((c[e+45] / c[e] - 1) * 100)
            r60_list.append((c[e+60] / c[e] - 1) * 100)
            r90_list.append((c[e+90] / c[e] - 1) * 100)
            count += 1
    
    for period_name, r_list in [("r30", r30_list), ("r45", r45_list), ("r60", r60_list), ("r90", r90_list)]:
        r = np.array(r_list)
        wr = (r > 0).sum() / len(r) * 100
        days = int(period_name[1:])
        annual_med = ((1 + np.median(r)/100) ** (252/days) - 1) * 100
        annual_mean = ((1 + np.mean(r)/100) ** (252/days) - 1) * 100
        # 夏普近似: 均值/std
        sharpe = np.mean(r) / np.std(r) if np.std(r) > 0 else 0
        results.append({
            'config': cfg_name, 'period': period_name,
            'n': len(r), 'mean': np.mean(r), 'med': np.median(r),
            'wr': wr, 'annual_med': annual_med, 'annual_mean': annual_mean,
            'sharpe': sharpe, 'signals_per_year': len(r) / 17,
        })

df_r = pd.DataFrame(results)

# Print matrix: configs x periods
periods = ['r30', 'r45', 'r60', 'r90']
print(f"{'Config':20s}", end="")
for p in periods:
    print(f" {'n':>5s} {'Avg':>6s} {'Med':>6s} {'WR':>4s} {'年化(中)':>7s}", end="")
print()

for cfg_name, _, _, _, _, _, _, _ in configs:
    cfg_data = df_r[df_r['config'] == cfg_name]
    print(f"{cfg_name:<20s}", end="")
    for p in periods:
        row = cfg_data[cfg_data['period'] == p]
        if len(row) == 0:
            print(f" {'—':>5s} {'—':>6s} {'—':>6s} {'—':>4s} {'—':>7s}", end="")
        else:
            r = row.iloc[0]
            print(f" {int(r['n']):>5d} {r['mean']:>+5.1f}% {r['med']:>+5.1f}% {r['wr']:>3.0f}% {r['annual_med']:>+6.0f}%", end="")
    print()

# Also: MA60 exit approach
print(f"\n=== MA60 Exit (v1.0 style) on different pools ===")
for pool_name, pool_sectors in [("半导体+通信", {'半导体','通信设备'}), ("TOP5", MFG_TOP5), ("全制造17", MFG_ALL)]:
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
        for e in range(249, n - 250):
            ma60 = np.mean(c[e-59:e+1])
            if c[e] > ma60:
                continue
            lb = min(120, e)
            pi = e - lb + np.argmax(c[e-lb:e+1])
            dd = (c[e] / c[pi] - 1) * 100
            if dd > -20 or dd < -25:
                continue
            decline_speed = abs(dd) / max(e - pi, 1)
            if decline_speed < 0.5:
                continue
            r120 = (c[e] / c[e-120] - 1) * 100
            if r120 <= 20 or r120 > 50:
                continue
            # Find MA60 recovery
            exit_day = None
            for fwd in range(1, min(250, n - e)):
                if c[e+fwd] > np.mean(c[e+fwd-59:e+fwd+1]):
                    exit_day = fwd
                    break
            if exit_day is None:
                continue
            ret = (c[e+exit_day] / c[e] - 1) * 100
            records.append({'ret': ret, 'days': exit_day})
    
    if records:
        df_ma = pd.DataFrame(records)
        wr = (df_ma['ret'] > 0).sum() / len(df_ma) * 100
        print(f"  {pool_name}: n={len(df_ma)}  ret_mean={df_ma['ret'].mean():+.2f}%  ret_med={df_ma['ret'].median():+.2f}%  "
              f"days_med={df_ma['days'].median():.0f}d  WR={wr:.0f}%  "
              f"annual_med={((1+df_ma['ret'].median()/100)**(252/df_ma['days'].median())-1)*100:+.0f}%")
