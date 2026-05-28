"""短周期扩展测试: 全制造17赛道 + 分年表现"""
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

TOP5 = {'半导体', '纺织', '特种钢', '矿物制品', '铝'}

# 浅坑版 + MA60 90天上限
DD_LO, DD_HI = -22, -18
MAX_HOLD = 90

for pool_name, pool_sectors in [("全制造17", MFG_17), ("TOP5", TOP5)]:
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
        for e in range(249, n - MAX_HOLD):
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
            if r120 <= 20 or r120 > 50:
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
            records.append({
                'sec': sec, 'ret': ret, 'days': exit_day,
                'year': int(str(df['trade_date'].iloc[e])[:4]),
                'recovered': exit_day < MAX_HOLD
            })
    
    df_r = pd.DataFrame(records)
    wr = (df_r['ret'] > 0).sum() / len(df_r) * 100
    rec_rate = df_r['recovered'].sum() / len(df_r) * 100
    annual_med = ((1 + df_r['ret'].median()/100) ** (252/df_r['days'].median()) - 1) * 100
    
    print(f"\n=== {pool_name} 浅坑MA60+90d ===")
    print(f"n={len(df_r)}  均值={df_r['ret'].mean():+.2f}%  中位={df_r['ret'].median():+.2f}%  WR={wr:.0f}%  "
          f"持有中位={df_r['days'].median():.0f}d  恢复率={rec_rate:.0f}%  年化(中)={annual_med:+.0f}%")
    print(f"信号/年={len(df_r)/17:.0f}")
    
    # 分年
    print(f"\n  {'Year':>4s} {'n':>5s} {'Mean':>7s} {'Med':>7s} {'WR':>5s} {'Rec%':>5s}")
    for yr in sorted(df_r['year'].unique()):
        yd = df_r[df_r['year'] == yr]
        w = (yd['ret'] > 0).sum() / len(yd) * 100
        r = yd['recovered'].sum() / len(yd) * 100
        print(f"  {yr:>4d} {len(yd):>5d} {yd['ret'].mean():>+6.2f}% {yd['ret'].median():>+6.2f}% {w:>4.0f}% {r:>4.0f}%")
    
    # 分行业
    print(f"\n  {'行业':12s} {'n':>5s} {'中位':>7s} {'WR':>5s} {'Rec%':>5s} {'Med天':>5s}")
    for sec in sorted(df_r['sec'].unique()):
        sd = df_r[df_r['sec'] == sec]
        if len(sd) < 10:
            continue
        w = (sd['ret'] > 0).sum() / len(sd) * 100
        r = sd['recovered'].sum() / len(sd) * 100
        print(f"  {sec:<12s} {len(sd):>5d} {sd['ret'].median():>+6.2f}% {w:>4.0f}% {r:>4.0f}% {sd['days'].median():>5.0f}d")
