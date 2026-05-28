"""牛回头v2.0 每日扫描脚本 — 替代_scan_today.py"""
import pickle, pandas as pd, numpy as np
from datetime import datetime

BASE = r'D:\AIHedgeFund\ai-hedge-fund-main'

# Load
with open(BASE + r'\data\cache\prices_full.pkl', 'rb') as f:
    prices_dict = pickle.load(f)

sw = pd.read_pickle(BASE + r'\data\cache\sw_industry.pkl')
sector_map = dict(zip(sw['ts_code'], sw['industry']))
name_map = dict(zip(sw['ts_code'], sw['name']))

MFG_17 = {
    '半导体', '纺织', '纺织机械', '专用机械', '轻工机械',
    '铝', '电器仪表', '矿物制品', '特种钢', '铜', '钢加工',
    '机床制造', '机械基件', '工程机械', '电气设备', '汽车配件', '通信设备',
}

# Pre-filter to MFG codes only (fast path)
mfg_codes = [c for c in prices_dict 
             if not c.startswith('688') and c[:3] not in ('300','301')
             and sector_map.get(c,'') in MFG_17]

DD_LO, DD_HI = -28, -24
R120_LO, R120_HI = 20, 45

# ── Scan signals ──
signals = []

for code in mfg_codes:
    sec = sector_map.get(code, '')
    
    df = prices_dict[code].copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    df = df.dropna(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
    if len(df) < 300:
        continue
    
    c = df['close'].values.astype(float)
    n = len(df)
    e = n - 1  # latest day
    
    # MA60
    if e < 59: continue
    ma60 = np.mean(c[e-59:e+1])
    if c[e] > ma60:
        continue  # must be below MA60
    
    # DD from 120d high
    lb = min(120, e)
    pi = e - lb + np.argmax(c[e-lb:e+1])
    dd = (c[e] / c[pi] - 1) * 100
    if dd > DD_HI or dd < DD_LO:
        continue
    
    # Decline speed
    days_high = e - pi
    if days_high <= 0:
        continue
    speed = abs(dd) / days_high
    if speed < 0.5:
        continue
    
    # r120
    if e < 120: continue
    r120 = (c[e] / c[e-120] - 1) * 100
    if r120 < R120_LO or r120 > R120_HI:
        continue
    
    # Check if already recovering (close > MA60 within last 5 days?)
    # Actually, if we're below MA60 and DD is in range, we're good
    
    name = name_map.get(code, code)
    high_price = c[pi]
    latest_date = str(df['trade_date'].iloc[e])[:10]
    
    signals.append({
        'code': code, 'name': name, 'sec': sec,
        'close': c[e], 'dd': dd, 'r120': r120,
        'speed': speed, 'days_high': days_high,
        'high': high_price, 'ma60': ma60,
        'date': latest_date
    })

# Sort by DD (deeper first = better entry)
signals.sort(key=lambda x: x['dd'])

# ── Print ──
print(f"\n{'='*70}")
print(f"  牛回头 v2.0 每日扫描")
print(f"  参数: DD[-28~-24%] r120[20~45%] MA60下 跌速>=0.5%")
print(f"  池子: 17制造业赛道(主板)  N=3 同板块限1只")
print(f"  出场: MA60恢复即卖, 90天上限强制砍")
print(f"  日期: {signals[0]['date'] if signals else '无数据'}")
print(f"{'='*70}")

if not signals:
    print("\n  无信号。")
else:
    print(f"\n  BUY信号: {len(signals)}只")
    print(f"  {'#':>3s} {'代码':<12s} {'名称':<10s} {'行业':<10s} {'价格':>7s} {'DD':>7s} {'r120':>7s} {'跌速':>6s} {'距高点':>6s}")
    print(f"  {'-'*68}")
    
    for i, s in enumerate(signals[:30]):
        print(f"  {i+1:>3d} {s['code']:<12s} {s['name']:<10s} {s['sec']:<10s} "
              f"{s['close']:>7.2f} {s['dd']:>+6.1f}% {s['r120']:>+6.1f}% {s['speed']:>5.2f} {s['days_high']:>6d}d")
    
    # Top 3 picks (respect sector diversity)
    picked = []
    picked_secs = set()
    for s in signals:
        if len(picked) >= 3:
            break
        if s['sec'] not in picked_secs:
            picked.append(s)
            picked_secs.add(s['sec'])
    
    print(f"\n  === TOP 3推荐 (行业分散) ===")
    for i, s in enumerate(picked):
        print(f"  #{i+1}: {s['code']} {s['name']} ({s['sec']}) "
              f"RMB{s['close']:.2f}  DD={s['dd']:+.1f}%  r120={s['r120']:+.1f}%")

print(f"\n{'='*70}")
