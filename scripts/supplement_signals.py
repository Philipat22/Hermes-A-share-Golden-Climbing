"""
补充信号缓存: 扫描 2026-03-05 ~ 最新 的新信号
用法: 
  先设置: $env:NPY_PICKLE_COMPAT="1"    (PowerShell)
  或:     set NPY_PICKLE_COMPAT=1       (CMD)
  然后:   python supplement_signals.py
输出: 更新 golden_pit_signals_all.pkl
"""
import os
os.environ.setdefault('NPY_PICKLE_COMPAT', '1')
import pickle, pandas as pd, numpy as np, warnings, time
from datetime import datetime
warnings.filterwarnings('ignore')

# ====== 配置 ======
PRICES_FILE = r"D:\AIHedgeFund\ai-hedge-fund-main\data\cache\prices_full.pkl"
SIG_CACHE = r"D:\AIHedgeFund\ai-hedge-fund-main\data\cache\golden_pit_signals_all.pkl"
TOKEN = "6aa54b486cea6bdaedcf9fe758e16d037364dad9bcc4bd3b8d69a3d7"
START_DATE = "20260305"
EXCLUDE_SECTORS = {'银行', '港口', '证券', '钢加工', '旅游景点'}

print("[1/4] 加载数据...")
with open(PRICES_FILE, 'rb') as f:
    prices = pickle.load(f)
with open(SIG_CACHE, 'rb') as f:
    sig = pickle.load(f)

import tushare as ts
pro = ts.pro_api(TOKEN)
df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
name_map = dict(zip(df_basic['ts_code'], df_basic['name']))
sector_map = dict(zip(df_basic['ts_code'], df_basic['industry']))

# 已有信号日期
existing_dates = set(sig['date'].astype(str).str[:8])
print(f"现有信号: {len(sig)}笔, {len(existing_dates)}个日期")

# 找新交易日
print("[2/4] 获取交易日历...")
cal = pro.trade_cal(exchange='SSE', start_date=START_DATE, end_date=datetime.now().strftime('%Y%m%d'))
trade_days = cal[cal['is_open']==1]['cal_date'].tolist()
new_dates = [d for d in trade_days if d not in existing_dates]
print(f"新交易日: {len(new_dates)}天 ({new_dates[0]}~{new_dates[-1]})")

# 获取prices中所有股票代码和最新日期
all_codes = list(prices.keys())
# 检查prices是否覆盖新日期
sample_dates = []
for code in all_codes[:200]:
    df = prices[code]
    if hasattr(df,'columns') and len(df)>0:
        d = str(df['trade_date'].max())[:8]
        if d: sample_dates.append(d)
from collections import Counter
latest_date = Counter(sample_dates).most_common(1)[0][0]
print(f"Prices最新日期: {latest_date}")
if latest_date < max(new_dates):
    print(f"⚠️ Prices数据落后于日历 — 先运行 update_prices_today.py")

# 只扫prices有的日期
scannable = [d for d in new_dates if d <= latest_date]
print(f"可扫描: {len(scannable)}天")

# ====== 扫描 ======
print(f"[3/4] 扫描信号 ({len(all_codes)}只股票)...")
new_signals = []
processed = 0

for code in all_codes:
    df_raw = prices.get(code)
    if df_raw is None or not hasattr(df_raw, 'columns'): continue
    if '.BJ' in code or code.startswith('688'): continue
    if 'ST' in name_map.get(code, ''): continue
    if sector_map.get(code,'其他') in EXCLUDE_SECTORS: continue
    
    df = df_raw.sort_values('trade_date').reset_index(drop=True)
    if len(df) < 120: continue
    df['date_str'] = df['trade_date'].astype(str)
    
    c = df['close'].values
    v = df['vol'].values
    n = len(df)
    
    # 预计算均线(只用一次)
    ma20 = pd.Series(c).rolling(20).mean().values
    ma60 = pd.Series(c).rolling(60).mean().values
    ma120 = pd.Series(c).rolling(120).mean().values
    vm = pd.Series(v).rolling(20).mean().values
    
    for target_date in scannable:
        match = df[df['date_str'].str.startswith(target_date)]
        if len(match) == 0: continue
        idx = match.index[0]
        if idx < 120: continue
        if np.isnan(ma60[idx]): continue
        
        score = sum([ma20[idx]>ma60[idx], ma60[idx]>ma120[idx], 
                     idx>=5 and ma60[idx]>ma60[idx-5], c[idx]>ma20[idx], c[idx]>ma60[idx]])
        if score < 4: continue
        
        lb = min(20, idx)
        pk = idx - lb + np.argmax(c[idx-lb:idx+1])
        dd = (c[idx]/c[pk]-1)*100
        if dd > -10: continue
        
        spd = abs(dd)/max(idx-pk, 1)
        if spd <= 0.5: continue
        
        vr = v[idx]/vm[idx] if vm[idx]>0 else 99
        if vr >= 3: continue
        
        ret120 = (c[idx]/c[max(0,idx-120)]-1)*100
        if ret120 <= abs(dd)*2: continue
        
        new_signals.append({
            'code': code,
            'name': name_map.get(code,'?'),
            'sector': sector_map.get(code,'其他'),
            'date': int(target_date),
            'trend': score,
            'speed': round(spd, 2),
            'dd': round(dd, 2),
            'r120': round(ret120, 2),
            'r30': np.nan, 'r40': np.nan, 'r50': np.nan, 'r60': np.nan
        })
    
    processed += 1
    if processed % 500 == 0:
        print(f"  进度: {processed}/{len(all_codes)}, 信号: {len(new_signals)}")

print(f"  完成! 新信号: {len(new_signals)}笔")

# ====== 保存 ======
print(f"[4/4] 保存...")
if new_signals:
    new_df = pd.DataFrame(new_signals)
    sig_updated = pd.concat([sig, new_df], ignore_index=True)
    with open(SIG_CACHE, 'wb') as f:
        pickle.dump(sig_updated, f)
    print(f"信号库: {len(sig)} → {len(sig_updated)}笔")
    
    # 统计
    new_df['yyyymm'] = new_df['date'].astype(str).str[:6]
    for m in sorted(new_df['yyyymm'].unique()):
        m_data = new_df[new_df['yyyymm']==m]
        sweet = len(m_data[(m_data['dd']>-18)&(m_data['dd']<=-14)])
        print(f"  {m}: {len(m_data)}个信号 (甜点{sweet}个)")
else:
    print("无新信号 — prices数据可能需先更新")
    with open(SIG_CACHE, 'wb') as f:
        pickle.dump(sig, f)

print("\n✅ 完成")
print("注意: r30/r40/r50/r60暂时为NaN — 需要等40个交易日逐步填充")
