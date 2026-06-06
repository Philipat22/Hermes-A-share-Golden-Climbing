"""
Dalio宏观安检门 A股验证
验证: 丑去杠杆条件 / 信用收缩 / 政策转向 → A股后续表现
"""
import pandas as pd, numpy as np, akshare as ak, warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Dalio宏观安检门 A股验证")
print("=" * 60)

# ============================================================
# 1. 数据获取
# ============================================================
print("\n[1] 获取宏观数据...")

# GDP (季) - 列0=季度, 列1=绝对值, 列2=同比
gdp = ak.macro_china_gdp()
gdp = gdp.iloc[:, [0, 1, 2]].copy()
gdp.columns = ['quarter', 'gdp_value', 'gdp_yoy']
gdp['quarter'] = gdp['quarter'].str.extract(r'(\d{4})').astype(int)
gdp = gdp[gdp['quarter'] >= 2005]
print(f"  GDP: {len(gdp)} 季 ({gdp['quarter'].min()}-{gdp['quarter'].max()})")

# CPI (月) - 列1=日期(datetime.date), 列2=值
cpi = ak.macro_china_cpi_monthly()
cpi = cpi.iloc[:, [1, 2]].copy()
cpi.columns = ['date', 'cpi']
cpi['date'] = pd.to_datetime(cpi['date'])
cpi = cpi.dropna(subset=['cpi'])
cpi = cpi.sort_values('date')
print(f"  CPI: {len(cpi)} 月 ({cpi['date'].min().date()}-{cpi['date'].max().date()})")

# 社融 (月) - 列0=月份, 列1=增量
sf = ak.macro_china_shrzgm()
sf = sf.iloc[:, [0, 1]].copy()
sf.columns = ['month', 'sf_flow']
sf['date'] = pd.to_datetime(sf['month'].astype(str) + '01', format='%Y%m%d')
sf = sf.sort_values('date')
# 计算12个月滚动社融
sf['sf_12m'] = sf['sf_flow'].rolling(12).sum()
print(f"  SF: {len(sf)} 月 ({sf['date'].min().date()}-{sf['date'].max().date()})")

# 上证综指 (已有)
CACHE = r'C:\Users\Zenta\.qclaw\workspace-agent-b9c8dcea\data\cache'
idx = pd.read_pickle(f'{CACHE}/sh000001_daily.pkl')
idx['trade_date'] = pd.to_datetime(idx['trade_date'])
idx = idx.set_index('trade_date').sort_index()

# 月度指数 (取月末)
idx_monthly = idx['close'].resample('ME').last()
print(f"  SH Idx: {len(idx_monthly)} 月")

# ============================================================
# 2. 数据对齐
# ============================================================
print("\n[2] 对齐数据...")

# 先把CPI和社融合并到月度
monthly = pd.DataFrame({'close': idx_monthly})
monthly.index.name = 'date'
monthly = monthly.reset_index()

# 合并CPI (用最近的值)
cpi_sorted = cpi.set_index('date').sort_index()
monthly['cpi'] = np.nan
for i, row in monthly.iterrows():
    prev = cpi_sorted[cpi_sorted.index <= row['date']]
    if len(prev) > 0:
        monthly.at[i, 'cpi'] = prev['cpi'].iloc[-1]

# 合并社融
sf_sorted = sf.set_index('date').sort_index()
monthly['sf_12m'] = np.nan
for i, row in monthly.iterrows():
    prev = sf_sorted[sf_sorted.index <= row['date']]
    if len(prev) > 0:
        monthly.at[i, 'sf_12m'] = prev['sf_12m'].iloc[-1]

# GDP是季度的，填充到每月
monthly['quarter'] = monthly['date'].dt.year
monthly['gdp_yoy'] = np.nan
for _, g in gdp.iterrows():
    mask = monthly['date'].dt.year == g['quarter']
    monthly.loc[mask, 'gdp_yoy'] = g['gdp_yoy']

# 前向填充GDP
monthly['gdp_yoy'] = monthly['gdp_yoy'].ffill()

monthly = monthly.dropna(subset=['cpi', 'gdp_yoy', 'sf_12m'])
# 去掉CPI=0的异常值
monthly = monthly[monthly['cpi'].abs() > 0.01]
print(f"  对齐后: {len(monthly)} 月")

# ============================================================
# 3. Dalio信号计算
# ============================================================
print("\n[3] 计算Dalio信号...")

# 名义GDP = 实际GDP + CPI
monthly['nominal_gdp'] = monthly['gdp_yoy'] + monthly['cpi']

# 信号1: 丑去杠杆 - CPI<0 连续3个月
monthly['deflation'] = monthly['cpi'] < 0
monthly['deflation_3m'] = monthly['deflation'].rolling(3).sum() >= 2  # 3个月中有2个月通缩
monthly['nominal_gdp_low'] = monthly['nominal_gdp'] < 3  # 名义GDP<3%

# 信号2: 信用收缩 - 社融12月滚动 < 名义GDP
# (社融是绝对值，需要和GDP绝对值比较——这里用简化: 社融增速<名义GDP增速)
monthly['sf_12m_chg'] = monthly['sf_12m'].pct_change(12) * 100
monthly['credit_contraction'] = monthly['sf_12m_chg'] < monthly['nominal_gdp']

# 信号3: 综合风险评分
monthly['dalio_risk'] = (
    monthly['deflation_3m'].astype(int) +
    monthly['nominal_gdp_low'].astype(int) +
    monthly['credit_contraction'].astype(int)
)

print(f"  通缩3月以上: {monthly['deflation_3m'].sum()} 月")
print(f"  名义GDP<3%: {monthly['nominal_gdp_low'].sum()} 月")
print(f"  信用收缩: {monthly['credit_contraction'].sum()} 月")
print(f"  风险分>0: {(monthly['dalio_risk'] > 0).sum()} 月")

# ============================================================
# 4. 后续A股收益
# ============================================================
print("\n[4] Dalio信号 → A股后续收益...")

# 计算月收益
monthly['fwd_1m'] = monthly['close'].shift(-1) / monthly['close'] - 1
monthly['fwd_3m'] = monthly['close'].shift(-3) / monthly['close'] - 1
monthly['fwd_6m'] = monthly['close'].shift(-6) / monthly['close'] - 1
monthly['fwd_12m'] = monthly['close'].shift(-12) / monthly['close'] - 1

for horizon, label in [(1,'1月'),(3,'3月'),(6,'6月'),(12,'12月')]:
    col = f'fwd_{horizon}m'
    
    # 高vs低风险
    high = monthly[(monthly['dalio_risk'] > 0) & monthly[col].notna()]
    low = monthly[(monthly['dalio_risk'] == 0) & monthly[col].notna()]
    
    if len(high) > 0 and len(low) > 0:
        print(f"\n  后{label}:")
        print(f"    风险高(dalio_risk>0): avg={high[col].mean()*100:+.1f}% med={high[col].median()*100:+.1f}% "
              f"胜率={(high[col]>0).mean()*100:.0f}% n={len(high)}")
        print(f"    风险低(dalio_risk=0): avg={low[col].mean()*100:+.1f}% med={low[col].median()*100:+.1f}% "
              f"胜率={(low[col]>0).mean()*100:.0f}% n={len(low)}")
        diff = high[col].mean() - low[col].mean()
        print(f"    差值: {diff*100:+.1f}% {'(高风险时期收益更差)' if diff < 0 else '(意外: 高风险时期收益反而更好)'}")

# 单独测试每个条件
print("\n[5] 单条件测试 (后6月):")
for cond, label in [('deflation_3m','通缩3月+'),('nominal_gdp_low','名义GDP<3%'),('credit_contraction','信用收缩')]:
    yes = monthly[(monthly[cond]) & monthly['fwd_6m'].notna()]
    no = monthly[(~monthly[cond]) & monthly['fwd_6m'].notna()]
    if len(yes) > 3:
        print(f"  {label}: yes={yes['fwd_6m'].mean()*100:+.1f}% (n={len(yes)}) vs no={no['fwd_6m'].mean()*100:+.1f}% (n={len(no)}) | "
              f"diff={(yes['fwd_6m'].mean()-no['fwd_6m'].mean())*100:+.1f}%")

# 最严重时期
print(f"\n[6] 历史风险最高时期:")
top_risk = monthly.nlargest(10, 'dalio_risk')[['date','cpi','gdp_yoy','nominal_gdp','dalio_risk','fwd_6m']].copy()
for _, r in top_risk.iterrows():
    print(f"  {r['date'].strftime('%Y-%m')}: cpi={r['cpi']:+.1f}% gdp={r['gdp_yoy']:.1f}% "
          f"risk={int(r['dalio_risk'])}/3 fwd6m={r['fwd_6m']*100:+.1f}%")

print("\nDone.")
