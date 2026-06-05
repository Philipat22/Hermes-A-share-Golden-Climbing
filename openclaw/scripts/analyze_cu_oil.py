"""铜油比 vs 沪深300"""
import akshare as ak, pandas as pd, numpy as np

# 铜
cu = ak.futures_main_sina(symbol='CU0')
cu = cu.iloc[:, [0, 1]].copy()
cu.columns = ['date', 'cu']
cu['date'] = pd.to_datetime(cu['date'])
cu['cu'] = pd.to_numeric(cu['cu'], errors='coerce')
cu = cu.dropna().sort_values('date')

# 油
oil = ak.futures_main_sina(symbol='SC0')
oil = oil.iloc[:, [0, 1]].copy()
oil.columns = ['date', 'oil']
oil['date'] = pd.to_datetime(oil['date'])
oil['oil'] = pd.to_numeric(oil['oil'], errors='coerce')
oil = oil.dropna().sort_values('date')

# 合并
df = cu.merge(oil, on='date', how='inner')
df['cu_oil'] = df['cu'] / df['oil']  # 铜油比
print(f'铜油比数据: {len(df)}天 ({df.date.min().date()} ~ {df.date.max().date()})')

# 沪深300
idx = pd.read_pickle(r'C:\Users\Zenta\.qclaw\workspace-agent-b9c8dcea\data\cache\sh000001_daily.pkl')
idx['trade_date'] = pd.to_datetime(idx['trade_date'])
idx = idx.set_index('trade_date').sort_index()

# 用沪深300... 算了，用上证综指近似 (CKSH cache里只有上证)
# 实际上应该用Tushare拿沪深300
import tushare as ts
pro = ts.pro_api('6aa54b486cea6bdaedcf9fe758e16d037364dad9bcc4bd3b8d69a3d7')
hs300 = pro.index_daily(ts_code='000300.SH', start_date='20180326', end_date='20260605')
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300 = hs300.set_index('trade_date')['close'].sort_index()

# 合并
df = df.set_index('date')
merged = df.join(hs300.rename('hs300'), how='inner')
print(f'合并后: {len(merged)}天')

# 相关性
for horizon in [1, 5, 20, 60, 120, 250]:
    merged[f'hs300_fwd_{horizon}d'] = merged['hs300'].shift(-horizon) / merged['hs300'] - 1
    
print('\n铜油比 vs 沪深300 收益率相关性:')
for horizon in [1, 5, 20, 60, 120, 250]:
    col = f'hs300_fwd_{horizon}d'
    valid = merged[[col, 'cu_oil']].dropna()
    if len(valid) > 30:
        corr = valid['cu_oil'].corr(valid[col])
        print(f'  后{horizon:3d}天: r={corr:+.3f}')

# 铜油比分位判断
merged['cu_oil_pct'] = merged['cu_oil'].rolling(252, min_periods=100).rank(pct=True)

print(f'\n当前铜油比: {merged["cu_oil"].iloc[-1]:.1f}')
print(f'  铜: {merged["cu"].iloc[-1]:.0f} 元/吨')
print(f'  油: {merged["oil"].iloc[-1]:.0f} 元/桶')
print(f'  1年分位: {merged["cu_oil_pct"].iloc[-1]*100:.0f}%')
print(f'  均值: {merged["cu_oil"].mean():.1f}')
print(f'  历史区间: {merged["cu_oil"].min():.0f} ~ {merged["cu_oil"].max():.0f}')

# 高分位 vs 低分位 → 沪深300收益
print('\n铜油比高分位(>70%) vs 低分位(<30%) 后N日沪深300收益:')
high = merged[merged['cu_oil_pct'] > 0.7]
low = merged[merged['cu_oil_pct'] < 0.3]
for h in [5, 20, 60, 120]:
    col = f'hs300_fwd_{h}d'
    h_ret = high[col].dropna() * 100
    l_ret = low[col].dropna() * 100
    if len(h_ret) > 10:
        print(f'  后{h:3d}天: 高铜油比 avg={h_ret.mean():+.2f}% med={h_ret.median():+.2f}% | '
              f'低铜油比 avg={l_ret.mean():+.2f}% med={l_ret.median():+.2f}% | '
              f'差值={h_ret.mean()-l_ret.mean():+.2f}%')

# 当前方向
cu_oil_now = merged['cu_oil'].iloc[-1]
cu_oil_1m_ago = merged['cu_oil'].iloc[-22] if len(merged) >= 22 else merged['cu_oil'].iloc[0]
trend = '上升' if cu_oil_now > cu_oil_1m_ago else '下降'
print(f'\n铜油比趋势(1月): {trend} ({cu_oil_1m_ago:.1f} → {cu_oil_now:.1f})')
print(f'含义: 铜油比上升=工业需求预期走强 vs 能源成本 = 风险偏好上升')
print(f'      铜油比下降=增长预期走弱 + 能源成本挤压 = 风险偏好下降')
