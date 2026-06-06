"""A股宏观信号快速扫描: 哪个最有用?"""
import akshare as ak, pandas as pd, numpy as np

# 沪深300
import tushare as ts
pro = ts.pro_api('6aa54b486cea6bdaedcf9fe758e16d037364dad9bcc4bd3b8d69a3d7')
hs300 = pro.index_daily(ts_code='000300.SH', start_date='20150101', end_date='20260605')
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300 = hs300.set_index('trade_date')['close'].sort_index()

def test_signal(name, series, hs300):
    """测试信号对沪深300的预测力"""
    merged = pd.DataFrame({'signal': series, 'hs300': hs300}).dropna()
    results = {}
    for h in [5, 20, 60]:
        merged[f'fwd_{h}d'] = merged['hs300'].shift(-h) / merged['hs300'] - 1
        corr = merged['signal'].corr(merged[f'fwd_{h}d'])
        # 高vs低分组
        hi = merged[merged['signal'] > merged['signal'].quantile(0.7)]
        lo = merged[merged['signal'] < merged['signal'].quantile(0.3)]
        h_ret = hi[f'fwd_{h}d'].dropna().mean() * 100
        l_ret = lo[f'fwd_{h}d'].dropna().mean() * 100
        results[h] = {'corr': f'{corr:+.3f}', 'hi': f'{h_ret:+.1f}%', 'lo': f'{l_ret:+.1f}%', 'diff': f'{h_ret-l_ret:+.1f}%'}
    return results

# ============================================
# 1. 股债性价比 (ERP) —— 最重要的A股估值锚
# ============================================
print('1. 股债性价比 (1/PE - 10Y国债)')
try:
    pe_data = pro.index_dailybasic(ts_code='000300.SH', start_date='20150101', end_date='20260605',
                                    fields='trade_date,pe')
    pe = pe_data.set_index(pd.to_datetime(pe_data['trade_date']))['pe']
    erp = 1/pe * 100  # 盈利收益率%
    # 用SHIBOR做债券代理
    shibor = ak.macro_china_shibor_all()
    shibor = shibor.iloc[:, [0, 1]].copy()
    shibor.columns = ['date', 'rate']
    shibor['date'] = pd.to_datetime(shibor['date'])
    shibor = shibor.set_index('date')['rate'].resample('D').ffill()
    
    merged = pd.DataFrame({'erp': erp, 'shibor': shibor, 'hs300': hs300}).dropna()
    merged['erp_spread'] = merged['erp'] - merged['shibor']
    
    for h in [20, 60, 120]:
        merged[f'fwd'] = merged['hs300'].shift(-h) / merged['hs300'] - 1
        corr = merged['erp_spread'].corr(merged['fwd'])
        hi = merged[merged['erp_spread'] > merged['erp_spread'].quantile(0.7)]['fwd'].mean()*100
        lo = merged[merged['erp_spread'] < merged['erp_spread'].quantile(0.3)]['fwd'].mean()*100
        now = merged['erp_spread'].iloc[-1]
        pct = (merged['erp_spread'] < now).mean() * 100
        print(f'  后{h:3d}d: r={corr:+.3f} | 高ERP={hi:+.1f}% vs 低ERP={lo:+.1f}% diff={hi-lo:+.1f}% | 当前={now:.1f}%({pct:.0f}%分位)')
except Exception as e:
    print(f'  Error: {e}')

# ============================================
# 2. 人民币汇率 (USDCNH) —— 外资情绪的即时温度计  
# ============================================
print('\n2. 离岸人民币 USD/CNH')
try:
    cnh = ak.fx_spot_quote()
    if cnh is not None and len(cnh) > 0:
        # Try currency pair data
        for pair in ['USDCNH', 'USD/CNH', '美元/离岸人民币']:
            pass
    # Fallback: use fx_pair
    try:
        cnh = ak.currency_boc_sina(symbol='美元')
        if cnh is not None:
            print(f'  BOC: {cnh.shape}')
    except:
        pass
    print('  ⚠️ 汇率接口需适配，以下用Tushare替代')
    
    # Tushare fx
    try:
        fx = pro.fx_obasic(trade_date='20260604', fields='ts_code,name')
        print(f'  FX codes: {fx.head(5).to_string() if fx is not None else "None"}')
    except:
        pass
except Exception as e:
    print(f'  Error: {e}')

# ============================================
# 3. 铁矿石期货 —— 中国基建/地产的纯价格信号
# ============================================
print('\n3. 铁矿石 vs 沪深300')
try:
    io = ak.futures_main_sina(symbol='I0')
    io = io.iloc[:, [0, 1]].copy()
    io.columns = ['date', 'price']
    io['date'] = pd.to_datetime(io['date'])
    io = io.set_index('date')['price']
    
    merged = pd.DataFrame({'iron': io, 'hs300': hs300}).dropna()
    for h in [20, 60, 120]:
        merged[f'fwd'] = merged['hs300'].shift(-h) / merged['hs300'] - 1
        corr = merged['iron'].corr(merged['fwd'])
        hi = merged[merged['iron'] > merged['iron'].quantile(0.7)]['fwd'].mean()*100
        lo = merged[merged['iron'] < merged['iron'].quantile(0.3)]['fwd'].mean()*100
        print(f'  后{h:3d}d: r={corr:+.3f} | 高铁矿={hi:+.1f}% vs 低铁矿={lo:+.1f}% diff={hi-lo:+.1f}%')
except Exception as e:
    print(f'  Error: {e}')

# ============================================
# 4. 信用利差 —— A股内部的风险定价
# ============================================
print('\n4. 信用利差 (AA企业债 - 国债)')
try:
    # 用AKShare
    bond_aa = ak.bond_china_yield(start_date='20150101', end_date='20260605')
    if bond_aa is not None and len(bond_aa) > 0:
        print(f'  企业债: {bond_aa.shape}, cols: {list(bond_aa.columns)[:6]}')
    else:
        print('  数据空')
except Exception as e:
    print(f'  Error: {e}')

# ============================================
# 5. 金铜比 —— 全球风险偏好的反向指标
# ============================================
print('\n5. 金铜比 (AU0/CU0) vs 沪深300')
try:
    au = ak.futures_main_sina(symbol='AU0')
    au = au.iloc[:, [0, 1]].copy()
    au.columns = ['date', 'au']
    au['date'] = pd.to_datetime(au['date'])
    au = au.set_index('date')['au']
    
    cu = ak.futures_main_sina(symbol='CU0')
    cu = cu.iloc[:, [0, 1]].copy()
    cu.columns = ['date', 'cu']
    cu['date'] = pd.to_datetime(cu['date'])
    cu = cu.set_index('date')['cu']
    
    merged = pd.DataFrame({'au': au, 'cu': cu, 'hs300': hs300}).dropna()
    merged['au_cu'] = merged['au'] / merged['cu']
    
    for h in [20, 60, 120]:
        merged[f'fwd'] = merged['hs300'].shift(-h) / merged['hs300'] - 1
        corr = merged['au_cu'].corr(merged['fwd'])
        # 注意: 金铜比高=风险厌恶 → 应该做反向
        hi = merged[merged['au_cu'] > merged['au_cu'].quantile(0.7)]['fwd'].mean()*100  # 高金铜比=恐慌
        lo = merged[merged['au_cu'] < merged['au_cu'].quantile(0.3)]['fwd'].mean()*100  # 低金铜比=乐观
        print(f'  后{h:3d}d: r={corr:+.3f} | 高金铜(恐慌)={hi:+.1f}% vs 低金铜(贪婪)={lo:+.1f}% diff={lo-hi:+.1f}%')
except Exception as e:
    print(f'  Error: {e}')

print('\nDone.')
