"""
安检门规则A股真实数据验证
验证3条最可量化的规则：
1. Marks: 恐慌日(=涨跌比极端)后N日市场表现
2. Marks: PE分位>80%后的收益（需Tushare获取PE）
3. 洪灏: 中美息差-沪深300相关性（需AKShare获取债券收益率）
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

CACHE = r'C:\Users\Zenta\.qclaw\workspace-agent-b9c8dcea\data\cache'

# ============================================================
# VERIFICATION 1: Marks恐慌日规则
# 规则: "跌停/涨停>5 → 恐慌买入" → 我们检验恐慌日后N日收益
# 代理: 日下跌占比>70% 且 平均涨跌幅<-3% → 极端恐慌日
# ============================================================
print("=" * 60)
print("验证1: Marks恐慌日规则 → A股恐慌日后收益")
print("=" * 60)

bp = pd.read_pickle(f'{CACHE}/bp_daily_2019_2026.pkl')
bp['trade_date'] = pd.to_datetime(bp['trade_date'])

# 每日市场广度
daily_stats = bp.groupby('trade_date').agg(
    n_stocks=('pct_chg', 'count'),
    avg_pct=('pct_chg', 'mean'),
    up_ratio=('pct_chg', lambda x: (x > 0).mean()),
    down_ratio=('pct_chg', lambda x: (x < 0).mean()),
    extreme_down=('pct_chg', lambda x: (x < -9).sum()),  # 跌停级
    extreme_up=('pct_chg', lambda x: (x > 9).sum()),      # 涨停级
    med_pct=('pct_chg', 'median'),
).reset_index()

# 定义恐慌日: 下跌占比>70% 且 跌停级数量 > 涨停级数量*3
daily_stats['panic_score'] = (
    (daily_stats['down_ratio'] > 0.70).astype(int) +
    (daily_stats['extreme_down'] > daily_stats['extreme_up'] * 3).astype(int) +
    (daily_stats['avg_pct'] < -2.5).astype(int)
)
daily_stats['is_panic'] = daily_stats['panic_score'] >= 2

# 定义狂喜日: 上涨占比>70%
daily_stats['is_euphoria'] = (daily_stats['up_ratio'] > 0.70) & (daily_stats['extreme_up'] > daily_stats['extreme_down'] * 3)

# 计算恐慌日后N日收益
idx = pd.read_pickle(f'{CACHE}/sh000001_daily.pkl')
idx['trade_date'] = pd.to_datetime(idx['trade_date'])
idx = idx.set_index('trade_date')['close']

for horizon in [1, 5, 10, 20, 40]:
    daily_stats[f'fwd_{horizon}d_ret'] = np.nan
    for i, row in daily_stats.iterrows():
        target_date = row['trade_date'] + pd.Timedelta(days=horizon)
        # 找最近的交易日
        future_close = idx[idx.index <= target_date + pd.Timedelta(days=3)]
        if len(future_close) > 0:
            closest = future_close.index[-1]
            daily_stats.at[i, f'fwd_{horizon}d_ret'] = (future_close.iloc[-1] / idx.loc[row['trade_date']] - 1) * 100

panic_days = daily_stats[daily_stats['is_panic']]
normal_days = daily_stats[~daily_stats['is_panic'] & ~daily_stats['is_euphoria']]
euphoria_days = daily_stats[daily_stats['is_euphoria']]

print(f"\n恐慌日: {len(panic_days)}天 ({len(panic_days)/len(daily_stats)*100:.1f}%)")
print(f"正常日: {len(normal_days)}天")
print(f"狂喜日: {len(euphoria_days)}天")

print("\n恐慌日后收益 vs 正常日:")
for h in [1, 5, 10, 20, 40]:
    p_ret = panic_days[f'fwd_{h}d_ret'].dropna()
    n_ret = normal_days[f'fwd_{h}d_ret'].dropna()
    if len(p_ret) == 0: continue
    print(f"  后{h:2d}天: 恐慌日 avg={p_ret.mean():+.2f}% med={p_ret.median():+.2f}% 胜率={(p_ret>0).mean()*100:.0f}% | "
          f"正常日 avg={n_ret.mean():+.2f}% med={n_ret.median():+.2f}% 胜率={(n_ret>0).mean()*100:.0f}%")

print(f"\n最近5个恐慌日:")
recent_panic = panic_days.nlargest(5, 'trade_date')
for _, r in recent_panic.iterrows():
    print(f"  {r['trade_date'].strftime('%Y-%m-%d')} avg={r['avg_pct']:+.2f}% down_ratio={r['down_ratio']*100:.0f}% "
          f"跌停级:{int(r['extreme_down'])} 涨停级:{int(r['extreme_up'])} → 后20天:{r.get('fwd_20d_ret', np.nan):+.2f}%")

# Marks结论: 恐慌日后市场是否反弹(均值回归) vs 继续跌(动量)
# 如果恐慌日后收益>正常日 → Marks的"恐惧时买入"在A股成立

print("\n" + "=" * 60)
print("验证2: Marks PE分位规则 → PE>80%分位后N日收益")
print("=" * 60)

# 尝试从Tushare获取PE数据
try:
    import tushare as ts
    pro = ts.pro_api('6aa54b486cea6bdaedcf9fe758e16d037364dad9bcc4bd3b8d69a3d7')
    
    # 获取沪深300日估值数据
    pe_data = pro.index_dailybasic(
        ts_code='000300.SH',
        start_date='20180101',
        end_date='20260605',
        fields='trade_date,pe,pe_ttm,pb'
    )
    if pe_data is not None and len(pe_data) > 0:
        pe_data['trade_date'] = pd.to_datetime(pe_data['trade_date'])
        pe_data = pe_data.sort_values('trade_date')
        print(f"获取PE数据: {len(pe_data)}天 PE范围: {pe_data['pe'].min():.1f} ~ {pe_data['pe'].max():.1f}")
        
        # 计算PE滚动分位
        for window in [252, 504, 756]:  # 1年/2年/3年滚动
            rolling_min = pe_data['pe'].rolling(window, min_periods=window//2).min()
            rolling_max = pe_data['pe'].rolling(window, min_periods=window//2).max()
            pe_data[f'pe_pct_{window}d'] = (pe_data['pe'] - rolling_min) / (rolling_max - rolling_min)
        
        # 在PE>80%分位日买入后N日收益
        idx300 = pro.index_daily(ts_code='000300.SH', start_date='20180101', end_date='20260531')
        idx300['trade_date'] = pd.to_datetime(idx300['trade_date'])
        idx300 = idx300.set_index('trade_date')['close'].sort_index()
        
        for window in [252, 504]:
            col = f'pe_pct_{window}d'
            if col not in pe_data.columns: continue
            
            pe_data_merged = pe_data.set_index('trade_date').join(
                pd.DataFrame({'close': idx300}), how='inner'
            )
            
            high_pe = pe_data_merged[pe_data_merged[col] > 0.8]
            low_pe = pe_data_merged[pe_data_merged[col] < 0.2]
            mid_pe = pe_data_merged[(pe_data_merged[col] >= 0.4) & (pe_data_merged[col] <= 0.6)]
            
            print(f"\nPE分位(滚动{window}天) > 80%: {len(high_pe)}天")
            for h in [5, 20, 60, 120]:
                high_ret = []
                low_ret = []
                mid_ret = []
                for df, lst in [(high_pe, high_ret), (low_pe, low_ret), (mid_pe, mid_ret)]:
                    for i in range(len(df) - 1):
                        t = df.index[i]
                        future_t = t + pd.Timedelta(days=h)
                        future = idx300[idx300.index <= future_t + pd.Timedelta(days=5)]
                        if len(future) > 0 and future.index[-1] > t:
                            ret = future.iloc[-1] / idx300.loc[t] - 1
                            lst.append(ret * 100)
                
                if high_ret:
                    print(f"  后{h:3d}天: 高PE avg={np.mean(high_ret):+.2f}% med={np.median(high_ret):+.2f}% 胜率={np.mean(np.array(high_ret)>0)*100:.0f}% | "
                          f"低PE avg={np.mean(low_ret):+.2f}% med={np.median(low_ret):+.2f}% 胜率={np.mean(np.array(low_ret)>0)*100:.0f}% | "
                          f"中PE avg={np.mean(mid_ret):+.2f}% | 高-低差值={np.mean(high_ret)-np.mean(low_ret):+.2f}%")
    else:
        print("Tushare PE数据获取失败或为空")
        
except Exception as e:
    print(f"Tushare PE获取失败: {e}")
    print("需要确认Tushare Token和网络")

print("\n" + "=" * 60)
print("验证3: 洪灏 中美息差-沪深300相关性")
print("=" * 60)

# 尝试从AKShare获取中美债券收益率
try:
    import akshare as ak
    
    # 中国10年期国债收益率
    try:
        cn_bond = ak.bond_china_yield(start_date='20180101', end_date='20260605')
        if cn_bond is not None and len(cn_bond) > 0:
            print(f"获取中国国债数据: {len(cn_bond)}行")
            print(f"Columns: {list(cn_bond.columns)}")
        else:
            print("中国国债数据为空")
    except Exception as e:
        print(f"中国国债获取失败: {e}")
    
    # 美国10年期国债收益率
    try:
        us_bond = ak.bond_us_yield()
        if us_bond is not None and len(us_bond) > 0:
            print(f"获取美国国债数据: {len(us_bond)}行")
            print(f"Columns: {list(us_bond.columns)}")
        else:
            print("美国国债数据为空")
    except Exception as e:
        print(f"美国国债获取失败: {e}")
        
except ImportError:
    print("akshare未安装")

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
能够直接验证的规则（有现有数据）:
  ✅ Marks恐慌日 → 用bp_daily涨跌比
  ✅ Marks PE分位 → 用Tushare PE数据
  ⚠️ 洪灏息差 → 需要债市数据源，数据获取可能失败

之后补充验证:
  - 冯柳三杀分类（需要财务数据+股价）
  - 冯柳关注度×购买度（需要社交数据）
  - 洪灏850天周期（可用上证综指直接算）
""")
