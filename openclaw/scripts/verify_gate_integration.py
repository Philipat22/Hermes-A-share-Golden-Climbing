"""
安检门集成回测: 四人规则 → DD信号过滤 → 收益对比
核心问题: 过安检门的信号 vs 没过的信号，收益有显著差异吗？
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

CACHE = r'C:\Users\Zenta\.qclaw\workspace-agent-b9c8dcea\data\cache'

print("=" * 70)
print("安检门集成回测: 四人规则 → DD≥25%信号 → 前后收益对比")
print("=" * 70)

# ============================================================
# 1. 数据准备
# ============================================================
print("\n[1/5] 加载数据...")
bp = pd.read_pickle(f'{CACHE}/bp_daily_2019_2026.pkl')
bp['trade_date'] = pd.to_datetime(bp['trade_date'])
bp = bp.sort_values(['ts_code','trade_date'])

# DD计算
bp['roll_high'] = bp.groupby('ts_code')['close'].transform(lambda x: x.rolling(252, min_periods=100).max())
bp['DD'] = (bp['close'] / bp['roll_high'] - 1) * 100

# Forward returns
bp = bp.sort_values(['ts_code','trade_date'])
for h in [5, 10, 20, 40, 60]:
    bp[f'fwd{h}'] = bp.groupby('ts_code')['close'].transform(lambda x: x.shift(-h))
    bp[f'ret{h}'] = (bp[f'fwd{h}'] / bp['close'] - 1) * 100

# 沪深300指数
idx300 = pd.read_pickle(f'{CACHE}/sh000001_daily.pkl')
idx300['trade_date'] = pd.to_datetime(idx300['trade_date'])
idx300 = idx300.set_index('trade_date').sort_index()

# ============================================================
# 2. 提取DD≥25%信号
# ============================================================
print("[2/5] 提取DD≥25%信号...")
sigs = bp[bp['DD'] <= -25].copy()

# 每个信号附加: 日期、DD、retN 和 市场状态上下文
# 市值加权信号（去重：同一天同只票只取最近交易日最低DD那个）
sigs = sigs.sort_values(['ts_code','trade_date','DD'])

# 每天每只票的最新DD信号
sigs['ret5'] = sigs['ret5'].astype(float)
sigs['ret10'] = sigs['ret10'].astype(float)
sigs['ret20'] = sigs['ret20'].astype(float)
sigs['ret40'] = sigs['ret40'].astype(float)
sigs['ret60'] = sigs['ret60'].astype(float)

print(f"   原始DD信号: {len(sigs)}条")

# ============================================================
# 3. 安检门规则计算
# ============================================================
print("[3/5] 计算安检门规则...")

# ---- 3a. 市场广度状态 (Marks恐慌日) ----
daily_stats = bp.groupby('trade_date').agg(
    n_stocks=('pct_chg','count'),
    avg_pct=('pct_chg','mean'),
    up_ratio=('pct_chg', lambda x: (x>0).mean()),
    down_ratio=('pct_chg', lambda x: (x<0).mean()),
    extreme_down=('pct_chg', lambda x: (x<-9).sum()),
    extreme_up=('pct_chg', lambda x: (x>9).sum()),
).reset_index()

daily_stats['panic_score'] = (
    (daily_stats['down_ratio'] > 0.70).astype(int) +
    (daily_stats['extreme_down'] > daily_stats['extreme_up'] * 3).astype(int) +
    (daily_stats['avg_pct'] < -2.5).astype(int)
)
daily_stats['is_panic'] = daily_stats['panic_score'] >= 2
daily_stats['is_euphoria'] = (daily_stats['up_ratio'] > 0.70) & (daily_stats['extreme_up'] > daily_stats['extreme_down'] * 3)

# 合并到信号
sigs = sigs.merge(daily_stats[['trade_date','is_panic','is_euphoria','avg_pct','down_ratio']], 
                  left_on='trade_date', right_on='trade_date', how='left')

# ---- 3b. PE分位 (Marks) ----
try:
    import tushare as ts
    pro = ts.pro_api('6aa54b486cea6bdaedcf9fe758e16d037364dad9bcc4bd3b8d69a3d7')
    pe_data = pro.index_dailybasic(ts_code='000300.SH', start_date='20170101', end_date='20260605',
                                    fields='trade_date,pe')
    if pe_data is not None and len(pe_data) > 0:
        pe_data['trade_date'] = pd.to_datetime(pe_data['trade_date'])
        pe_data = pe_data.sort_values('trade_date')
        for w in [252, 504, 756]:
            rmin = pe_data['pe'].rolling(w, min_periods=w//2).min()
            rmax = pe_data['pe'].rolling(w, min_periods=w//2).max()
            pe_data[f'pe_pct_{w}'] = (pe_data['pe'] - rmin) / (rmax - rmin)
        sigs = sigs.merge(pe_data[['trade_date','pe','pe_pct_252','pe_pct_504','pe_pct_756']],
                         left_on='trade_date', right_on='trade_date', how='left')
        print("   PE数据: OK")
    else:
        print("   PE数据: 获取失败，跳过PE规则")
except Exception as e:
    print(f"   PE数据: {e}")

# ---- 3c. 850天周期位置 (洪灏) ----
# 上证综指850日滚动最低点
window = 850
idx = pd.read_pickle(f'{CACHE}/sh000001_daily.pkl')
idx['trade_date'] = pd.to_datetime(idx['trade_date'])
idx = idx.set_index('trade_date').sort_index()

idx['roll_min_850'] = idx['close'].rolling(window, min_periods=200).min()
idx['dist_from_850_low'] = (idx['close'] / idx['roll_min_850'] - 1) * 100
idx['roll_max_850'] = idx['close'].rolling(window, min_periods=200).max()

# 找最近底部
bottoms = []
for i in range(window, len(idx)):
    wdata = idx['close'].iloc[i-window:i]
    min_date = wdata.idxmin()
    if not bottoms or (min_date - bottoms[-1]).days > 150:
        bottoms.append(min_date)

def get_days_since_last_bottom(date, bottoms):
    prev = [b for b in bottoms if b <= date]
    if prev: return (date - prev[-1]).days
    return 9999

sigs['days_since_850_low'] = sigs['trade_date'].apply(
    lambda d: get_days_since_last_bottom(pd.Timestamp(d), bottoms))
sigs['years_since_bottom'] = sigs['days_since_850_low'] / 365.25

idx_merged = idx.reset_index()[['trade_date','dist_from_850_low']]
sigs = sigs.merge(idx_merged, left_on='trade_date', right_on='trade_date', how='left')
print(f"   850天周期: 找到{len(bottoms)}个底部")

# ---- 3d. 冯柳三杀 (财务数据) ----
fina = pd.read_pickle(f'{CACHE}/bp_fina_2019_2026.pkl')
fina['ann_date'] = pd.to_datetime(fina['ann_date'])
fina = fina.dropna(subset=['roe','tr_yoy']).sort_values(['ts_code','ann_date'])

# 给每个信号匹配最近一期财务
print("   匹配财务数据...")
matched = []
for ts, group in sigs.groupby('ts_code'):
    fgroup = fina[fina['ts_code'] == ts]
    if len(fgroup) == 0: continue
    f_ann = fgroup['ann_date'].values
    f_roe = fgroup['roe'].values
    f_tr = fgroup['tr_yoy'].values
    
    for orig_idx, row in group.iterrows():
        sd = row['trade_date']
        fi = np.searchsorted(f_ann, sd, side='right') - 1
        if fi >= 0:
            matched.append({
                'orig_idx': orig_idx,
                'f_roe': f_roe[fi], 'f_tr_yoy': f_tr[fi]
            })

fmatch = pd.DataFrame(matched).set_index('orig_idx')
sigs['f_roe'] = fmatch['f_roe']
sigs['f_tr_yoy'] = fmatch['f_tr_yoy']
print(f"   三杀数据: {len(fmatch)}/{len(sigs)}条匹配")

# ============================================================
# 4. 应用安检门规则
# ============================================================
print("[4/5] 应用安检门规则...")

# 初始化规则列
sigs['gate_marks_panic'] = False  # 恐慌日 → Marks说应该买
sigs['gate_marks_pe'] = False     # PE<20% → Marks说应该买
sigs['gate_honghao_cycle'] = False # 距850天底部<365天 → 洪灏说周期底部
sigs['gate_honghao_trend'] = False # 距850天低点涨幅<20% → 还在底部区域
sigs['gate_fengliu_depth'] = False # DD>=40% → 深度超跌（替代三杀）

# Marks: 恐慌日规则
sigs.loc[sigs['is_panic'] == True, 'gate_marks_panic'] = True

# Marks: PE<20%分位
if 'pe_pct_252' in sigs.columns:
    sigs.loc[sigs['pe_pct_252'] < 0.2, 'gate_marks_pe'] = True

# 洪灏: 距850天底部<365天
sigs.loc[sigs['days_since_850_low'] < 365, 'gate_honghao_cycle'] = True

# 洪灏: 距850天低点涨幅<20%（还在底部区域）
sigs.loc[sigs['dist_from_850_low'] < 20, 'gate_honghao_trend'] = True

# 冯柳: DD>=40%（深度超跌）
sigs.loc[sigs['DD'] <= -40, 'gate_fengliu_depth'] = True

# 综合规则 (AND: 所有通过的组合)
# 至少2条规则通过 = 安检门放行
rule_cols = ['gate_marks_panic', 'gate_marks_pe', 'gate_honghao_cycle', 
             'gate_honghao_trend', 'gate_fengliu_depth']
sigs['n_rules_passed'] = sigs[rule_cols].sum(axis=1)
sigs['gate_passed'] = sigs['n_rules_passed'] >= 2

print(f"\n规则统计:")
for col in rule_cols:
    n = sigs[col].sum()
    print(f"  {col}: {n}条 ({n/len(sigs)*100:.1f}%)")

print(f"\n  通过≥2条: {sigs['gate_passed'].sum()}条 ({sigs['gate_passed'].mean()*100:.1f}%)")
print(f"  未通过: {(~sigs['gate_passed']).sum()}条")

# ============================================================
# 5. 收益对比: 安检门过了 vs 没过
# ============================================================
print("\n" + "=" * 70)
print("核心结果: 安检门通过 vs 未通过 → 后N日收益对比")
print("=" * 70)

passed = sigs[sigs['gate_passed']]
failed = sigs[~sigs['gate_passed']]

for h in [5, 10, 20, 40, 60]:
    p_ret = passed[f'ret{h}'].dropna()
    f_ret = failed[f'ret{h}'].dropna()
    if len(p_ret) == 0 or len(f_ret) == 0: continue
    
    p_mean, p_med, p_win = p_ret.mean(), p_ret.median(), (p_ret>0).mean()*100
    f_mean, f_med, f_win = f_ret.mean(), f_ret.median(), (f_ret>0).mean()*100
    diff = p_mean - f_mean
    
    # 简单t检验
    from scipy import stats
    try:
        t_stat, p_val = stats.ttest_ind(p_ret.dropna(), f_ret.dropna(), equal_var=False)
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
    except:
        p_val = 1.0; sig = ""
    
    print(f"  {h:2d}天: 通过={p_mean:+.2f}% med={p_med:+.2f}% win={p_win:.0f}% (n={len(p_ret)}) | "
          f"未通过={f_mean:+.2f}% med={f_med:+.2f}% win={f_win:.0f}% (n={len(f_ret)}) | "
          f"差值={diff:+.2f}% p={p_val:.3f}{sig}")

# 逐条规则的收益
print("\n" + "=" * 70)
print("逐条规则: 满足 vs 不满足 → 后20天收益")
print("=" * 70)

rule_labels = {
    'gate_marks_panic': 'Marks:恐慌日买入',
    'gate_marks_pe': 'Marks:PE<20%买入',
    'gate_honghao_cycle': '洪灏:距底部<1年',
    'gate_honghao_trend': '洪灏:距850低点<20%',
    'gate_fengliu_depth': '冯柳:DD≥40%深跌',
}

for col, label in rule_labels.items():
    if col not in sigs.columns: continue
    meet = sigs[sigs[col] & sigs['ret20'].notna()]
    not_meet = sigs[~sigs[col] & sigs['ret20'].notna()]
    if len(meet) < 10: continue
    
    m_ret = meet['ret20'].mean()
    n_ret = not_meet['ret20'].mean()
    diff = m_ret - n_ret
    try:
        _, p_val = stats.ttest_ind(meet['ret20'].dropna(), not_meet['ret20'].dropna(), equal_var=False)
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
    except:
        p_val = 1.0; sig = ""
    
    print(f"  {label:20s}: 满足={m_ret:+.2f}% (n={len(meet)}) | 不满足={n_ret:+.2f}% (n={len(not_meet)}) | "
          f"差值={diff:+.2f}% p={p_val:.3f}{sig}")

# 规则数量的边际效应
print("\n" + "=" * 70)
print("规则数量效应: 通过n条规则 → 后20天收益")
print("=" * 70)
for n in range(6):
    subset = sigs[(sigs['n_rules_passed'] == n) & sigs['ret20'].notna()]
    if len(subset) < 5: continue
    r = subset['ret20']
    print(f"  通过{n}条规则: avg={r.mean():+.2f}% med={r.median():+.2f}% win={(r>0).mean()*100:.0f}% n={len(r)} | DD均值={subset['DD'].mean():.1f}%")

# DD深度作为benchmark
print("\n" + "=" * 70)
print("Baseline: 纯DD深度 vs 规则过滤")
print("=" * 70)
for lo, hi, label in [(-25,-30,'DD25-30'),(-30,-40,'DD30-40'),(-40,-50,'DD40-50'),(-50,-200,'DD50+')]:
    sub = sigs[(sigs['DD'] <= lo) & (sigs['DD'] > hi) & sigs['ret20'].notna()]
    sub_gated = sub[sub['gate_passed']]
    sub_ungated = sub[~sub['gate_passed']]
    if len(sub_gated) < 10: continue
    print(f"  {label}: 全部={sub['ret20'].mean():+.2f}% | 过安检={sub_gated['ret20'].mean():+.2f}% (n={len(sub_gated)}) | "
          f"没过={sub_ungated['ret20'].mean():+.2f}% (n={len(sub_ungated)})")

print("\nDone.")
