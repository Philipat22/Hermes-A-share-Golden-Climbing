"""向量化三杀验证 - pandas 3.0 merge_asof bug workaround"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

CACHE = r'C:\Users\Zenta\.qclaw\workspace-agent-b9c8dcea\data\cache'

print("Loading data...")
bp = pd.read_pickle(f'{CACHE}/bp_daily_2019_2026.pkl')
bp['trade_date'] = pd.to_datetime(bp['trade_date'])
bp = bp.sort_values(['ts_code','trade_date'])

# 向量化DD
print("Computing DD...")
bp['roll_high'] = bp.groupby('ts_code')['close'].transform(lambda x: x.rolling(252, min_periods=100).max())
bp['DD'] = (bp['close'] / bp['roll_high'] - 1) * 100

# 向前收益
print("Computing forward returns...")
for h in [5, 10, 20, 40, 60]:
    bp[f'fwd{h}'] = bp.groupby('ts_code')['close'].transform(lambda x: x.shift(-h))
    bp[f'ret{h}'] = (bp[f'fwd{h}'] / bp['close'] - 1) * 100

# DD信号
sigs = bp[bp['DD'] <= -25].copy()
print(f'DD<=-25% signals: {len(sigs)}')

# 加载财务 - 用searchsorted做无bug匹配
fina = pd.read_pickle(f'{CACHE}/bp_fina_2019_2026.pkl')
fina['ann_date'] = pd.to_datetime(fina['ann_date'])
fina = fina.dropna(subset=['roe','tr_yoy']).sort_values(['ts_code','ann_date'])
print(f'Financial records: {len(fina)}')

# 手动按组匹配
print("Matching financials...")
results = []
for ts, group in sigs.groupby('ts_code'):
    fgroup = fina[fina['ts_code'] == ts]
    if len(fgroup) == 0:
        continue
    f_ann = fgroup['ann_date'].values
    f_roe = fgroup['roe'].values
    f_tr = fgroup['tr_yoy'].values
    f_np = fgroup['netprofit_yoy'].values
    
    for _, row in group.iterrows():
        sd = row['trade_date']
        # 找ann_date <= sd的最大索引
        idx = np.searchsorted(f_ann, sd, side='right') - 1
        if idx >= 0:
            results.append({
                'ts_code': ts, 'date': sd,
                'DD': row['DD'], 'close': row['close'],
                'roe': f_roe[idx], 'tr_yoy': f_tr[idx],
                'netprofit_yoy': f_np[idx],
                'ret5': row.get('ret5', np.nan), 'ret10': row.get('ret10', np.nan),
                'ret20': row.get('ret20', np.nan), 'ret40': row.get('ret40', np.nan),
                'ret60': row.get('ret60', np.nan)
            })

ms = pd.DataFrame(results)
print(f'Matched: {len(ms)}')

# 三杀分类
ms['kill_type'] = 'kill_earnings'
ms.loc[(ms['roe'] >= 5) & (ms['tr_yoy'] > 0), 'kill_type'] = 'kill_valuation'
ms.loc[(ms['roe'] < 0) & (ms['tr_yoy'] < -20), 'kill_type'] = 'kill_logic'

print(f'\n三杀分布:')
for kt in ['kill_valuation', 'kill_earnings', 'kill_logic']:
    print(f'  {kt}: {(ms.kill_type==kt).sum()}')

print(f'\n{"="*60}')
print(f'冯柳三杀分类: DD<=-25%信号后N日收益')
print(f'{"="*60}')
for kt in ['kill_valuation', 'kill_earnings', 'kill_logic']:
    sub = ms[ms['kill_type'] == kt]
    print(f'\n{kt} (n={len(sub)}):')
    for h in [5, 10, 20, 40, 60]:
        r = sub[f'ret{h}'].dropna()
        if len(r) > 0:
            print(f'  {h:2d}d: avg={r.mean():+.2f}% med={r.median():+.2f}% win={(r>0).mean()*100:.0f}%')

print(f'\n{"="*60}')
print(f'整体DD<=-25%（不分类）')
print(f'{"="*60}')
for h in [5, 10, 20, 40, 60]:
    r = ms[f'ret{h}'].dropna()
    print(f'  {h:2d}d: avg={r.mean():+.2f}% med={r.median():+.2f}% win={(r>0).mean()*100:.0f}% n={len(r)}')

print(f'\n{"="*60}')
print(f'DD深度 vs 收益')
print(f'{"="*60}')
for lo, hi, label in [(-25,-30,'DD25-30%'),(-30,-40,'DD30-40%'),(-40,-50,'DD40-50%'),(-50,-200,'DD>50%')]:
    sub = ms[(ms['DD'] <= lo) & (ms['DD'] > hi)]
    if len(sub) < 10: continue
    r20 = sub['ret20'].dropna()
    r40 = sub['ret40'].dropna()
    print(f'  {label:12s} n={len(sub):5d} 20d={r20.mean():+.2f}% med={r20.median():+.2f}% win={(r20>0).mean()*100:.0f}%'
          f'  40d={r40.mean():+.2f}% med={r40.median():+.2f}% win={(r40>0).mean()*100:.0f}%')

print('\nDone.')
