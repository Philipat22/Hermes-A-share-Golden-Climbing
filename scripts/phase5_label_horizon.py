"""Phase 5: Label Horizon Comparison
Test 5d / 10d / 20d / 60d forward return windows with top-40 features"""
import sys, os, json, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# ── 1. Load data ────────────────────────────────────────────────────────
print('Loading data...')
with open('data/cache/factor_dataset.pkl', 'rb') as f:
    data = pickle.load(f)
with open('data/cache/backtest_prices.pkl', 'rb') as f:
    prices = pickle.load(f)

X_full = data['X']  # (49859, 160)
meta = data['meta'].copy()
names = data['factor_names']
print(f'  Factors: {len(names)}, Samples: {len(meta)}')

# ── 2. Get top-40 feature indices ────────────────────────────────────────
# Recompute importance to rank features
meta['dt'] = pd.to_datetime(meta['datetime'])
split_dt = meta['dt'].quantile(0.7)
train_idx = np.where((meta['dt'] < split_dt).values)[0]
test_idx  = np.where((meta['dt'] >= split_dt).values)[0]
y_all = data['y']

params = {
    'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
    'num_leaves': 63, 'max_depth': 6, 'learning_rate': 0.03,
    'feature_fraction': 0.8, 'bagging_fraction': 0.85,
    'bagging_freq': 5, 'min_child_samples': 50, 'random_state': 42,
}
dtrain = lgb.Dataset(X_full[train_idx], y_all[train_idx])
dval = lgb.Dataset(X_full[test_idx], y_all[test_idx], reference=dtrain)
model = lgb.train(params, dtrain, num_boost_round=500,
                  valid_sets=[dval], valid_names=['valid'],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
imp = model.feature_importance(importance_type='gain')
feat_imp = list(zip(names, imp))
feat_imp.sort(key=lambda x: -x[1])
top40_names = [f for f, _ in feat_imp[:40]]
name_to_idx = {n: i for i, n in enumerate(names)}
top40_idx = [name_to_idx[n] for n in top40_names]
X_top40 = X_full[:, top40_idx]
print(f'  Top-40 selected')

# ── 3. Compute forward returns for all horizons ──────────────────────────
print('\nComputing forward returns for each horizon...')
# Build price lookup: stock_code -> {dates_array, closes_array}
stock_lookup = {}
for ts_code, df in prices.items():
    df_sorted = df.sort_values('date').reset_index(drop=True)
    stock_lookup[ts_code] = {
        'dates': df_sorted['date'].astype(str).values,
        'closes': df_sorted['close'].values.astype(np.float64),
    }

def compute_forward_ret(meta_df, horizon):
    """Compute forward return over N trading days for each sample."""
    col = f'fwd_{horizon}d'
    meta_df[col] = np.nan
    count = 0
    for idx, row in meta_df.iterrows():
        sd = stock_lookup.get(row['vt_symbol'])
        if sd is None:
            continue
        # Find date position
        date_str = str(row['datetime'])[:10]
        positions = np.where(sd['dates'] == date_str)[0]
        if len(positions) == 0:
            continue
        pos = positions[0]
        j = pos + horizon
        if j >= len(sd['closes']):
            continue
        ret = (sd['closes'][j] - sd['closes'][pos]) / sd['closes'][pos]
        meta_df.at[idx, col] = ret
        count += 1
        if count % 10000 == 0:
            print(f'  Computed {count}/{len(meta_df)}...')
    print(f'  Done: {count}/{len(meta_df)} valid for {horizon}d')
    return meta_df[col].values

# Compute for each horizon (5, 10, 20, 60)
horizons = [5, 10, 20, 60]
thresholds = {5: 0.05, 10: 0.075, 20: 0.10, 60: 0.15}

for h in horizons:
    rets = compute_forward_ret(meta, h)
    # Also add a percentage-based threshold: top 15% per horizon
    pct_cutoff = np.nanpercentile(rets, 85) if np.sum(~np.isnan(rets)) > 0 else 0.10
    print(f'  {h}d: threshold={thresholds[h]:.1%}, top15% cutoff={pct_cutoff:.2%}')

# ── 4. Train & compare models for each horizon ──────────────────────────
print('\n' + '='*70)
print('LABEL HORIZON COMPARISON')
print('='*70)

results = {}
for h in horizons:
    fwd_col = f'fwd_{h}d'
    # Create label: forward_ret >= threshold
    th = thresholds[h]
    y_h = (meta[fwd_col].fillna(0) >= th).astype(np.int32).values
    
    # Time split
    train_mask = (meta['dt'] < split_dt).values
    test_mask = (meta['dt'] >= split_dt).values
    X_tr_h = X_top40[train_mask]
    X_te_h = X_top40[test_mask]
    y_tr_h = y_h[train_mask]
    y_te_h = y_h[test_mask]
    
    surge_rate_tr = y_tr_h.mean()
    surge_rate_te = y_te_h.mean()
    
    # Train
    print(f'\n── {h}d horizon (threshold >= {th:.0%}) ──')
    print(f'  Positive rate: train={surge_rate_tr:.1%}, test={surge_rate_te:.1%}')
    
    if surge_rate_tr < 0.01:
        print(f'  SKIP: too few positive samples')
        continue
    
    dtrain_h = lgb.Dataset(X_tr_h, y_tr_h)
    dval_h = lgb.Dataset(X_te_h, y_te_h, reference=dtrain_h)
    model_h = lgb.train(params, dtrain_h, num_boost_round=500,
                        valid_sets=[dval_h], valid_names=['valid'],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    probs_h = model_h.predict(X_te_h)
    if len(np.unique(y_te_h)) < 2:
        print(f'  SKIP: only one class in test set')
        continue
    auc_h = roc_auc_score(y_te_h, probs_h)
    
    # Evaluate excess return
    mte = meta.iloc[test_idx].copy()
    mte['score'] = probs_h
    mte['fwd_ret'] = meta[fwd_col].values[test_idx]
    
    print(f'  AUC: {auc_h:.4f}')
    print(f'  {"Thresh":>8} | {"Picks":>6} | {"Win%":>7} | {"AvgRet":>10}')
    print('  ' + '-'*40)
    
    best_excess, best_th, best_picks = 0, 0.50, 0
    for th_eval in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        picks = mte[mte['score'] >= th_eval]
        if len(picks) < 5:
            continue
        wr = (picks['fwd_ret'] > 0).mean()
        ar = picks['fwd_ret'].mean()
        print(f'  {th_eval:>8.2f} | {len(picks):>6} | {wr*100:>6.1f}% | {ar*100:>+9.2f}%')
        if ar > best_excess:
            best_excess, best_th, best_picks = ar, th_eval, len(picks)
    
    results[h] = {
        'auc': round(auc_h, 4),
        'best_excess': round(best_excess, 4),
        'best_threshold': best_th,
        'best_picks': best_picks,
        'surge_rate_test': round(surge_rate_te, 4),
    }
    print(f'  >> Best: excess={best_excess*100:+.2f}%  th={best_th:.2f}  picks={best_picks}')

# ── 5. Final comparison table ────────────────────────────────────────────
print('\n' + '='*70)
print('FINAL COMPARISON')
print('='*70)
print(f'{"Horizon":>8} | {"AUC":>6} | {"SurgeRate":>9} | {"Excess":>8} | {"Picks":>6} | {"Threshold"}>')
print('-'*60)
best_h, best_ex = 20, 0
for h in horizons:
    if h not in results:
        continue
    r = results[h]
    print(f'  {h:>4}d  | {r["auc"]:>6.4f} | {r["surge_rate_test"]*100:>7.2f}% | {r["best_excess"]*100:>+7.2f}% | {r["best_picks"]:>5}  | {r["best_threshold"]:.2f}')
    if r['best_excess'] > best_ex:
        best_h, best_ex = h, r['best_excess']

print(f'\n🏆 Best horizon: {best_h}d (excess={best_ex*100:+.2f}%)')

# Save report
report_path = os.path.join(ROOT, 'data', 'models', 'label_horizon_report.json')
with open(report_path, 'w') as f:
    json.dump({'best_horizon': best_h, 'results': results}, f, indent=2, ensure_ascii=False)
print(f'\nReport saved: {report_path}')
