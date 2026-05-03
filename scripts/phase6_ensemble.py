"""Phase 6: XGBoost + Ensemble with LightGBM
Load cached data -> train XGBoost -> compare with LightGBM -> ensemble voting -> save"""
import sys, os, json, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import lightgbm as lgb
import xgboost as xgb
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

X_full = data['X']
meta = data['meta'].copy()
names = data['factor_names']
print(f'  Factors: {len(names)}, Samples: {len(meta)}')

# ── 2. Top-40 features ──────────────────────────────────────────────────
meta['dt'] = pd.to_datetime(meta['datetime'])
split_dt = meta['dt'].quantile(0.7)
train_idx = np.where((meta['dt'] < split_dt).values)[0]
test_idx  = np.where((meta['dt'] >= split_dt).values)[0]
y_all_original = data['y']

params_lgb = {
    'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
    'num_leaves': 63, 'max_depth': 6, 'learning_rate': 0.03,
    'feature_fraction': 0.8, 'bagging_fraction': 0.85,
    'bagging_freq': 5, 'min_child_samples': 50, 'random_state': 42,
}
dtrain = lgb.Dataset(X_full[train_idx], y_all_original[train_idx])
dval = lgb.Dataset(X_full[test_idx], y_all_original[test_idx], reference=dtrain)
model_lgb = lgb.train(params_lgb, dtrain, num_boost_round=500,
                      valid_sets=[dval], valid_names=['valid'],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
imp = model_lgb.feature_importance(importance_type='gain')
feat_imp = list(zip(names, imp))
feat_imp.sort(key=lambda x: -x[1])
top40_names = [f for f, _ in feat_imp[:40]]
name_to_idx = {n: i for i, n in enumerate(names)}
top40_idx = [name_to_idx[n] for n in top40_names]
X_top40 = X_full[:, top40_idx]
print(f'  Top-40 features selected')

# ── 3. Build 20d label ──────────────────────────────────────────────────
print('\nComputing 20-day forward returns...')
stock_lookup = {}
for ts_code, df in prices.items():
    df_sorted = df.sort_values('date').reset_index(drop=True)
    stock_lookup[ts_code] = {
        'dates': df_sorted['date'].astype(str).values,
        'closes': df_sorted['close'].values.astype(np.float64),
    }
fwd_20d = np.full(len(meta), np.nan)
for idx, row in meta.iterrows():
    sd = stock_lookup.get(row['vt_symbol'])
    if sd is None: continue
    date_str = str(row['datetime'])[:10]
    pos = np.where(sd['dates'] == date_str)[0]
    if len(pos) == 0: continue
    j = pos[0] + 20
    if j >= len(sd['closes']): continue
    fwd_20d[idx] = (sd['closes'][j] - sd['closes'][pos[0]]) / sd['closes'][pos[0]]
valid = ~np.isnan(fwd_20d)
print(f'  Valid: {valid.sum()}/{len(valid)}')

y_binary = (fwd_20d >= 0.10).astype(np.int32)
sr = y_binary.mean()
print(f'  Surge rate: {sr:.2%}')

# ── 4. Train XGBoost ────────────────────────────────────────────────────
print('\n── Training XGBoost ──')
train_mask_bool = np.zeros(valid.shape, dtype=bool)
train_mask_bool[train_idx] = True
test_mask_bool = np.zeros(valid.shape, dtype=bool)
test_mask_bool[test_idx] = True

X_tr = X_top40[train_mask_bool & valid]
X_te = X_top40[test_mask_bool & valid]
y_tr = y_binary[train_mask_bool & valid]
y_te = y_binary[test_mask_bool & valid]
mte_idx = np.where(test_mask_bool & valid)[0]
print(f'  Train: {len(y_tr)}, Test: {len(y_te)}, Surge train={y_tr.mean():.1%}, test={y_te.mean():.1%}')

params_xgb = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.85,
    'colsample_bytree': 0.8, 'min_child_weight': 50,
    'random_state': 42, 'verbosity': 0, 'n_jobs': 4,
}
dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr)
dval_xgb = xgb.DMatrix(X_te, label=y_te)
model_xgb = xgb.train(
    params_xgb, dtrain_xgb, num_boost_round=500,
    evals=[(dval_xgb, 'valid')],
    early_stopping_rounds=50, verbose_eval=0,
)
probs_xgb = model_xgb.predict(dval_xgb)
auc_xgb = roc_auc_score(y_te, probs_xgb)

# ── 5. Train LightGBM (same data, for fair comparison) ──────────────────
print('\n── Training LightGBM (same split) ──')
dtrain_lgb = lgb.Dataset(X_tr, y_tr)
dval_lgb = lgb.Dataset(X_te, y_te, reference=dtrain_lgb)
model_lgb2 = lgb.train(params_lgb, dtrain_lgb, num_boost_round=500,
                       valid_sets=[dval_lgb], valid_names=['valid'],
                       callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
probs_lgb = model_lgb2.predict(X_te)
auc_lgb = roc_auc_score(y_te, probs_lgb)

# ── 6. Ensemble: average probabilities ──────────────────────────────────
probs_avg = (probs_lgb + probs_xgb) / 2.0
auc_avg = roc_auc_score(y_te, probs_avg)

# ── 7. Evaluate all three ───────────────────────────────────────────────
def evaluate(name, probs, y_true, fwd_rets):
    meth = {ts_code: stops for ts_code, stops in meta.iloc[mte_idx]['vt_symbol'].items()}  # not used
    mte = meta.iloc[mte_idx].copy()
    mte['score'] = probs
    mte['fwd_ret'] = fwd_rets[mte_idx]
    auc_v = roc_auc_score(y_true, probs)
    
    print(f'\n  {name}:')
    print(f'    AUC: {auc_v:.4f}')
    print(f'    {"Thresh":>7} | {"Picks":>6} | {"Win%":>7} | {"AvgRet":>10}')
    print(f'    ' + '-'*40)
    
    best_excess, best_th, best_picks = 0, 0.50, 0
    for th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        picks = mte[mte['score'] >= th]
        if len(picks) < 5: continue
        wr = (picks['fwd_ret'] > 0).mean()
        ar = picks['fwd_ret'].mean()
        print(f'    {th:>7.2f} | {len(picks):>6} | {wr*100:>6.1f}% | {ar*100:>+9.2f}%')
        if ar > best_excess:
            best_excess, best_th, best_picks = ar, th, len(picks)
    
    return {'auc': round(auc_v, 4), 'best_excess': round(best_excess, 4), 'best_th': best_th, 'picks': best_picks}

print('\n' + '='*70)
print('MODEL COMPARISON')
print('='*70)

r_lgb = evaluate('LightGBM', probs_lgb, y_te, fwd_20d)
r_xgb = evaluate('XGBoost', probs_xgb, y_te, fwd_20d)
r_avg = evaluate('Ensemble(AVG)', probs_avg, y_te, fwd_20d)

# ── 8. Final table ──────────────────────────────────────────────────────
print('\n')
print('='*70)
print('FINAL COMPARISON')
print('='*70)
print(f'{"Model":>16} | {"AUC":>6} | {"Excess":>8} | {"Picks":>6} | {"Threshold"}>')
print('-'*55)
models = [('LightGBM', r_lgb), ('XGBoost', r_xgb), ('Ensemble(AVG)', r_avg)]
best_ex, best_name = 0, ''
for name, r in models:
    print(f'  {name:>14} | {r["auc"]:>6.4f} | {r["best_excess"]*100:>+7.2f}% | {r["picks"]:>5}  | {r["best_th"]:.2f}')
    if r['best_excess'] > best_ex:
        best_ex, best_name = r['best_excess'], name

print(f'\n>> Best: {best_name} (excess={best_ex*100:+.2f}%)')

# ── 9. Save all ─────────────────────────────────────────────────────────
# Save XGBoost model
model_xgb_path = os.path.join(ROOT, 'data', 'models', 'surge_xgboost.json')
model_xgb.save_model(model_xgb_path)
print(f'\nXGBoost model saved: {model_xgb_path}')

# Save LightGBM2 (trained on same split)
model_lgb2_path = os.path.join(ROOT, 'data', 'models', 'surge_lgbm.pkl')
with open(model_lgb2_path, 'wb') as f:
    pickle.dump(model_lgb2, f)
print(f'LightGBM model saved: {model_lgb2_path}')

# Save report
report = {
    'lightgbm': r_lgb,
    'xgboost': r_xgb,
    'ensemble_avg': r_avg,
    'best_model': best_name,
    'ensemble_method': 'avg',
}
report_path = os.path.join(ROOT, 'data', 'models', 'ensemble_report.json')
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f'Report saved: {report_path}')

# Update params.json with ensemble config
pj = 'src/surge/params.json'
with open(pj) as f:
    p = json.load(f)
p['ensemble'] = {
    'enabled': True,
    'method': 'avg',
    'models': ['surge_lgbm.pkl', 'surge_xgboost.json'],
    'threshold': r_avg['best_th'],
}
p['ml_threshold'] = r_avg['best_th']
with open(pj, 'w') as f:
    json.dump(p, f, indent=2, ensure_ascii=False)
print(f'params.json updated (ensemble, threshold={r_avg["best_th"]})')
