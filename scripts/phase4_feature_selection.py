"""Phase 4: Feature Selection
Pipeline: importance rank -> correlation filter -> forward selection -> compare"""
import sys, os, pickle, json, warnings, gc
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from copy import deepcopy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── 1. Load cached dataset ──────────────────────────────────────────────
CACHE = os.path.join(ROOT, 'data', 'cache', 'factor_dataset.pkl')
print('Loading cached dataset...')
with open(CACHE, 'rb') as f:
    data = pickle.load(f)
print(f'  Samples: {len(data["X"])}, Factors: {len(data["factor_names"])}')

X_all = data['X']         # (n_samples, n_features)
y_all = data['y']         # (n_samples,)
meta = data['meta'].copy()
names = data['factor_names']
n_features = len(names)

# ── 2. Time split ────────────────────────────────────────────────────────
meta['dt'] = pd.to_datetime(meta['datetime'])
split_dt = meta['dt'].quantile(0.7)
train_idx = np.where((meta['dt'] < split_dt).values)[0]
test_idx  = np.where((meta['dt'] >= split_dt).values)[0]
print(f'  Train: {len(train_idx)}, Test: {len(test_idx)}')

X_tr, y_tr = X_all[train_idx], y_all[train_idx]
X_te, y_te = X_all[test_idx], y_all[test_idx]

# ── 3. Baseline ──────────────────────────────────────────────────────────
BASE_PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
    'num_leaves': 63, 'max_depth': 6, 'learning_rate': 0.03,
    'feature_fraction': 0.8, 'bagging_fraction': 0.85,
    'bagging_freq': 5, 'min_child_samples': 50, 'random_state': 42,
}
def evaluate(X_tr, y_tr, X_te, y_te, feature_names, label=''):
    dtrain = lgb.Dataset(X_tr, y_tr)
    dval = lgb.Dataset(X_te, y_te, reference=dtrain)
    model = lgb.train(BASE_PARAMS, dtrain, num_boost_round=500,
                      valid_sets=[dval], valid_names=['valid'],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    auc = roc_auc_score(y_te, model.predict(X_te))
    # Excess return at best threshold
    probs = model.predict(X_te)
    mte = meta.iloc[test_idx].copy()
    mte['score'] = probs
    mte['forward_ret'] = mte['forward_ret'].astype(float)
    best_excess, best_th, best_picks = 0, 0.5, 0
    for th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        picks = mte[mte['score'] >= th]
        if len(picks) < 5: continue
        ar = picks['forward_ret'].mean()
        if ar > best_excess:
            best_excess, best_th, best_picks = ar, th, len(picks)
    imp = model.feature_importance(importance_type='gain')
    feat_imp = list(zip(feature_names, imp))
    feat_imp.sort(key=lambda x: -x[1])
    print(f'  [{label:>12}] AUC={auc:.4f}  Best: excess={best_excess*100:+7.2f}%  th={best_th:.2f}  picks={best_picks}  features={len(feature_names)}')
    return model, auc, best_excess, best_th, best_picks, probs, feat_imp

print('\n── Baseline: 160 factors ──')
base_model, base_auc, base_excess, base_th, base_picks, base_probs, base_imp = evaluate(X_tr, y_tr, X_te, y_te, names, '160 factors')
gc.collect()

# ── 4. Remove near-zero variance ─────────────────────────────────────────
print('\n── Step 1: Near-zero variance filter ──')
X_tr_df = pd.DataFrame(X_tr)
variances = X_tr_df.var()
nzv_cols = set(np.where(variances < 1e-6)[0])
print(f'  Removed (var<1e-6): {len(nzv_cols)}')

# ── 5. Remove highly correlated ──────────────────────────────────────────
print('\n── Step 2: Correlation filter ──')
corr = X_tr_df.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr = set(np.where(upper > 0.95)[0])
to_drop = nzv_cols | high_corr
print(f'  Removed (corr>0.95): {len(high_corr)}')
print(f'  Total removed: {len(to_drop)}, remaining: {n_features - len(to_drop)}')

keep = sorted(set(range(n_features)) - to_drop)
keep_names = [names[i] for i in keep]
print(f'  Remaining features: {len(keep_names)}')
X_tr_f = X_tr[:, keep]
X_te_f = X_te[:, keep]

gc.collect()

# ── 6. Train baseline after cleaning ────────────────────────────────────
print('\n── After cleaning ( {} features ) ──'.format(len(keep_names)))
clean_model, clean_auc, clean_excess, clean_th, clean_picks, clean_probs, clean_imp = evaluate(X_tr_f, y_tr, X_te_f, y_te, keep_names, 'cleaned')
gc.collect()

# ── 7. Forward selection: test different feature counts ──────────────────
print('\n── Step 3: Forward selection (varying feature counts) ──')
# Use importance from cleaned model
sorted_imp = [n for n, _ in sorted(clean_imp, key=lambda x: -x[1])]
sorted_idx = [keep_names.index(n) for n in sorted_imp]
trial_sizes = [10, 20, 30, 40, 50, 60, 80, 100]
results = []
for n in trial_sizes:
    n_actual = min(n, len(keep_names))
    idx = sorted_idx[:n_actual]
    X_tr_n, X_te_n = X_tr_f[:, idx], X_te_f[:, idx]
    fn = [keep_names[i] for i in idx]
    _, auc_n, ex_n, th_n, pk_n, _, _ = evaluate(X_tr_n, y_tr, X_te_n, y_te, fn, f'top-{n_actual}')
    results.append({'n_features': n_actual, 'auc': round(auc_n, 4), 'excess': round(ex_n, 4), 'threshold': th_n, 'picks': pk_n})
    gc.collect()

# ── 8. Results summary ────────────────────────────────────────────────────
print('\n' + '='*70)
print('FEATURE SELECTION RESULTS')
print('='*70)
print(f'{"Model":>20} | {"AUC":>6} | {"Excess":>8} | {"Win%":>6} | {"Picks":>6} | {"Features"}')
print('-'*70)
print(f'{"Baseline(160)":>20} | {base_auc:>6.4f} | {base_excess*100:>+7.2f}% | {f"{(base_excess>0)}":>6} | {base_picks:>6} | 160')
print(f'{"Cleaned("+str(len(keep_names))+")":>20} | {clean_auc:>6.4f} | {clean_excess*100:>+7.2f}% | | {clean_picks:>6} | {len(keep_names)}')
for r in results:
    print(f'{"Top-"+str(r["n_features"]):>20} | {r["auc"]:>6.4f} | {r["excess"]*100:>+7.2f}% | | {r["picks"]:>6} | {r["n_features"]}')

# ── 9. Determine optimal count ───────────────────────────────────────────
best_r = max(results, key=lambda r: r['excess'] if r['auc'] >= base_auc else r['auc'])
print(f'\n>> Optimal: top-{best_r["n_features"]} features (AUC={best_r["auc"]}, excess={best_r["excess"]*100:+.2f}%)')

# ── 10. Train final model with optimal feature set ───────────────────────
opt_n = best_r['n_features']
opt_idx = sorted_idx[:opt_n]
opt_names = [keep_names[i] for i in opt_idx]
X_tr_opt, X_te_opt = X_tr_f[:, opt_idx], X_te_f[:, opt_idx]
final_model, final_auc, final_excess, final_th, final_picks, final_probs, final_imp = evaluate(X_tr_opt, y_tr, X_te_opt, y_te, opt_names, f'final-{opt_n}')

# ── 11. Save results ──────────────────────────────────────────────────────
report = {
    'n_total': n_features,
    'n_removed_nzv': int(len(nzv_cols)),
    'n_removed_corr': int(len(high_corr)),
    'n_cleaned': int(len(keep_names)),
    'baseline': {'auc': round(base_auc, 4), 'excess': round(base_excess, 4), 'threshold': base_th, 'picks': base_picks},
    'cleaned': {'auc': round(clean_auc, 4), 'excess': round(clean_excess, 4), 'threshold': clean_th, 'picks': clean_picks},
    'trials': results,
    'optimal': {'n_features': opt_n, 'auc': round(final_auc, 4), 'excess': round(final_excess, 4), 'threshold': final_th, 'picks': final_picks},
    'optimal_features': opt_names,
    'feature_importance_top30': sorted_imp[:30],
}
out_path = os.path.join(ROOT, 'data', 'models', 'feature_selection_report.json')
with open(out_path, 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f'\nReport saved: {out_path}')

# ── 12. Save model ────────────────────────────────────────────────────────
model_path = os.path.join(ROOT, 'data', 'models', 'surge_lgbm.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)
print(f'Model saved: {model_path}')

# Save feature list for pipeline
feat_path = os.path.join(ROOT, 'data', 'models', 'selected_features.json')
with open(feat_path, 'w') as f:
    json.dump({'selected_features': opt_names, 'threshold': final_th, 'n_features': opt_n}, f, indent=2)
print(f'Feature list saved: {feat_path}')

# ── 13. Update pipeline config ────────────────────────────────────────────
pj = os.path.join(ROOT, 'src', 'surge', 'params.json')
with open(pj) as f:
    params = json.load(f)
params['selected_features'] = opt_names
params['ml_threshold'] = final_th
with open(pj, 'w') as f:
    json.dump(params, f, indent=2, ensure_ascii=False)
print(f'params.json updated with feature list ({opt_n} features, threshold={final_th})')

print('\nDone!')
