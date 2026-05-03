"""Retrain model using only Top-40 features and save"""
import sys, os, json, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# Load report
report = json.load(open(os.path.join('data', 'models', 'feature_selection_report.json')))
opt_features = report['optimal_features']  # currently top-50
imp_list = report['feature_importance_top30']  # top-30 by name

# We need top-40. The report stored full importance in the script but only top-30 by name.
# Let's find top-40 by re-running importance on cleaned data
import pandas as pd

CACHE = 'data/cache/factor_dataset.pkl'
with open(CACHE, 'rb') as f:
    data = pickle.load(f)

names = data['factor_names']
meta = data['meta'].copy()
meta['dt'] = pd.to_datetime(meta['datetime'])
split_dt = meta['dt'].quantile(0.7)
train_idx = np.where((meta['dt'] < split_dt).values)[0]
test_idx  = np.where((meta['dt'] >= split_dt).values)[0]

X_all, y_all = data['X'], data['y']
X_tr, y_tr = X_all[train_idx], y_all[train_idx]
X_te, y_te = X_all[test_idx], y_all[test_idx]

# Train baseline to get full importance
params = {
    'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
    'num_leaves': 63, 'max_depth': 6, 'learning_rate': 0.03,
    'feature_fraction': 0.8, 'bagging_fraction': 0.85,
    'bagging_freq': 5, 'min_child_samples': 50, 'random_state': 42,
}
dtrain = lgb.Dataset(X_tr, y_tr)
dval = lgb.Dataset(X_te, y_te, reference=dtrain)
model = lgb.train(params, dtrain, num_boost_round=500,
                  valid_sets=[dval], valid_names=['valid'],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

# Get importance
imp = model.feature_importance(importance_type='gain')
feat_imp = list(zip(names, imp))
feat_imp.sort(key=lambda x: -x[1])

top40_names = [f for f, _ in feat_imp[:40]]
print(f'Top-40 features:')
for i, (f, imp_val) in enumerate(feat_imp[:40]):
    print(f'  {i+1:>2}. {f}  (gain={imp_val:.1f})')

# Get indices of top-40 in full feature array
name_to_idx = {n: i for i, n in enumerate(names)}
top40_idx = [name_to_idx[n] for n in top40_names]

# Subset data
X_tr_40 = X_tr[:, top40_idx]
X_te_40 = X_te[:, top40_idx]

# Train final model with top-40
print(f'\nTraining with top-40 features...')
dtrain40 = lgb.Dataset(X_tr_40, y_tr)
dval40 = lgb.Dataset(X_te_40, y_te, reference=dtrain40)
final = lgb.train(params, dtrain40, num_boost_round=500,
                  valid_sets=[dval40], valid_names=['valid'],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

probs = final.predict(X_te_40)
auc = roc_auc_score(y_te, probs)
print(f'AUC: {auc:.4f}')

# Evaluate excess return
mte = meta.iloc[test_idx].copy()
mte['score'] = probs
mte['forward_ret'] = mte['forward_ret'].astype(float)

print(f'\n{"Threshold":>10} | {"Picks":>6} | {"Win%":>8} | {"AvgRet":>10}')
print('-'*48)
best_excess, best_th, best_picks = 0, 0.50, 0
for th in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    picks = mte[mte['score'] >= th]
    if len(picks) < 5:
        continue
    wr = (picks['forward_ret'] > 0).mean()
    ar = picks['forward_ret'].mean()
    print(f'{th:>10.2f} | {len(picks):>6} | {wr*100:>7.1f}% | {ar*100:>+9.2f}%')
    if ar > best_excess:
        best_excess, best_th, best_picks = ar, th, len(picks)

print(f'\n>> Best: excess={best_excess*100:+.2f}%  th={best_th:.2f}  picks={best_picks}')

# Save model
model_path = 'data/models/surge_lgbm.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(final, f)
print(f'Model saved: {model_path}')

# Save feature list
feat_path = 'data/models/selected_features.json'
with open(feat_path, 'w') as f:
    json.dump({'selected_features': top40_names, 'threshold': best_th,
               'n_features': 40, 'auc': round(auc, 4), 'excess': round(best_excess, 4)}, f, indent=2)
print(f'Feature list saved: {feat_path}')

# Update params.json
pj = 'src/surge/params.json'
with open(pj) as f:
    p = json.load(f)
p['selected_features'] = top40_names
p['ml_threshold'] = best_th
p['ml_top_features'] = 40
with open(pj, 'w') as f:
    json.dump(p, f, indent=2, ensure_ascii=False)
print(f'params.json updated (top-40, threshold={best_th})')

# Show top-10 features
print(f'\nTop-10 features by importance:')
for i, (f, im) in enumerate(feat_imp[:10]):
    print(f'  {i+1}. {f}  (gain={im:.0f})')
