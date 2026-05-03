"""Train Top-50 model for comparison (same method as Top-40)"""
import sys, os, json, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# Load cached data
with open('data/cache/factor_dataset.pkl', 'rb') as f:
    data = pickle.load(f)
names = data['factor_names']
meta = data['meta'].copy()
meta['dt'] = pd.to_datetime(meta['datetime'])
split_dt = meta['dt'].quantile(0.7)
train_idx = np.where((meta['dt'] < split_dt).values)[0]
test_idx = np.where((meta['dt'] >= split_dt).values)[0]
X_tr, y_tr = data['X'][train_idx], data['y'][train_idx]
X_te, y_te = data['X'][test_idx], data['y'][test_idx]

# Train baseline to get importance
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

imp = model.feature_importance(importance_type='gain')
feat_imp = list(zip(names, imp))
feat_imp.sort(key=lambda x: -x[1])

# Top-50 indices
name_to_idx = {n: i for i, n in enumerate(names)}
top50_names = [f for f, _ in feat_imp[:50]]
top50_idx = [name_to_idx[n] for n in top50_names]

X_tr_50 = X_tr[:, top50_idx]
X_te_50 = X_te[:, top50_idx]

print(f'Training with top-50 features...')
dtrain50 = lgb.Dataset(X_tr_50, y_tr)
dval50 = lgb.Dataset(X_te_50, y_te, reference=dtrain50)
final = lgb.train(params, dtrain50, num_boost_round=500,
                  valid_sets=[dval50], valid_names=['valid'],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

probs = final.predict(X_te_50)
auc = roc_auc_score(y_te, probs)
print(f'AUC: {auc:.4f}')

mte = meta.iloc[test_idx].copy()
mte['score'] = probs
mte['forward_ret'] = mte['forward_ret'].astype(float)

print(f'\n{"Threshold":>10} | {"Picks":>6} | {"Win%":>8} | {"AvgRet":>10}')
print('-'*48)
best_excess, best_th, best_picks = 0, 0.50, 0
for th in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    picks = mte[mte['score'] >= th]
    if len(picks) < 5: continue
    wr = (picks['forward_ret'] > 0).mean()
    ar = picks['forward_ret'].mean()
    print(f'{th:>10.2f} | {len(picks):>6} | {wr*100:>7.1f}% | {ar*100:>+9.2f}%')
    if ar > best_excess:
        best_excess, best_th, best_picks = ar, th, len(picks)

print(f'\n>> Best: excess={best_excess*100:+.2f}%  th={best_th:.2f}  picks={best_picks}')
print(f'>> AUC={auc:.4f}')
