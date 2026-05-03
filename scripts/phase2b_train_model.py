"""Train model with balanced params (#2 from grid search)"""
import sys, os, pickle, json, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Load cached dataset
with open(os.path.join(ROOT, 'data', 'cache', 'factor_dataset.pkl'), 'rb') as f:
    data = pickle.load(f)
print(f'Loaded: {len(data["X"])} samples, {len(data["factor_names"])} factors')

# Time split
meta = data['meta'].copy()
meta['dt'] = pd.to_datetime(meta['datetime'])
split_dt = meta['dt'].quantile(0.7)
train_mask = (meta['dt'] < split_dt).values
X_tr, y_tr = data['X'][train_mask], data['y'][train_mask]
X_te, y_te = data['X'][~train_mask], data['y'][~train_mask]
print(f'Split: {len(X_tr)} train, {len(X_te)} test')

# Train (#2 from grid: lr=0.03, nl=63, ff=0.8, bf=0.85, depth=6)
params = {
    'objective': 'binary', 'metric': 'auc',
    'num_leaves': 63, 'max_depth': 6, 'learning_rate': 0.03,
    'feature_fraction': 0.8, 'bagging_fraction': 0.85,
    'bagging_freq': 5, 'min_child_samples': 50,
    'random_state': 42, 'verbosity': -1,
}
dtrain = lgb.Dataset(X_tr, y_tr)
dval = lgb.Dataset(X_te, y_te, reference=dtrain)
model = lgb.train(params, dtrain, num_boost_round=800,
                  valid_sets=[dval], valid_names=['valid'],
                  callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)])

# Evaluate
probs = model.predict(X_te)
auc = roc_auc_score(y_te, probs)
print(f'\nAUC: {auc:.4f}')
mte = meta.iloc[~train_mask].copy()
mte['score'] = probs
mte['forward_ret'] = mte['forward_ret'].astype(float)

print(f'{"Threshold":>10} | {"Picks":>6} | {"Win%":>6} | {"AvgRet":>8}')
print(f'{"-"*40}')
for th in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
    picks = mte[mte['score'] >= th]
    if len(picks) < 5:
        continue
    wr = (picks['forward_ret'] > 0).mean()
    ar = picks['forward_ret'].mean()
    print(f'{th:>10.2f} | {len(picks):>6} | {wr*100:>6.1f}% | {ar*100:>+7.2f}%')

# Save model
path = os.path.join(ROOT, 'data', 'models', 'surge_lgbm.pkl')
with open(path, 'wb') as f:
    pickle.dump(model, f)
print(f'\nModel saved: {path}')

# Update params.json
pj = os.path.join(ROOT, 'src', 'surge', 'params.json')
with open(pj) as f:
    p = json.load(f)
p['lgbm_params'] = params
with open(pj, 'w') as f:
    json.dump(p, f, indent=2)
print(f'params.json updated with lr=0.03, nl=63')
