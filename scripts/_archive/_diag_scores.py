"""Quick diagnostic: check score distribution across all samples"""
import sys, os, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
os.chdir(ROOT)

with open('data/cache/factor_dataset.pkl', 'rb') as f:
    data = pickle.load(f)
meta = data['meta'].copy()
names = data['factor_names']

# Get valid samples
valid_mask = ~meta['forward_ret'].isna().values
meta_valid = meta[valid_mask].copy()
meta_valid['datetime'] = pd.to_datetime(meta_valid['datetime'])

print(f'Sample count: {len(meta_valid)}')
print(f'Date range: {meta_valid["datetime"].min()} ~ {meta_valid["datetime"].max()}')
print(f'Year distribution:')
print(meta_valid['datetime'].dt.year.value_counts().sort_index())

# Load models & score
import lightgbm as lgb
import xgboost as xgb
X_full = data['X'][valid_mask]

with open('data/models/surge_lgbm.pkl', 'rb') as f:
    lgb_model = pickle.load(f)
xgb_model = xgb.Booster(model_file='data/models/surge_xgboost.json')

imp = lgb_model.feature_importance(importance_type='gain')
feat_imp = list(zip(names, imp))
feat_imp.sort(key=lambda x: -x[1])
top40_names = [f for f, _ in feat_imp[:40]]
name_to_idx = {n: i for i, n in enumerate(names)}
top40_idx = np.array([name_to_idx[n] for n in top40_names])
X_top40 = X_full[:, top40_idx]

probs_lgb = lgb_model.predict(X_top40)
probs_xgb = xgb_model.predict(xgb.DMatrix(X_top40))
scores = (probs_lgb + probs_xgb) / 2.0

print(f'\nScore statistics:')
print(f'  Min: {scores.min():.4f}')
print(f'  Max: {scores.max():.4f}')
print(f'  Mean: {scores.mean():.4f}')
print(f'  Median: {np.median(scores):.4f}')
print(f'  Std: {scores.std():.4f}')
print(f'\nPercentiles:')
for p in [10, 25, 50, 75, 90, 95, 99, 99.9]:
    print(f'  P{p}: {np.percentile(scores, p):.4f}')
print(f'\nAbove thresholds:')
for th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    print(f'  >= {th:.2f}: {(scores >= th).sum()} samples ({(scores >= th).mean()*100:.2f}%)')
