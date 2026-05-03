"""Diagnose model feature alignment issue"""
import sys, os, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
import lightgbm as lgb
import xgboost as xgb

ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
os.chdir(ROOT)

with open('data/cache/factor_dataset.pkl', 'rb') as f:
    data = pickle.load(f)
with open('data/models/surge_lgbm.pkl', 'rb') as f:
    lgb_model = pickle.load(f)

names = data['factor_names']
model_feat_names = lgb_model.feature_name()
print(f'Original factor names count: {len(names)}')
print(f'Model expects {len(model_feat_names)} features')
print(f'First 5 model feature names: {model_feat_names[:5]}')
print(f'Last 5 model feature names: {model_feat_names[-5:]}')

# Check that model feature names exist in original names
name_to_idx = {n: i for i, n in enumerate(names)}
missing = [f for f in model_feat_names if f not in name_to_idx]
if missing:
    print(f'MISSING features: {missing}')
else:
    # Map model features to original indices
    feat_idx = [name_to_idx[f] for f in model_feat_names]
    print(f'Feature indices in X: {feat_idx[:5]}...')

# Score using CORRECT feature alignment
X_full = data['X']

# CORRECT: use model's feature names directly
X_correct = X_full[:, feat_idx]

# WRONG (what diag did): use importance-based re-sort
imp = lgb_model.feature_importance(importance_type='gain')
feat_imp = list(zip(names[:len(imp)], imp))  # THIS IS WRONG
feat_imp.sort(key=lambda x: -x[1])
wrong_names = [f for f, _ in feat_imp[:40]]
wrong_idx = [name_to_idx[f] for f in wrong_names]
X_wrong = X_full[:, wrong_idx]

# Compare
valid_mask = ~data['meta']['forward_ret'].isna().values
scores_correct = lgb_model.predict(X_correct[valid_mask])
scores_wrong = lgb_model.predict(X_wrong[valid_mask])

print(f'\nCorrect feature selection:')
print(f'  Min: {scores_correct.min():.4f}, Max: {scores_correct.max():.4f}')
print(f'  Mean: {scores_correct.mean():.4f}, Median: {np.median(scores_correct):.4f}')
print(f'  Above 0.30: {(scores_correct >= 0.30).sum()}')
print(f'  Above 0.40: {(scores_correct >= 0.40).sum()}')
print(f'  Above 0.50: {(scores_correct >= 0.50).sum()}')

print(f'\nWrong feature selection:')
print(f'  Min: {scores_wrong.min():.4f}, Max: {scores_wrong.max():.4f}')
print(f'  Mean: {scores_wrong.mean():.4f}, Median: {np.median(scores_wrong):.4f}')

# Check if model was trained with original or recomputed labels
y_original = data['y'][valid_mask]
# Recompute labels for 20d
meta = data['meta'].copy()
meta_v = meta[valid_mask].copy()
meta_v['datetime'] = pd.to_datetime(meta_v['datetime'])
split_dt = meta_v['datetime'].quantile(0.7)
test_mask = meta_v['datetime'] >= split_dt
y_test = y_original[test_mask.values]
scores_test = scores_correct[test_mask.values]
print(f'\nTest set ({test_mask.sum()} samples):')
print(f'  Score range: {scores_test.min():.4f} ~ {scores_test.max():.4f}')
print(f'  Original label surge rate: {y_test.mean():.1%}')
print(f'  Scores > 0.40: {(scores_test >= 0.40).sum()} samples')
print(f'  Max score: {scores_test.max():.4f}')

# Now load XGB and check ensemble
xgb_model = xgb.Booster(model_file='data/models/surge_xgboost.json')
xgb_feat_names = xgb_model.feature_names
print(f'\nXGBoost feature count: {len(xgb_feat_names)}')
print(f'XGB first 5 features: {xgb_feat_names[:5]}')
print(f'Match LGB features: {xgb_feat_names == lgb_model.feature_name()}')
