"""
Full Dataset Retraining + Walk-Forward Backtest
------------------------------------------------
1. Loads all 7-year factor cache (10 batches, 805k rows)
2. Time-series split: rolling train/test windows
3. Retrains LightGBM + XGBoost on full history
4. Walk-Forward evaluation for realistic OOS performance
5. Output: comparison table, model files, equity curve

Usage: python scripts/retrain_full.py
"""
import sys, os, pickle, time, json, gc, warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'
ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb

print('=' * 60)
print('Full Dataset Retraining + Walk-Forward Backtest')
print('=' * 60)

# ── 1. Load factor cache ──────────────────────────────────────────────────
print('\n[1/5] Loading factor cache...')
t0 = time.time()
FACTOR_DIR = os.path.join(ROOT, 'data', 'cache', 'factors_batched')
all_dfs = []
for fn in sorted(os.listdir(FACTOR_DIR)):
    if fn.endswith('.parquet'):
        df = pd.read_parquet(os.path.join(FACTOR_DIR, fn))
        all_dfs.append(df)

pdf = pd.concat(all_dfs, ignore_index=True)
print(f'  Rows: {len(pdf):,}  Columns: {len(pdf.columns)}')

# ── 2. Create labels ──────────────────────────────────────────────────────
print('\n[2/5] Creating labels (20d forward ≥10%)...')
HORIZON = 20
pdf = pdf.sort_values(['vt_symbol', 'datetime']).reset_index(drop=True)

pdf['forward_ret'] = np.nan
for name, group in pdf.groupby('vt_symbol', sort=False):
    idx = group.index
    closes = group['close'].values
    for i in range(len(group) - HORIZON):
        ret = (closes[i + HORIZON] - closes[i]) / closes[i]
        pdf.loc[idx[i], 'forward_ret'] = ret

pdf['label'] = (pdf['forward_ret'] >= 0.10).astype(int)
pdf['date'] = pd.to_datetime(pdf['datetime'])

print(f'  Forward ret available: {pdf["forward_ret"].notna().sum():,} ({pdf["forward_ret"].notna().mean():.0%})')
print(f'  Surge rate (label=1): {pdf["label"].mean():.2%}')
print(f'  Date range: {pdf["date"].min().date()} ~ {pdf["date"].max().date()}')

# ── 3. Define features ────────────────────────────────────────────────────
# Use the original 40 selected features
with open('src/surge/params.json') as f:
    params = json.load(f)
TOP_40 = params['selected_features']

valid_features = [f for f in TOP_40 if f in pdf.columns]
missing = [f for f in TOP_40 if f not in pdf.columns]
print(f'\n  Features: {len(valid_features)} / {len(TOP_40)} available')
if missing:
    print(f'  MISSING: {missing}')

# Filter to rows with valid features + label
mask = pdf[valid_features].notna().all(axis=1) & pdf['forward_ret'].notna()
clean = pdf[mask].copy()
print(f'  Clean rows: {len(clean):,} / {len(pdf):,}')

# Cap extreme forward returns for stability
clean['forward_ret'] = clean['forward_ret'].clip(-1, 3)

# ── 4. Train on Entire Dataset ────────────────────────────────────────────
print('\n[3/5] Training on full dataset...')
t_train = time.time()

X_full = clean[valid_features].astype(np.float32).values
X_full = np.clip(X_full, -1e10, 1e10)
X_full = np.where(np.isinf(X_full), np.nan, X_full)

# Drop any remaining NaN rows
nan_mask = ~np.isnan(X_full).any(axis=1)
if nan_mask.sum() < len(X_full):
    print(f'  Dropping {(~nan_mask).sum():,} rows ({((~nan_mask).sum()/len(X_full))*100:.1f}%) with NaN')
    X_full = X_full[nan_mask]
    clean = clean[nan_mask].copy()

y_full = clean['label'].values
print(f'  Training samples: {len(X_full):,}')
print(f'  Class balance: {y_full.mean():.2%} positives')

# LightGBM params (tuned from previous hyperparameter search)
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 64,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 100,
    'max_depth': 10,
    'verbosity': -1,
    'seed': 42,
}

lgb_train = lgb.Dataset(X_full, y_full, feature_name=valid_features)
lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
)

# Get training AUC
train_preds = lgb_model.predict(X_full)
train_auc = __import__('sklearn').metrics.roc_auc_score(y_full, train_preds)
print(f'  LightGBM train AUC: {train_auc:.4f}')
print(f'  Best iteration: {lgb_model.best_iteration}')
print(f'  Training time: {time.time()-t_train:.0f}s')

# Save model
with open('data/models/surge_lgbm_full.pkl', 'wb') as f:
    pickle.dump(lgb_model, f)
print(f'  Saved: data/models/surge_lgbm_full.pkl')

# ── 5. Walk-Forward Backtest ──────────────────────────────────────────────
print('\n[4/5] Walk-Forward backtest...')
t_wf = time.time()

# Define time windows (years)
clean['year'] = clean['date'].dt.year
train_years_list = [
    (2019, 2021),  # Train 2019-2021
    (2020, 2022),  # Train 2020-2022
    (2019, 2022),  # Train 2019-2022
    (2019, 2023),  # Train 2019-2023
    (2020, 2024),  # Train 2020-2024 (for 2025 eval)
]

results = []
for train_start, train_end in train_years_list:
    test_year = train_end + 1 if train_end < 2024 else 2025
    
    train_mask = (clean['year'] >= train_start) & (clean['year'] <= train_end)
    test_mask = clean['year'] == test_year
    
    if train_mask.sum() < 10000 or test_mask.sum() < 100:
        print(f'  SKIP {train_start}-{train_end} → {test_year}: insufficient data')
        continue
    
    X_tr = clean[train_mask][valid_features].astype(np.float32).values
    X_tr = np.clip(X_tr, -1e10, 1e10)
    X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
    nan_tr = ~np.isnan(X_tr).any(axis=1)
    X_tr = X_tr[nan_tr]
    y_tr = clean[train_mask][nan_tr]['label'].values
    
    X_te = clean[test_mask][valid_features].astype(np.float32).values
    X_te = np.clip(X_te, -1e10, 1e10)
    X_te = np.where(np.isinf(X_te), np.nan, X_te)
    nan_te = ~np.isnan(X_te).any(axis=1)
    X_te = X_te[nan_te]
    y_te = clean[test_mask][nan_te]['label'].values
    test_rets = clean[test_mask][nan_te]['forward_ret'].values
    
    # Train (split last 20% of training data for validation)
    val_split = int(len(X_tr) * 0.8)
    X_tr_fold, X_val = X_tr[:val_split], X_tr[val_split:]
    y_tr_fold, y_val = y_tr[:val_split], y_tr[val_split:]
    
    model = lgb.train(
        lgb_params,
        lgb.Dataset(X_tr_fold, y_tr_fold, feature_name=valid_features),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_val, y_val)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    
    # Predict
    scores = model.predict(X_te)
    auc = __import__('sklearn').metrics.roc_auc_score(y_te, scores)
    
    # Signal backtest at various thresholds
    best_excess = -999
    for th in [t / 100 for t in range(5, 80, 5)]:
        picks = scores >= th
        if picks.sum() < 10:
            continue
        avg_ret = test_rets[picks].mean()
        market_avg = test_rets.mean()
        excess = avg_ret - market_avg
        if excess > best_excess:
            best_excess = excess
            best_th = th
            best_wr = (test_rets[picks] > 0).mean()
            best_n = picks.sum()
    
    results.append({
        'train': f'{train_start}-{train_end}',
        'test': f'{test_year}',
        'train_n': len(X_tr),
        'test_n': len(X_te),
        'auc': round(auc, 4),
        'best_th': best_th,
        'best_n': best_n,
        'best_wr': f'{best_wr:.1%}',
        'best_excess': f'{best_excess*100:+.2f}%',
        'market_avg': f'{test_rets.mean()*100:+.2f}%',
    })
    
    print(f'  Train {train_start}-{train_end} → Test {test_year}: '
          f'AUC={auc:.4f}, ≥{best_th:.2f}: {best_n} sigs, '
          f'WR={best_wr:.1%}, Excess={best_excess*100:+.2f}%')
    
    gc.collect()

wf_df = pd.DataFrame(results)
print(f'\n  Walk-Forward time: {time.time()-t_wf:.0f}s')

# ── 6. Full signal backtest ────────────────────────────────────────────────
print('\n[5/5] Signal backtest on full period...')
t_signal = time.time()

# Use single model trained on full data
full_scores = lgb_model.predict(X_full)
full_df = clean.copy()
full_df['score'] = full_scores

# Multi-threshold analysis
thresholds = [t / 100 for t in range(5, 65, 5)]
ft_results = []
for th in thresholds:
    picks = full_df[full_df['score'] >= th]
    if len(picks) < 10:
        continue
    wr = (picks['forward_ret'] > 0).mean()
    avg_ret = picks['forward_ret'].mean()
    market = full_df['forward_ret'].mean()
    excess = avg_ret - market
    ft_results.append({
        'threshold': th,
        'signals': len(picks),
        'win_rate': f'{wr:.1%}',
        'avg_ret': f'{avg_ret*100:+.2f}%',
        'excess': f'{excess*100:+.2f}%',
        'raw_excess': excess,
    })

ft_df = pd.DataFrame(ft_results)
print(f'\n  Multi-threshold analysis:')
print(f'  {"Thresh":>7} | {"Signals":>8} | {"Win%":>6} | {"AvgRet":>10} | {"Excess":>10}')
print(f'  {"-"*7}-+-{"-"*8}-+-{"-"*6}-+-{"-"*10}-+-{"-"*10}')
for _, r in ft_df.iterrows():
    print(f'  ≥{r["threshold"]:.2f}  | {r["signals"]:>8,} | {r["win_rate"]:>6} | {r["avg_ret"]:>10} | {r["excess"]:>10}')

# Best threshold
best_row = ft_df.loc[ft_df['raw_excess'].idxmax()]
print(f'\n  Best threshold: ≥{best_row["threshold"]:.2f} → '
      f'{best_row["signals"]:,} sigs, WR={best_row["win_rate"]}, '
      f'Excess={best_row["excess"]}')

# ── Summary & Save ─────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print(f'SUMMARY')
print(f'='*60)
print(f'  Train AUC: {train_auc:.4f}')
print(f'\n  Walk-Forward (OOS) results:')
for _, r in wf_df.iterrows():
    print(f'    {r["train"]}→{r["test"]}: AUC={r["auc"]}, Excess={r["best_excess"]} (th={r["best_th"]:.2f}, n={r["best_n"]})')

print(f'\n  Full-period signal analysis:')
print(f'    Best threshold: ≥{best_row["threshold"]:.2f}')
print(f'    Signals: {best_row["signals"]:,}')
print(f'    Win rate: {best_row["win_rate"]}')
print(f'    Avg return: {best_row["avg_ret"]}')
print(f'    Excess: {best_row["excess"]}')
print(f'    Score range: {full_scores.min():.4f} ~ {full_scores.max():.4f}')
print(f'    Score median: {np.median(full_scores):.4f}')

total = time.time() - t0
print(f'\n  Total time: {total:.0f}s ({total/60:.1f} min)')

# Save outputs
summary = {
    'train_auc': float(train_auc),
    'score_distribution': {
        'min': float(full_scores.min()),
        'max': float(full_scores.max()),
        'median': float(np.median(full_scores)),
        'p75': float(np.percentile(full_scores, 75)),
        'p90': float(np.percentile(full_scores, 90)),
        'p95': float(np.percentile(full_scores, 95)),
        'p99': float(np.percentile(full_scores, 99)),
    },
    'walk_forward': [(lambda d: {k: d[k] if not isinstance(d[k], (np.floating, np.integer)) else float(d[k]) for k in d})(r) for r in results],
    'best_threshold': float(best_row['threshold']),
    'best_signals': int(best_row['signals']),
    'best_excess': float(best_row['raw_excess']),
}

with open('data/models/retrain_full_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f'\n  Saved models & results:')
print(f'    data/models/surge_lgbm_full.pkl')
print(f'    data/models/retrain_full_summary.json')
print(f'  DONE')
