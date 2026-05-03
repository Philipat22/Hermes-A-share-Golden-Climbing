"""
Phase 1d: Label Horizon & Threshold Systematic Sweep
======================================================
For each (horizon, threshold) combination, train LightGBM via Walk-Forward
and report AUC + excess return + pick count.

Labels:
  5d: 3%, 5%, 7%, 10%
 10d: 5%, 7%, 10%, 15%
 20d: 7%, 10%, 15%
 30d: 10%, 15%, 20%
 60d: 15%, 20%, 30%
"""
import os, sys, json, warnings, time
import pandas as pd, numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')
ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'

# -- Config ----------------------------------------------------------
with open(os.path.join(ROOT, 'src', 'surge', 'params.json')) as f:
    PARAMS = json.load(f)
FEATURES = PARAMS['selected_features']  # 40 features
LGB_PARAMS = PARAMS['lgbm_params']

TRAIN_WINDOWS = [
    ('2019-01-01', '2022-01-01'),   # 2019-2021 train -> 2022 test
    ('2019-01-01', '2023-01-01'),   # 2019-2022 train -> 2023 test
    ('2019-01-01', '2024-01-01'),   # 2019-2023 train -> 2024 test
]
TEST_WINDOWS = [
    ('2022-01-01', '2023-01-01'),
    ('2023-01-01', '2024-01-01'),
    ('2024-01-01', '2025-01-01'),
]
WINDOW_NAMES = ['2022 Bear', '2023 Sideways', '2024 Recovery']

LABEL_CONFIGS = [
    (5, 0.03),  (5, 0.05),  (5, 0.07),  (5, 0.10),
    (10, 0.05), (10, 0.07), (10, 0.10), (10, 0.15),
    (20, 0.07), (20, 0.10), (20, 0.15),
    (30, 0.10), (30, 0.15), (30, 0.20),
    (60, 0.15), (60, 0.20), (60, 0.30),
]

print("=" * 70)
print("Phase 1d: Label Horizon & Threshold Systematic Sweep")
print("=" * 70)

# -- 1. Load factor cache --------------------------------------------
print(f"\n[1/4] Loading factor cache...")
t0 = time.time()
FACTOR_DIR = os.path.join(ROOT, 'data', 'cache', 'factors_batched')
dfs = []
for fn in sorted(os.listdir(FACTOR_DIR)):
    if fn.endswith('.parquet'):
        dfs.append(pd.read_parquet(os.path.join(FACTOR_DIR, fn)))
pdf = pd.concat(dfs, ignore_index=True)
pdf['date'] = pd.to_datetime(pdf['datetime'])
pdf = pdf.sort_values(['vt_symbol', 'date']).reset_index(drop=True)
print(f"  {len(pdf):,} rows, {pdf['vt_symbol'].nunique()} stocks")
print(f"  Date: {pdf['date'].min().date()} ~ {pdf['date'].max().date()}")
print(f"  {time.time()-t0:.0f}s")

# -- 2. Compute forward returns for all horizons ---------------------
print(f"\n[2/4] Computing forward returns for all horizons...")
t0 = time.time()

# Sort within each stock to compute forward close correctly
close_series = pdf['close'].values
group_boundaries = pdf.groupby('vt_symbol', sort=False).indices
horizons = sorted(set(h for h, _ in LABEL_CONFIGS))

# Initialize forward return columns as NaN
for h in horizons:
    pdf[f'fwd_ret_{h}d'] = np.nan

for symbol, idx in group_boundaries.items():
    idx = sorted(idx)
    closes = pdf.loc[idx, 'close'].values
    for h in horizons:
        # fwd_ret = (future price - current price) / current price
        if len(closes) > h:
            fwd = np.full(len(closes), np.nan)
            fwd[:-h] = (closes[h:] - closes[:-h]) / closes[:-h]
            pdf.loc[idx, f'fwd_ret_{h}d'] = fwd

print(f"  Horizons: {horizons}")
print(f"  {time.time()-t0:.0f}s")

# -- 3. Walk-Forward for each label config ---------------------------
print(f"\n[3/4] Running Walk-Forward for {len(LABEL_CONFIGS)} label configs...")
t0 = time.time()

results = []
for horizon, threshold in LABEL_CONFIGS:
    label_col = f'fwd_ret_{horizon}d'
    config_name = f"{horizon}d_{threshold*100:.0f}%"

    # Create label: 1 if forward_ret >= threshold
    pdf['label'] = (pdf[label_col] >= threshold).astype('int')

    # Check positive rate
    pos_rate = pdf['label'].mean()
    if pos_rate < 0.03 or pos_rate > 0.60:
        print(f"  {config_name}: skip (pos_rate={pos_rate:.1%})")
        continue

    config_results = []
    for wi, (train_start, train_end) in enumerate(TRAIN_WINDOWS):
        test_start, test_end = TEST_WINDOWS[wi]
        wname = WINDOW_NAMES[wi]

        # Split data
        train_mask = (pdf['date'] >= train_start) & (pdf['date'] < train_end)
        test_mask = (pdf['date'] >= test_start) & (pdf['date'] < test_end)

        train_df = pdf[train_mask].copy()
        test_df = pdf[test_mask].copy()

        # Split train into train/val (80/20 by time)
        train_dates = sorted(train_df['date'].unique())
        split_idx = int(len(train_dates) * 0.8)
        val_date_thresh = train_dates[split_idx]
        tr_df = train_df[train_df['date'] < val_date_thresh]
        vl_df = train_df[train_df['date'] >= val_date_thresh]

        # Prepare features (keep NaN for LightGBM)
        X_tr = tr_df[FEATURES].astype(np.float32).values
        X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
        y_tr = tr_df['label'].values

        X_vl = vl_df[FEATURES].astype(np.float32).values
        X_vl = np.where(np.isinf(X_vl), np.nan, X_vl)
        y_vl = vl_df['label'].values

        X_te = test_df[FEATURES].astype(np.float32).values
        X_te = np.where(np.isinf(X_te), np.nan, X_te)
        y_te = test_df['label'].values

        # Drop rows where ALL features are NaN
        tr_keep = ~np.isnan(X_tr).all(axis=1)
        vl_keep = ~np.isnan(X_vl).all(axis=1)
        te_keep = ~np.isnan(X_te).all(axis=1)
        X_tr, y_tr = X_tr[tr_keep], y_tr[tr_keep]
        X_vl, y_vl = X_vl[vl_keep], y_vl[vl_keep]
        X_te, y_te, te_df = X_te[te_keep], y_te[te_keep], test_df[te_keep].copy()

        # Need at least some positive samples
        if y_tr.sum() < 50 or y_vl.sum() < 10 or y_te.sum() < 10:
            continue

        # Train
        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_val = lgb.Dataset(X_vl, y_vl, reference=lgb_train)
        model = lgb.train(
            LGB_PARAMS, lgb_train,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )

        # Predict
        scores = model.predict(X_te)
        from sklearn.metrics import roc_auc_score
        if y_te.sum() > 0 and (y_te == 0).sum() > 0:
            auc = float(roc_auc_score(y_te, scores))
        else:
            auc = 0.5

        # Find best threshold (from 0.1 to 0.9, step 0.05)
        best_excess = -999
        best_th = 0.5
        best_picks = 0
        for th in [x / 100 for x in range(10, 95, 5)]:
            picks = scores >= th
            n_picks = picks.sum()
            if n_picks < 10:
                continue
            pick_rets = te_df.loc[picks, label_col].values
            avg_pick_ret = float(np.nanmean(pick_rets))
            # Market average return
            market_ret = float(np.nanmean(te_df[label_col].values))
            excess = avg_pick_ret - market_ret
            if excess > best_excess:
                best_excess = excess
                best_th = th
                best_picks = n_picks

        config_results.append({
            'window': wname, 'auc': round(auc, 4),
            'excess': round(best_excess * 100, 2),
            'threshold': best_th, 'picks': best_picks,
            'pos_rate': round(float(y_te.mean()), 3)
        })

    if config_results:
        avg_auc = np.mean([r['auc'] for r in config_results])
        avg_excess = np.mean([r['excess'] for r in config_results])
        total_picks = sum(r['picks'] for r in config_results)
        results.append({
            'config': config_name,
            'horizon': horizon,
            'threshold': threshold,
            'pos_rate_train': round(pos_rate, 3),
            'avg_auc': round(float(avg_auc), 4),
            'avg_excess': round(float(avg_excess), 2),
            'total_picks': int(total_picks),
            'details': config_results
        })
        print(f"  OK {config_name}: AUC={avg_auc:.4f} Excess={avg_excess:+.2f}% Picks={total_picks}")
    else:
        print(f"  FAIL {config_name}: No valid windows")

print(f"  Total sweep: {time.time()-t0:.0f}s")

# -- 4. Output results ----------------------------------------------
print(f"\n[4/4] Results summary")
print("=" * 70)
print(f"{'Config':<18} {'PosRate':>8} {'Avg AUC':>8} {'Avg Excess':>10} {'Picks':>8} {'Score':>8}")
print("-" * 70)

# Score = AUC * abs(excess) * sqrt(picks) / sqrt(total possible)
# Normalize so higher is better
max_picks_total = max(r['total_picks'] for r in results) if results else 1
for r in results:
    aucex = r['avg_excess'] + 20  # shift positive
    score = r['avg_auc'] * max(0, aucex) * (r['total_picks'] / max_picks_total) ** 0.3
    r['score'] = round(float(score), 2)
    score_str = f"{r['score']:.1f}"

    print(f"{r['config']:<18} {r['pos_rate_train']:>7.1%} {r['avg_auc']:>8.4f} {r['avg_excess']:>+9.2f}% {r['total_picks']:>8} {score_str:>8}")

# Sort by score
results.sort(key=lambda r: r['score'], reverse=True)

print("\n" + "=" * 70)
print("TOP 5 LABEL CONFIGURATIONS")
print("=" * 70)
for i, r in enumerate(results[:5]):
    print(f"\n#{i+1} {r['config']:<15} Score={r['score']}")
    print(f"   PosRate train={r['pos_rate_train']:.1%}  Avg AUC={r['avg_auc']:.4f}  Avg Excess={r['avg_excess']:+.2f}%  Total picks={r['total_picks']}")
    for d in r['details']:
        print(f"   ├- {d['window']}: AUC={d['auc']:.4f} Excess={d['excess']:+.2f}% th={d['threshold']:.2f} picks={d['picks']}")

# -- Save to file
output = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'rankings': [{'rank': i+1, **{k: r[k] for k in ['config','horizon','threshold','pos_rate_train','avg_auc','avg_excess','total_picks','score']},
                  'details': r['details']} for i, r in enumerate(results)]
}
os.makedirs(os.path.join(ROOT, 'data', 'models'), exist_ok=True)
with open(os.path.join(ROOT, 'data', 'models', 'phase1d_label_sweep.json'), 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: data/models/phase1d_label_sweep.json")
print(f"Total time: {(time.time()-t0)/60:.1f} min")
