"""
Phase 1e: Dual Label Optimization (5d_10% + 10d_15%)
=====================================================
For each label:
  1. Feature importance ranking (LightGBM on full 160 factors)
  2. Walk-Forward for feature counts [20, 30, 40, 50]
  3. Report best combo
  4. Save final model

Total estimated: 45-60 min
"""
import os, sys, json, warnings, time, gc
import numpy as np, pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'

with open(os.path.join(ROOT, 'src', 'surge', 'params.json')) as f:
    PARAMS = json.load(f)

LGB_PARAMS = PARAMS['lgbm_params'].copy()

TRAIN_WINDOWS = [
    ('2019-01-01', '2022-01-01'),
    ('2019-01-01', '2023-01-01'),
    ('2019-01-01', '2024-01-01'),
]
TEST_WINDOWS = [
    ('2022-01-01', '2023-01-01'),
    ('2023-01-01', '2024-01-01'),
    ('2024-01-01', '2025-01-01'),
]
WINDOW_NAMES = ['2022 Bear', '2023 Sideways', '2024 Recovery']

LABELS = [
    ('5d_10%',  'fwd_ret_5d',  0.10),
    ('10d_15%', 'fwd_ret_10d', 0.15),
]

# Dynamically discover available features after loading
FEATURE_COUNTS = [20, 30, 40, 50]

print("=" * 72)
print("Phase 1e: Dual Label Optimization (5d_10% + 10d_15%)")
print("=" * 72)

# ── 1. Load data ─────────────────────────────────────────────────────
print(f"\n[1/5] Loading factor cache...")
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

# Discover all feature columns dynamically
ALL_FEATURES = [c for c in pdf.columns if c.startswith(('alpha','rsi_','macd','bb_','klen','rsqr','slope','std','vma','vosc'))]
ALPHA_ONLY = [c for c in ALL_FEATURES if c.startswith('alpha')]
print(f"  Discovered {len(ALL_FEATURES)} total features ({len(ALPHA_ONLY)} alpha + {len(ALL_FEATURES)-len(ALPHA_ONLY)} technical)")

# ── 2. Forward returns ──────────────────────────────────────────────
print(f"\n[2/5] Computing forward returns for 5d and 10d...")
t0 = time.time()
close_series = pdf['close'].values
group_idx = pdf.groupby('vt_symbol', sort=False).indices

for label_name, fwd_col, threshold in LABELS:
    h = int(fwd_col.replace('fwd_ret_', '').replace('d', ''))
    pdf[fwd_col] = np.nan
    for sym, idx in group_idx.items():
        idx = sorted(idx)
        closes = pdf.loc[idx, 'close'].values
        if len(closes) > h:
            fwd = np.full(len(closes), np.nan)
            fwd[:-h] = (closes[h:] - closes[:-h]) / closes[:-h]
            pdf.loc[idx, fwd_col] = fwd
    label = label_name
    pos = (pdf[fwd_col] >= threshold).mean()
    n_pos = (pdf[fwd_col] >= threshold).sum()
    print(f"  {label_name}: {n_pos:,} positive ({pos:.1%})")
print(f"  {time.time()-t0:.0f}s")

# ── 3. Feature importance (full data) for each label ────────────────
print(f"\n[3/5] Computing feature importance...")
t0 = time.time()

all_results = {}

for label_name, fwd_col, threshold in LABELS:
    print(f"\n  --- {label_name} ---")

    # Create label
    label_series = (pdf[fwd_col] >= threshold).astype('int')
    pos_rate = label_series.mean()
    print(f"  Train pos rate: {pos_rate:.1%}")

    # Prepare features (clean inf/nan minimally)
    X_all = pdf[ALL_FEATURES].astype(np.float32).values
    X_all = np.where(np.isinf(X_all), np.nan, X_all)
    y_all = label_series.values

    # Keep only rows with at least some real data
    keep = ~np.isnan(X_all).all(axis=1)
    X_all, y_all = X_all[keep], y_all[keep]

    # Quick LightGBM to get feature importance
    lgb_data = lgb.Dataset(X_all, y_all)
    tmp_model = lgb.train(
        {**LGB_PARAMS, 'verbosity': -1, 'num_leaves': 31, 'learning_rate': 0.1,
         'feature_fraction': 0.8, 'bagging_fraction': 0.8},
        lgb_data, num_boost_round=200,
        callbacks=[lgb.log_evaluation(0)]
    )

    # Get gain-based importance
    imp = pd.DataFrame({
        'feature': ALL_FEATURES,
        'importance': tmp_model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    imp['rank'] = range(1, len(imp) + 1)

    # Show top 10
    print(f"  Top 10 features:")
    for _, r in imp.head(10).iterrows():
        print(f"    {r['rank']:2d}. {r['feature']:<10} gain={r['importance']:.0f}")

    del tmp_model
    gc.collect()

    # ── 4. Walk-Forward for each feature count ──────────────────────
    print(f"\n  Walk-Forward for feature counts: {FEATURE_COUNTS}")
    best_overall = {'score': -999, 'auc': 0, 'excess': -999, 'n_feat': 0}

    for n_feat in FEATURE_COUNTS:
        sel_features = imp.head(n_feat)['feature'].tolist()
        col_idx = [i for i, c in enumerate(ALPHA_ONLY) if c in sel_features]

        wf_aucs, wf_excesses, wf_picks = [], [], []

        for wi in range(3):
            tr_s, tr_e = TRAIN_WINDOWS[wi]
            te_s, te_e = TEST_WINDOWS[wi]

            tr_m = (pdf['date'] >= tr_s) & (pdf['date'] < tr_e)
            te_m = (pdf['date'] >= te_s) & (pdf['date'] < te_e)

            # Get data
            X_tr_full = pdf[tr_m].index.values
            X_te_full = pdf[te_m].index.values

            X_tr = pdf.loc[tr_m, sel_features].astype(np.float32).values
            X_vl_mask = pdf['date'] >= pdf[tr_m]['date'].quantile(0.8)
            # Simpler: sort dates, take last 20% for validation
            tr_dates = sorted(pdf.loc[tr_m, 'date'].unique())
            vl_date_cut = tr_dates[int(len(tr_dates) * 0.8)]
            tr_idx = pdf[tr_m & (pdf['date'] < vl_date_cut)].index.values
            vl_idx = pdf[tr_m & (pdf['date'] >= vl_date_cut)].index.values
            te_idx = pdf[te_m].index.values

            # Build arrays
            X_tr = pdf.loc[tr_idx, sel_features].astype(np.float32).values
            X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
            y_tr = label_series.loc[tr_idx].values
            tr_ok = ~np.isnan(X_tr).all(axis=1)
            X_tr, y_tr = X_tr[tr_ok], y_tr[tr_ok]

            X_vl = pdf.loc[vl_idx, sel_features].astype(np.float32).values
            X_vl = np.where(np.isinf(X_vl), np.nan, X_vl)
            y_vl = label_series.loc[vl_idx].values
            vl_ok = ~np.isnan(X_vl).all(axis=1)
            X_vl, y_vl = X_vl[vl_ok], y_vl[vl_ok]

            X_te = pdf.loc[te_idx, sel_features].astype(np.float32).values
            X_te = np.where(np.isinf(X_te), np.nan, X_te)
            y_te = label_series.loc[te_idx].values
            te_ok = ~np.isnan(X_te).all(axis=1)
            X_te, y_te = X_te[te_ok], y_te[te_ok]
            te_fwd = pdf.loc[te_idx, fwd_col].values[te_ok]

            if y_tr.sum() < 30 or y_vl.sum() < 5 or y_te.sum() < 5:
                continue

            # Train
            lgb_tr = lgb.Dataset(X_tr, y_tr)
            lgb_vl = lgb.Dataset(X_vl, y_vl, reference=lgb_tr)
            model = lgb.train(
                LGB_PARAMS, lgb_tr, valid_sets=[lgb_vl],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
            )

            # Evaluate
            scores = model.predict(X_te)
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_te, scores)) if y_te.sum() > 0 and (y_te == 0).sum() > 0 else 0.5

            # Best threshold
            best_excess = -999.0
            best_picks = 0
            for th in [x/100 for x in range(10, 95, 5)]:
                picks = scores >= th
                n_picks = picks.sum()
                if n_picks < 5:
                    continue
                avg_pick_ret = float(np.nanmean(te_fwd[picks]))
                avg_mkt_ret = float(np.nanmean(te_fwd))
                excess = avg_pick_ret - avg_mkt_ret
                if excess > best_excess:
                    best_excess = excess
                    best_picks = n_picks

            wf_aucs.append(auc)
            wf_excesses.append(best_excess * 100)
            wf_picks.append(best_picks)

        if len(wf_aucs) >= 2:
            avg_auc = float(np.mean(wf_aucs))
            avg_exc = float(np.mean(wf_excesses))
            total_picks = int(np.sum(wf_picks))
            # Score: AUC * excess * log(picks)
            score = avg_auc * max(0, avg_exc + 10) * (np.log1p(total_picks))

            status = "BEST" if score > best_overall['score'] else ""
            if score > best_overall['score']:
                best_overall = {'score': score, 'auc': avg_auc, 'excess': avg_exc,
                                'n_feat': n_feat, 'picks': total_picks,
                                'wf_aucs': wf_aucs, 'wf_excesses': wf_excesses,
                                'wf_picks': wf_picks, 'features': sel_features}

            print(f"    n={n_feat:2d}: AUC={avg_auc:.4f} Excess={avg_exc:+.2f}% "
                  f"Picks={total_picks:5d} Score={score:.1f} {status}")

    # ── 5. Train final model with best config ────────────────────────
    print(f"\n  Best config: n={best_overall['n_feat']} features "
          f"(AUC={best_overall['auc']:.4f}, Excess={best_overall['excess']:+.2f}%)")
    print(f"  Best features: {best_overall['features'][:5]}...")

    # Train final model on all available data
    all_keep = ~np.isnan(X_all).all(axis=1)
    final_X = X_all[all_keep]
    final_y = y_all[all_keep]
    # Select best features
    feat_idx = [i for i, c in enumerate(ALL_FEATURES) if c in best_overall['features']]
    final_X = final_X[:, feat_idx]

    final_data = lgb.Dataset(final_X, final_y)
    final_model = lgb.train(
        {**LGB_PARAMS, 'verbosity': -1},
        final_data, num_boost_round=200,
        callbacks=[lgb.log_evaluation(0)]
    )

    model_path = os.path.join(ROOT, 'data', 'models', f'surge_{label_name}.pkl')
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)

    # Save config
    config = {
        'label': label_name, 'horizon': int(fwd_col.split('_')[2].replace('d','')),
        'threshold': threshold, 'n_features': best_overall['n_feat'],
        'features': best_overall['features'],
        'wf_aucs': [round(a,4) for a in best_overall['wf_aucs']],
        'wf_excesses': [round(e,2) for e in best_overall['wf_excesses']],
        'wf_picks': best_overall['wf_picks'],
        'avg_auc': round(best_overall['auc'], 4),
        'avg_excess': round(best_overall['excess'], 2),
        'pos_rate': round(float(pos_rate), 3),
    }
    all_results[label_name] = config

    print(f"  Model saved: surge_{label_name}.pkl")
    print(f"  Time elapsed: {(time.time()-t0)/60:.1f} min")

# ── 6. Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("RESULTS SUMMARY")
print("=" * 72)
print(f"{'Label':<12} {'nFeat':>6} {'PosRate':>8} {'AUC':>8} {'Excess':>10} {'Picks':>8}")
print("-" * 72)
for label_name, fwd_col, threshold in LABELS:
    r = all_results[label_name]
    print(f"{label_name:<12} {r['n_features']:>6} {r['pos_rate']:>7.1%} "
          f"{r['avg_auc']:>8.4f} {r['avg_excess']:>+9.2f}% {sum(r['wf_picks']):>8}")

# Save combined results
output_path = os.path.join(ROOT, 'data', 'models', 'phase1e_dual_label_results.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"\nResults saved: phase1e_dual_label_results.json")
print(f"Total time: {(time.time()-t0)/60:.1f} min")
