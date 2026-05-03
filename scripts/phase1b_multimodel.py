"""
Phase 1b: Multi-Model Fusion (Bull/Bear/Sideways)
===================================================
对比单模型 vs 多模型 (按市场状态各自训练)。

流程:
  1. 加载因子缓存 + 计算标签
  2. 加载CSI300 → 给每条数据标注市场状态
  3. 分市场训练三个模型 (BULL / BEAR / SIDEWAYS)
  4. Walk-Forward: 单模型 vs 多模型对比
  5. 输出谁赢了

用法: python scripts/phase1b_multimodel.py
"""
import sys, os, pickle, time, json, gc, warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TUSHARE_CACHE'] = 'data/cache'

ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

print('=' * 65)
print('Phase 1b: Multi-Model (Bull/Bear/Sideways) vs Single Model')
print('=' * 65)

# ── 1. Load data ──────────────────────────────────────────────────────────
print('\n[1/6] Loading factor cache...')
t0 = time.time()
FACTOR_DIR = os.path.join(ROOT, 'data', 'cache', 'factors_batched')
all_dfs = []
for fn in sorted(os.listdir(FACTOR_DIR)):
    if fn.endswith('.parquet'):
        all_dfs.append(pd.read_parquet(os.path.join(FACTOR_DIR, fn)))
pdf = pd.concat(all_dfs, ignore_index=True)
print(f'  Rows: {len(pdf):,}  Cols: {len(pdf.columns)}  {time.time()-t0:.0f}s')

# ── 2. Create labels ──────────────────────────────────────────────────────
print('\n[2/6] Creating labels...')
HORIZON = 20
pdf = pdf.sort_values(['vt_symbol', 'datetime']).reset_index(drop=True)
pdf['forward_ret'] = np.nan
for name, group in pdf.groupby('vt_symbol', sort=False):
    idx = group.index
    closes = group['close'].values
    for i in range(len(group) - HORIZON):
        pdf.loc[idx[i], 'forward_ret'] = (closes[i+HORIZON] - closes[i]) / closes[i]
pdf['label'] = (pdf['forward_ret'] >= 0.10).astype(int)
pdf['date'] = pd.to_datetime(pdf['datetime'])
pdf['year'] = pdf['date'].dt.year
print(f'  Surge rate: {pdf["label"].mean():.2%}')
print(f'  Date range: {pdf["date"].min().date()} ~ {pdf["date"].max().date()}')

# ── 3. Load regime classifier ────────────────────────────────────────────
print('\n[3/6] Loading regime classifier & merging...')
from src.surge.regime_classifier import RegimeClassifier
rc = RegimeClassifier()
bulk = rc.classify_bulk()
bulk['trade_date'] = pd.to_datetime(bulk['trade_date'])
print(f'  Regime days: {len(bulk)}')

# Merge regime into factor data
bulk_map = bulk.set_index('trade_date')['regime_id'].to_dict()
pdf['regime_id'] = pdf['date'].map(bulk_map).fillna(2).astype(int)  # default SIDEWAYS

# Distribution
regime_dist = pdf.groupby('year')['regime_id'].value_counts().unstack(fill_value=0)
regime_dist.columns = ['BULL', 'BEAR', 'SIDEWAYS']
print(f'\n  Factor rows per regime:')
print(f'  BULL:     {(pdf["regime_id"]==0).sum():>8,}')
print(f'  BEAR:     {(pdf["regime_id"]==1).sum():>8,}')
print(f'  SIDEWAYS: {(pdf["regime_id"]==2).sum():>8,}')

# ── 4. Prepare features ──────────────────────────────────────────────────
print('\n[4/6] Preparing features...')
with open(os.path.join(ROOT, 'src', 'surge', 'params.json')) as f:
    params = json.load(f)
TOP_40 = params['selected_features']
valid_features = [f for f in TOP_40 if f in pdf.columns]
print(f'  Features: {len(valid_features)} / {len(TOP_40)}')

# Clean NaN rows
feature_data = pdf[valid_features].astype(np.float32)
feature_data = np.clip(feature_data, -1e10, 1e10)
fv = feature_data.values
fv[np.isinf(fv)] = np.nan
nan_mask = ~np.isnan(fv).any(axis=1)
print(f'  Dropped {(~nan_mask).sum():,} NaN rows ({nan_mask.mean():.0%} valid)')
clean = pdf[nan_mask].copy()

# ── 5. Walk-Forward Backtest ─────────────────────────────────────────────
print('\n[5/6] Walk-Forward backtest...')

# Time windows
windows = [
    (2019, 2021, 2022, '2022 Bear'),
    (2019, 2022, 2023, '2023 Sideways'),
    (2019, 2023, 2024, '2024 Recovery'),
]

all_results = []


def train_and_test(train_df, test_df, label=''):
    """Train model(s) on train_df, test on test_df.

    Returns comparison dict.
    """
    if len(train_df) < 5000 or len(test_df) < 100:
        return None

    X_te_cols = [c for c in TOP_40 if c in test_df.columns]

    # Pre-clean test data
    X_te_raw = test_df[X_te_cols].astype(np.float32).values
    X_te_raw = np.clip(X_te_raw, -1e10, 1e10)
    X_te_raw = np.where(np.isinf(X_te_raw), np.nan, X_te_raw)
    te_nan = ~np.isnan(X_te_raw).any(axis=1)
    X_te = X_te_raw[te_nan]
    y_te = test_df[te_nan]['label'].values
    te_rets = test_df[te_nan]['forward_ret'].values
    te_regime = test_df[te_nan]['regime_id'].values

    if len(X_te) < 100:
        return None

    market_avg = te_rets.mean()

    lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'num_leaves': 64,
        'learning_rate': 0.02, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5,
        'min_data_in_leaf': 100, 'max_depth': 10, 'verbosity': -1, 'seed': 42,
    }

    # ── Train Single Model ────────────────────────────────────────────
    print(f'    [{label}] Training single model...')
    X_tr = train_df[X_te_cols].astype(np.float32).values
    X_tr = np.clip(X_tr, -1e10, 1e10)
    X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
    tr_nan = ~np.isnan(X_tr).any(axis=1)
    X_tr = X_tr[tr_nan]
    y_tr = train_df[tr_nan]['label'].values

    val_split = int(len(X_tr) * 0.8)
    single_model = lgb.train(
        lgb_params,
        lgb.Dataset(X_tr[:val_split], y_tr[:val_split],
                     feature_name=[str(i) for i in range(X_tr.shape[1])]),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_tr[val_split:], y_tr[val_split:])],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    single_scores = single_model.predict(X_te)
    single_auc = roc_auc_score(y_te, single_scores)

    # ── Train Multi Models ────────────────────────────────────────────
    print(f'    [{label}] Training multi models (by regime)...')
    regime_models = {}
    for rid in [0, 1, 2]:  # BULL, BEAR, SIDEWAYS
        regime_train = train_df[train_df['regime_id'] == rid]
        if len(regime_train) < 2000 or regime_train['label'].sum() < 50:
            print(f'      Regime {rid}: skip (n={len(regime_train)}, pos={regime_train["label"].sum()})')
            continue

        Xr = regime_train[X_te_cols].astype(np.float32).values
        Xr = np.clip(Xr, -1e10, 1e10)
        Xr = np.where(np.isinf(Xr), np.nan, Xr)
        r_nan = ~np.isnan(Xr).any(axis=1)
        Xr = Xr[r_nan]
        yr = regime_train[r_nan]['label'].values

        if len(Xr) < 1000 or yr.sum() < 30:
            print(f'      Regime {rid}: skip after NaN drop (n={len(Xr)}, pos={yr.sum()})')
            continue

        vs = int(len(Xr) * 0.8)
        m = lgb.train(
            lgb_params,
            lgb.Dataset(Xr[:vs], yr[:vs]),
            num_boost_round=500,
            valid_sets=[lgb.Dataset(Xr[vs:], yr[vs:])],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        regime_models[rid] = m
        print(f'      Regime {rid}: trained on {len(Xr)} samples, AUC_val done')

    if len(regime_models) == 0:
        print(f'    [{label}] No regime models trained, skip')
        return None

    # ── Test Multi Model ──────────────────────────────────────────────
    multi_scores = np.zeros(len(X_te))
    for i in range(len(X_te)):
        regime = te_regime[i]
        if regime in regime_models:
            multi_scores[i] = regime_models[regime].predict(X_te[i:i+1])[0]
        elif 0 in regime_models:
            # Fallback to BULL model
            multi_scores[i] = regime_models[0].predict(X_te[i:i+1])[0]
        else:
            multi_scores[i] = single_scores[i]  # fallback

    multi_auc = roc_auc_score(y_te, multi_scores)

    # ── Compare thresholds ────────────────────────────────────────────
    result_row = {
        'window': label,
        'test_n': len(X_te),
        'market_avg': f'{market_avg*100:+.2f}%',
        # Single
        'single_auc': round(single_auc, 4),
        'multi_auc': round(multi_auc, 4),
        'auc_delta': f'{(multi_auc - single_auc)*100:+.2f}%',
    }

    # Find best threshold for each
    for prefix, scores in [('single', single_scores), ('multi', multi_scores)]:
        best_excess = -999
        best_th = best_wr = best_n = 0
        for th in [t/100 for t in range(5, 60, 5)]:
            picks = scores >= th
            if picks.sum() < 10:
                continue
            avg_ret = te_rets[picks].mean()
            excess = avg_ret - market_avg
            if excess > best_excess:
                best_excess = excess
                best_th = th
                best_wr = (te_rets[picks] > 0).mean()
                best_n = picks.sum()
        result_row[f'{prefix}_best_th'] = best_th
        result_row[f'{prefix}_n'] = best_n
        result_row[f'{prefix}_excess'] = f'{best_excess*100:+.2f}%'
        result_row[f'{prefix}_wr'] = f'{best_wr:.1%}'

    # Delta
    single_ex = float(result_row['single_excess'].replace('%','').replace('+',''))
    multi_ex = float(result_row['multi_excess'].replace('%','').replace('+',''))
    result_row['excess_delta'] = f'{multi_ex - single_ex:+.2f}%'

    all_results.append(result_row)

    print(f'\n  [{label}] Single: AUC={single_auc:.4f} '
          f'>>{result_row["single_best_th"]:.2f} '
          f'Excess={result_row["single_excess"]} '
          f'WR={result_row["single_wr"]} ({result_row["single_n"]}trades)')
    print(f'           Multi:  AUC={multi_auc:.4f} '
          f'>>{result_row["multi_best_th"]:.2f} '
          f'Excess={result_row["multi_excess"]} '
          f'WR={result_row["multi_wr"]} ({result_row["multi_n"]}trades)')

    return result_row


# Run windows
for train_start, train_end, test_year, label in windows:
    train_mask = (clean['year'] >= train_start) & (clean['year'] <= train_end)
    test_mask = clean['year'] == test_year

    train_and_test(clean[train_mask], clean[test_mask], label)
    gc.collect()

# ── 6. Results ─────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('RESULTS: Single Model vs Multi-Model (Regime-Specific)')
print('='*65)

if not all_results:
    print('  No results generated (all windows skipped)')
    sys.exit(0)

print(f'\n{"Window":<20} | {"Single AUC":>10} | {"Multi AUC":>9} | {"AUC Delta":>9} | '
      f'{"Single Ex":>10} | {"Multi Ex":>10} | {"Ex Delta":>9}')
print(f'{"-"*20}-+-{"-"*10}-+-{"-"*9}-+-{"-"*9}-+-{"-"*10}-+-{"-"*10}-+-{"-"*9}')
for r in all_results:
    print(f'{r["window"]:<20} | {r["single_auc"]:>10.4f} | {r["multi_auc"]:>9.4f} | '
          f'{r["auc_delta"]:>9} | {r["single_excess"]:>10} | {r["multi_excess"]:>10} | '
          f'{r["excess_delta"]:>9}')

# Summary
avg_single_auc = np.mean([r['single_auc'] for r in all_results])
avg_multi_auc = np.mean([r['multi_auc'] for r in all_results])
ex_deltas = [float(r['excess_delta'].replace('%','').replace('+','')) for r in all_results]
avg_ex_delta = np.mean(ex_deltas)
wins = sum(1 for d in ex_deltas if d > 0)
losses = sum(1 for d in ex_deltas if d < 0)

print(f'{"-"*20}-+-{"-"*10}-+-{"-"*9}-+-{"-"*9}-+-{"-"*10}-+-{"-"*10}-+-{"-"*9}')
print(f'{"AVG":<20} | {avg_single_auc:>10.4f} | {avg_multi_auc:>9.4f} | '
      f'{((avg_multi_auc-avg_single_auc)*100):>+8.2f}% | {"":>10} | {"":>10} | '
      f'{avg_ex_delta:>+8.2f}%')

print(f'\n  Multi-model wins: {wins}/{len(all_results)}')
print(f'  Avg excess delta: {avg_ex_delta:+.2f}%')

if avg_ex_delta > 1.0:
    print(f'\n  Conclusion: MULTI-MODEL WINS (avg +{avg_ex_delta:.2f}% excess improvement)')
elif avg_ex_delta < -1.0:
    print(f'\n  Conclusion: SINGLE MODEL WINS (multi-model avg {avg_ex_delta:.2f}% worse)')
else:
    print(f'\n  Conclusion: TIE (difference within noise, {avg_ex_delta:.2f}%)')

# ── Save ───────────────────────────────────────────────────────────────────
print(f'\n  Saving results...')
out = {
    'title': 'Phase 1b: Multi-Model vs Single Model',
    'timestamp': pd.Timestamp.now().isoformat(),
    'features': len(valid_features),
    'results': all_results,
    'summary': {
        'avg_single_auc': round(avg_single_auc, 4),
        'avg_multi_auc': round(avg_multi_auc, 4),
        'avg_excess_delta': f'{avg_ex_delta:+.2f}%',
        'multi_model_wins': wins,
        'total_windows': len(all_results),
    },
}
out_path = os.path.join(ROOT, 'data', 'models', 'phase1b_results.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f'  Saved: {out_path}')
print(f'\n  Total time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)')
print('='*65)
