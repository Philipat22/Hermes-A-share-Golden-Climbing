"""
Phase 1: ML + Momentum Filter Combined Backtest
================================================
验证动量过滤是否提升ML选股的超额收益。

流程:
  1. 加载全量因子缓存 (805k rows)
  2. 计算ML评分（用全量训练的模型）
  3. 叠加动量过滤 (MA50/MA200)
  4. Walk-Forward OOS验证
  5. 输出对比表: ML-only vs ML+Momentum vs 纯动量

用法: python scripts/phase1_momentum_backtest.py
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

print('=' * 65)
print('Phase 1: ML + Momentum Filter Combined Backtest')
print('=' * 65)

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
print(f'  Rows: {len(pdf):,}  Columns: {len(pdf.columns)}  Time: {time.time()-t0:.0f}s')

# ── 2. Create labels (20d forward) ────────────────────────────────────────
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
pdf['year'] = pdf['date'].dt.year

print(f'  Surge rate (label=1): {pdf["label"].mean():.2%}')
print(f'  Date range: {pdf["date"].min().date()} ~ {pdf["date"].max().date()}')

# ── 3. Load model and features ────────────────────────────────────────────
print('\n[3/5] Loading ML model & features...')

# Load ensemble model
MODEL_PATH = os.path.join(ROOT, 'data', 'models', 'surge_lgbm_full.pkl')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(ROOT, 'data', 'models', 'surge_lgbm.pkl')

with open(MODEL_PATH, 'rb') as f:
    lgb_model = pickle.load(f)
print(f'  Model: {MODEL_PATH}')

# Load selected features
with open(os.path.join(ROOT, 'src', 'surge', 'params.json')) as f:
    params = json.load(f)
TOP_40 = params['selected_features']
valid_features = [f for f in TOP_40 if f in pdf.columns]
print(f'  Features: {len(valid_features)} / {len(TOP_40)} available')

# ── 4. Compute ML scores ──────────────────────────────────────────────────
print('\n[4/5] Computing ML scores...')
t_ml = time.time()

# Prepare feature matrix
feature_data = pdf[valid_features].astype(np.float32)
feature_data = np.clip(feature_data, -1e10, 1e10)

# Replace inf with nan (operate on values to keep DataFrame)
fv = feature_data.values
fv[np.isinf(fv)] = np.nan

# Drop NaN rows
nan_mask = ~np.isnan(fv).any(axis=1)
print(f'  Dropping {(~nan_mask).sum():,} rows with NaN ({nan_mask.mean():.0%} valid)')
X_valid = fv[nan_mask]
clean = pdf[nan_mask].copy()

# Predict
all_scores = lgb_model.predict(X_valid)
clean['ml_score'] = all_scores
print(f'  ML scores: min={all_scores.min():.4f} max={all_scores.max():.4f} '
      f'mean={all_scores.mean():.4f}')
print(f'  Time: {time.time()-t_ml:.0f}s')

# ── 5. Apply momentum filter ──────────────────────────────────────────────
print('\n[5/5] Walking forward & applying momentum filter...')
t_mom = time.time()

# Compute momentum indicators per stock
print('  Computing MA50/MA200 per stock...')
clean['ma50'] = np.nan
clean['ma200'] = np.nan
clean['trend_ok'] = False

for name, group in clean.groupby('vt_symbol', sort=False):
    idx = group.index
    closes = group['close'].values
    if len(closes) < 200:
        continue

    # MA50
    ma50_series = pd.Series(closes).rolling(50, min_periods=50).mean().values
    # MA200
    ma200_series = pd.Series(closes).rolling(200, min_periods=200).mean().values

    # Trend condition: close > MA50 > MA200
    trend_ok = (closes > ma50_series) & (ma50_series > ma200_series)
    # Also need ma50 > ma200*(previous period) ie. ma50 slope positive is implicit in condition

    # Only apply to aligned index
    n = len(idx)
    clean.loc[idx, 'ma50'] = ma50_series
    clean.loc[idx, 'ma200'] = ma200_series
    clean.loc[idx, 'trend_ok'] = trend_ok

valid_trend = clean['ma50'].notna()
print(f'  Rows with valid MA: {valid_trend.sum():,} ({valid_trend.mean():.0%})')
print(f'  Trend OK rate (full): {clean["trend_ok"].mean():.1%}')

# ── 6. Walk-Forward backtest ──────────────────────────────────────────────
print('\n  Running Walk-Forward backtest...')

# Time windows (rolling train/test)
windows = [
    (2019, 2021, 2022, '2022熊市'),
    (2019, 2021, 2023, '2023震荡'),
    (2019, 2022, 2023, '2023震荡(更长训练)'),
    (2019, 2022, 2024, '2024反弹'),
    (2019, 2023, 2025, '2025'),
]
# Also add a combined: train 2019-2023 test 2024-2025
windows.append((2019, 2023, None, '2024-2025合'))  # None = combined later

# Thresholds to test
ML_THRESHOLDS = [t/100 for t in range(5, 60, 5)]

def backtest_window(train_mask, test_mask, label=''):
    """Run one window backtest comparing ML vs ML+Momentum"""
    train_df = clean[train_mask].copy()
    test_df = clean[test_mask].copy()

    if len(train_df) < 10000 or len(test_df) < 100:
        return None

    # Train LightGBM on this window
    X_tr_cols = [c for c in valid_features if c in train_df.columns]
    X_tr = train_df[X_tr_cols].astype(np.float32).values
    X_tr = np.clip(X_tr, -1e10, 1e10)
    X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
    nan_tr = ~np.isnan(X_tr).any(axis=1)

    X_tr = X_tr[nan_tr]
    y_tr = train_df[nan_tr]['label'].values

    # Validation split for early stopping
    val_split = int(len(X_tr) * 0.8)
    model = lgb.train(
        {'objective': 'binary', 'metric': 'auc', 'num_leaves': 64,
         'learning_rate': 0.02, 'feature_fraction': 0.8,
         'bagging_fraction': 0.8, 'bagging_freq': 5,
         'min_data_in_leaf': 100, 'max_depth': 10, 'verbosity': -1, 'seed': 42},
        lgb.Dataset(X_tr[:val_split], y_tr[:val_split],
                     feature_name=[str(i) for i in range(X_tr.shape[1])]),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_tr[val_split:], y_tr[val_split:])],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    # Test
    X_te = test_df[X_tr_cols].astype(np.float32).values
    X_te = np.clip(X_te, -1e10, 1e10)
    X_te = np.where(np.isinf(X_te), np.nan, X_te)
    nan_te = ~np.isnan(X_te).any(axis=1)
    X_te = X_te[nan_te]
    test_data = test_df[nan_te].copy()

    scores = model.predict(X_te)
    test_data['wf_score'] = scores
    auc = __import__('sklearn').metrics.roc_auc_score(test_data['label'].values, scores)

    market_avg = test_data['forward_ret'].mean()

    rows = []
    for th in ML_THRESHOLDS:
        # ---- ML only ----
        picks_ml = test_data[test_data['wf_score'] >= th]
        if len(picks_ml) >= 10:
            ml_excess = picks_ml['forward_ret'].mean() - market_avg
            ml_wr = (picks_ml['forward_ret'] > 0).mean()
        else:
            ml_excess = 0
            ml_wr = 0

        # ---- ML + Momentum ----
        picks_mom = test_data[(test_data['wf_score'] >= th) & (test_data['trend_ok'])]
        if len(picks_mom) >= 5:
            mom_excess = picks_mom['forward_ret'].mean() - market_avg
            mom_wr = (picks_mom['forward_ret'] > 0).mean()
        else:
            mom_excess = 0
            mom_wr = 0

        rows.append({
            'threshold': th,
            'ml_n': len(picks_ml),
            'ml_excess': ml_excess,
            'ml_wr': ml_wr,
            'mom_n': len(picks_mom),
            'mom_excess': mom_excess,
            'mom_wr': mom_wr,
        })

    # Find best thresholds
    best_ml = max(rows, key=lambda r: r['ml_excess'])
    best_mom = max(rows, key=lambda r: r['mom_excess'])
    # Also best for momentum only (no ML)
    best_mom_only_rows = []
    for th in ML_THRESHOLDS:
        picks = test_data[(test_data['wf_score'] >= 0) & (test_data['trend_ok'])]
        if len(picks) >= 10:
            mom_only_excess = picks['forward_ret'].mean() - market_avg
            mom_only_wr = (picks['forward_ret'] > 0).mean()
            best_mom_only_rows.append({
                'n': len(picks), 'excess': mom_only_excess, 'wr': mom_only_wr
            })
    best_mom_only = max(best_mom_only_rows, key=lambda r: r['excess']) if best_mom_only_rows else {'n':0, 'excess':0, 'wr':0}

    # Additional: pure momentum (MA50>MA200 with no ML filter)
    pure_mom = test_data[test_data['trend_ok']]
    if len(pure_mom) >= 10:
        pure_mom_excess = pure_mom['forward_ret'].mean() - market_avg
        pure_mom_wr = (pure_mom['forward_ret'] > 0).mean()
    else:
        pure_mom_excess = 0
        pure_mom_wr = 0

    return {
        'window': label,
        'auc': round(auc, 4),
        'market_avg': f'{market_avg*100:+.2f}%',
        'test_n': len(test_data),
        # ML-only best
        'ml_best_th': best_ml['threshold'],
        'ml_n': best_ml['ml_n'],
        'ml_excess': f'{best_ml["ml_excess"]*100:+.2f}%',
        'ml_wr': f'{best_ml["ml_wr"]:.1%}',
        # ML + Momentum best
        'mom_best_th': best_mom['threshold'],
        'mom_n': best_mom['mom_n'],
        'mom_excess': f'{best_mom["mom_excess"]*100:+.2f}%',
        'mom_wr': f'{best_mom["mom_wr"]:.1%}',
        # Pure momentum
        'pure_mom_excess': f'{pure_mom_excess*100:+.2f}%',
        'pure_mom_wr': f'{pure_mom_wr:.1%}',
        # Improvement
        'delta': f'{(best_mom["mom_excess"] - best_ml["ml_excess"])*100:+.2f}%',
    }


all_results = []
for train_start, train_end, test_year, label in windows:
    if test_year is not None:
        train_mask = (clean['year'] >= train_start) & (clean['year'] <= train_end)
        test_mask = clean['year'] == test_year
    else:
        # Combined window (e.g., 2024-2025)
        train_mask = (clean['year'] >= train_start) & (clean['year'] <= train_end)
        test_mask = (clean['year'] >= 2024) & (clean['year'] <= 2025)

    result = backtest_window(train_mask, test_mask, label)
    if result:
        all_results.append(result)
        print(f'  [{label}] AUC={result["auc"]} | '
              f'ML: +{result["ml_excess"]}@{result["ml_best_th"]:.2f}({result["ml_n"]}sig) | '
              f'ML+Momentum: +{result["mom_excess"]}@{result["mom_best_th"]:.2f}({result["mom_n"]}sig) | '
              f'Δ={result["delta"]}')
    else:
        print(f'  [{label}] SKIP (insufficient data)')

    gc.collect()

# ── 7. Format results ─────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('WALK-FORWARD COMPARISON: ML-only vs ML+Momentum vs Pure Momentum')
print('='*65)
print(f'{"Window":<18} | {"AUC":>5} | {"ML Excess":>10} | {"ML+Mom Excess":>13} | {"PureMom Ex":>10} | {"Δ(excess)":>9} | {"ML WR":>6} | {"Mom WR":>6}')
print(f'{"-"*18}-+-{"-"*5}-+-{"-"*10}-+-{"-"*13}-+-{"-"*10}-+-{"-"*9}-+-{"-"*6}-+-{"-"*6}')
for r in all_results:
    print(f'{r["window"]:<18} | {r["auc"]:>5.4f} | {r["ml_excess"]:>10} | '
          f'{r["mom_excess"]:>13} | {r["pure_mom_excess"]:>10} | '
          f'{r["delta"]:>9} | {r["ml_wr"]:>6} | {r["mom_wr"]:>6}')

# Average improvement
if all_results:
    avg_ml = np.mean([float(r['ml_excess'].replace('%','').replace('+','')) for r in all_results])
    avg_mom = np.mean([float(r['mom_excess'].replace('%','').replace('+','')) for r in all_results])
    avg_pure = np.mean([float(r['pure_mom_excess'].replace('%','').replace('+','')) for r in all_results])
    avg_delta = avg_mom - avg_ml
    print(f'{"-"*18}-+-{"-"*5}-+-{"-"*10}-+-{"-"*13}-+-{"-"*10}-+-{"-"*9}-+-{"-"*6}-+-{"-"*6}')
    print(f'{"AVERAGE":<18} | {"":>5} | {avg_ml:>+9.2f}% | {avg_mom:>+12.2f}% | '
          f'{avg_pure:>+9.2f}% | {avg_delta:>+8.2f}% | {"":>6} | {"":>6}')

print(f'\n{"="*65}')
print('KEY FINDINGS')
print('='*65)
if avg_delta > 0:
    print(f'✅ 动量过滤平均提升超额 +{avg_delta:.2f}%')
else:
    print(f'❌ 动量过滤未提升超额 (Δ={avg_delta:.2f}%)')

if avg_ml > avg_pure:
    print(f'✅ ML选股优于纯动量策略 ({avg_ml:.2f}% vs {avg_pure:.2f}%)')
else:
    print(f'❌ 纯动量策略优于ML选股 ({avg_pure:.2f}% vs {avg_ml:.2f}%)')

# ── 8. Save results ───────────────────────────────────────────────────────
print(f'\n  Saving results...')
os.makedirs(os.path.join(ROOT, 'data', 'models'), exist_ok=True)
output = {
    'title': 'Phase 1: ML + Momentum Combined Backtest',
    'timestamp': pd.Timestamp.now().isoformat(),
    'model': os.path.basename(MODEL_PATH),
    'features': len(valid_features),
    'total_rows': len(clean),
    'results': all_results,
    'summary': {
        'avg_ml_excess': f'{avg_ml:+.2f}%',
        'avg_ml_momentum_excess': f'{avg_mom:+.2f}%',
        'avg_pure_momentum_excess': f'{avg_pure:+.2f}%',
        'avg_delta': f'{avg_delta:+.2f}%',
    }
}
out_path = os.path.join(ROOT, 'data', 'models', 'phase1_momentum_results.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f'  Saved: {out_path}')
print(f'\n  Total time: {time.time()-t0:.0f}s ({((time.time()-t0)/60):.1f}min)')
print(f'{"="*65}')
