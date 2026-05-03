"""
Full Factor Generation + Backtest Pipeline v2
-----------------------------------------------
Loads extended price data (2019-2026, 497 stocks),
processes stocks in batches of 50 to avoid memory/slowdown issues,
scores with trained LightGBM + XGBoost ensemble,
runs trade-by-trade backtest.

Usage: python scripts/run_full_backtest_v2.py
"""
import sys, os, pickle, time, json, gc, warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'
ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
import polars as pl
import lightgbm as lgb
import xgboost as xgb

# ── Config ────────────────────────────────────────────────────────────────
BATCH_SIZE = 50
with open('src/surge/params.json') as f:
    params = json.load(f)
TOP_40 = params['selected_features']
ens_threshold = params.get('ensemble', {}).get('threshold', 0.40)

print('=' * 60)
print(f'Top-40 features: {len(TOP_40)}')
print(f'Ensemble threshold: {ens_threshold}')
print('=' * 60)

# ── 1. Load extended price data ───────────────────────────────────────────
print('\n[1/6] Loading extended price data...')
t0 = time.time()
with open('data/cache/backtest_prices_extended.pkl', 'rb') as f:
    prices = pickle.load(f)

# Standardize columns
for sc in list(prices.keys()):
    df = prices[sc].copy()
    if 'turnover' in df.columns:
        df = df.rename(columns={'turnover': 'amount'})
    if 'amount' not in df.columns:
        df['amount'] = 0.0
    if 'adj_factor' not in df.columns:
        df['adj_factor'] = 1.0
    df['date'] = pd.to_datetime(df['date'])
    prices[sc] = df.sort_values('date').reset_index(drop=True)

stock_codes = sorted(prices.keys())
print(f'  Stocks: {len(stock_codes)}, Rows: {sum(len(d) for d in prices.values()):,}')
print(f'  Time: {time.time()-t0:.1f}s')

# ── 2. Compute factors in batches ─────────────────────────────────────────
print('\n[2/6] Computing factors in batches...')
from src.features.feature_generator import FeatureGenerator

GENERATED_DIR = os.path.join(ROOT, 'data', 'cache', 'factors_batched')
os.makedirs(GENERATED_DIR, exist_ok=True)
fg = FeatureGenerator(max_workers=1)

existing_batches = set()
for fn in os.listdir(GENERATED_DIR):
    if fn.endswith('.parquet'):
        existing_batches.add(fn)

batches = [stock_codes[i:i+BATCH_SIZE] for i in range(0, len(stock_codes), BATCH_SIZE)]
print(f'  {len(batches)} batches of {BATCH_SIZE} stocks each')

for batch_idx, batch_stocks in enumerate(batches):
    batch_name = f'batch_{batch_idx:03d}'
    batch_file = f'{batch_name}.parquet'
    
    if batch_file in existing_batches:
        print(f'  Batch {batch_idx+1}/{len(batches)}: SKIP (already cached)')
        continue
    
    t_batch = time.time()
    
    # Build Polars DF for this batch
    frames = []
    for code in batch_stocks:
        pdf = prices[code]
        if len(pdf) < 80:
            continue
        pldf = pl.from_pandas(pdf)
        pldf = pldf.rename({'date': 'datetime'})
        pldf = pldf.with_columns(pl.lit(code).alias('vt_symbol'))
        frames.append(pldf)
    
    batch_polars = pl.concat(frames)
    n_rows = len(batch_polars)
    print(f'  Batch {batch_idx+1}/{len(batches)}: {len(batch_stocks)} stocks, {n_rows:,} rows...', end=' ')
    
    try:
        result = fg.compute_all(batch_polars)
        # Save only factor columns + identification
        base_cols = {'datetime', 'vt_symbol', 'open', 'high', 'low', 'close',
                     'volume', 'amount', 'adj_factor', 'vwap'}
        factor_cols = [c for c in result.columns if c not in base_cols]
        
        # Save full result (we need closes for label computation)
        out_path = os.path.join(GENERATED_DIR, batch_file)
        result.write_parquet(out_path)
        
        elapsed = time.time() - t_batch
        print(f'OK ({elapsed:.0f}s, {len(factor_cols)} factors)')
        
        # Force garbage collection
        del batch_polars, result, frames
        gc.collect()
        
    except Exception as e:
        print(f'FAILED: {e}')
        # Try stock-by-stock fallback
        print(f'    Retrying stock-by-stock...')
        for code in batch_stocks:
            try:
                pdf = prices[code]
                if len(pdf) < 80:
                    continue
                pldf = pl.from_pandas(pdf).rename({'date': 'datetime'})
                pldf = pldf.with_columns(pl.lit(code).alias('vt_symbol'))
                result = fg.compute_all(pldf)
                
                single_file = os.path.join(GENERATED_DIR, f'single_{code}.parquet')
                result.write_parquet(single_file)
                
                del pldf, result
                gc.collect()
            except Exception as e2:
                print(f'      {code}: {e2}')

print(f'\n  Factor computation total: {time.time()-t0:.0f}s')

# ── 3. Load all factors, create labels, score ─────────────────────────────
print('\n[3/6] Loading factors and scoring...')
t3 = time.time()

# Load all parquet files
all_dfs = []
for fn in sorted(os.listdir(GENERATED_DIR)):
    if fn.endswith('.parquet'):
        df = pd.read_parquet(os.path.join(GENERATED_DIR, fn))
        all_dfs.append(df)
        print(f'  Loaded {fn}: {len(df)} rows')

if not all_dfs:
    print('  ERROR: No factor data computed!')
    sys.exit(1)

pdf = pd.concat(all_dfs, ignore_index=True)
print(f'  Total: {len(pdf):,} rows')

# Create labels (20-day forward return)
HORIZON = 20
pdf['forward_ret'] = np.nan
pdf = pdf.sort_values(['vt_symbol', 'datetime'])

for name, group in pdf.groupby('vt_symbol', sort=False):
    idx = group.index
    closes = group['close'].values
    rets = np.full(len(group), np.nan)
    for i in range(len(group)):
        j = i + HORIZON
        if j < len(group):
            rets[i] = (closes[j] - closes[i]) / closes[i]
    pdf.loc[idx, 'forward_ret'] = rets

pdf['label'] = (pdf['forward_ret'] >= 0.10).astype(int)

print(f'  Forward ret available: {pdf["forward_ret"].notna().sum():,} ({pdf["forward_ret"].notna().mean():.0%})')
print(f'  Surge rate: {pdf["label"].mean():.2%}')
print(f'  Time: {time.time()-t3:.1f}s')

# ── 4. Load models ────────────────────────────────────────────────────────
print('\n[4/6] Loading models...')
with open('data/models/surge_lgbm.pkl', 'rb') as f:
    lgb_model = pickle.load(f)
xgb_model = xgb.Booster()
xgb_model.load_model('data/models/surge_xgboost.json')
print(f'  LightGBM: {type(lgb_model).__name__}')
print(f'  XGBoost: loaded')

# ── 5. Score all rows ─────────────────────────────────────────────────────
print('\n[5/6] Scoring...')
t5 = time.time()

valid_features = [f for f in TOP_40 if f in pdf.columns]
missing = [f for f in TOP_40 if f not in pdf.columns]
if missing:
    print(f'  WARNING: Missing factors: {missing}')

# Filter to rows with all features + valid forward_ret
score_mask = (
    pdf[valid_features].notna().all(axis=1) &
    pdf['forward_ret'].notna()
)
score_df = pdf[score_mask].copy()
print(f'  Scoreable rows: {len(score_df):,} / {len(pdf):,}')

# Score in chunks to avoid memory issues
X_all = score_df[valid_features].astype(np.float32).values
n_samples = len(X_all)

# Handle inf/outliers: XGBoost crashes on inf and very large values
X_all = np.clip(X_all, -1e10, 1e10)  # Cap extreme values
X_all = np.where(np.isinf(X_all), np.nan, X_all)  # inf→NaN

# Drop rows with any NaN after cleaning
valid_row_mask = ~np.isnan(X_all).any(axis=1)
if valid_row_mask.sum() < n_samples:
    dropped = n_samples - valid_row_mask.sum()
    print(f'  Dropping {dropped:,} rows ({dropped/n_samples*100:.1f}%) with extreme values')
    score_df = score_df.iloc[np.where(valid_row_mask)[0]].copy()
    X_all = X_all[valid_row_mask]
    n_samples = valid_row_mask.sum()

CHUNK = 100000
lgb_scores = np.zeros(n_samples, dtype=np.float32)
xgb_scores = np.zeros(n_samples, dtype=np.float32)
print(f'  Scoring {n_samples:,} rows in chunks of {CHUNK:,}...')

for start in range(0, n_samples, CHUNK):
    end = min(start + CHUNK, n_samples)
    X_chunk = np.ascontiguousarray(X_all[start:end])
    lgb_scores[start:end] = lgb_model.predict(X_chunk, raw_score=False)
    xgb_scores[start:end] = xgb_model.predict(xgb.DMatrix(X_chunk, missing=np.nan))
    if (start // CHUNK) % 5 == 0:
        print(f'  Scored {end:,}/{n_samples:,}')

ensemble_scores = (lgb_scores + xgb_scores) / 2
score_df['lgb_score'] = lgb_scores
score_df['xgb_score'] = xgb_scores
score_df['ensemble_score'] = ensemble_scores

print(f'  Score range: {ensemble_scores.min():.4f} ~ {ensemble_scores.max():.4f}')
print(f'  Score percentiles: 50%={np.percentile(ensemble_scores,50):.4f}, '
      f'75%={np.percentile(ensemble_scores,75):.4f}, '
      f'90%={np.percentile(ensemble_scores,90):.4f}, '
      f'95%={np.percentile(ensemble_scores,95):.4f}, '
      f'99%={np.percentile(ensemble_scores,99):.4f}')
print(f'  Time: {time.time()-t5:.1f}s')

# ── 6. Multi-threshold backtest ──────────────────────────────────────────
print(f'\n[6/6] Signal backtest (multiple thresholds)...')
t6 = time.time()

results = []
for sc in score_df['vt_symbol'].unique():
    sub = score_df[score_df['vt_symbol'] == sc].sort_values('datetime').copy()
    for _, row in sub.iterrows():
        results.append({
            'stock': sc,
            'entry_date': row['datetime'],
            'forward_ret': row['forward_ret'],
            'score': row['ensemble_score'],
            'lgb_score': row['lgb_score'],
            'xgb_score': row['xgb_score'],
            'label': row['label'],
        })

all_trades = pd.DataFrame(results)
print(f'  Total samples: {len(all_trades):,}')

# Score distribution analysis
max_score = float(ensemble_scores.max())
thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]
if max_score > 0.30:
    thresholds += [0.32, 0.34, 0.35]
if max_score > 0.38:
    thresholds += [0.40]  # original training threshold
thresholds = sorted(set([t for t in thresholds if t <= max_score]))
print(f'\n  {"Threshold":>10} | {"Signals":>8} | {"Win%":>7} | {"AvgRet":>10} | {"Excess":>10}')
print(f'  {"-"*10}-+-{"-"*8}-+-{"-"*7}-+-{"-"*10}-+-{"-"*10}')

best_th, best_excess = 0, -999
all_summaries = {}
for th in thresholds:
    picks = all_trades[all_trades['score'] >= th]
    if len(picks) < 5:
        continue
    wr = picks['forward_ret'].mean() >= 0
    wr_val = (picks['forward_ret'] > 0).mean()
    avg_ret = picks['forward_ret'].mean()
    market_avg = all_trades['forward_ret'].mean()
    excess = avg_ret - market_avg
    print(f'  ≥{th:.3f}    | {len(picks):>8,} | {wr_val*100:>6.1f}% | {avg_ret*100:>+9.2f}% | {excess*100:>+9.2f}%')
    all_summaries[th] = {'signals': len(picks), 'win_rate': float(wr_val), 'avg_ret': float(avg_ret), 'excess': float(excess)}
    if excess > best_excess:
        best_excess, best_th = excess, th

print(f'\n  >> Best threshold: {best_th:.3f} (excess={best_excess*100:+.2f}%)')

# Detailed report at best threshold
print(f'\n  Detailed: threshold={best_th:.3f}')
best_picks = all_trades[all_trades['score'] >= best_th].copy()
best_picks['year'] = pd.to_datetime(best_picks['entry_date']).dt.year

print(f'    Total: {len(best_picks):,} signals')
print(f'    Win rate: {best_picks["forward_ret"].gt(0).mean():.1%}')
print(f'    Avg ret: {best_picks["forward_ret"].mean():.2%}')
cum = (1 + best_picks['forward_ret']).prod() - 1
print(f'    Cumulative: {cum:.2%}')
print(f'    Max DD: {best_picks["forward_ret"].min():.2%}')
sr = best_picks['forward_ret'].mean() / best_picks['forward_ret'].std() * np.sqrt(12) if best_picks['forward_ret'].std() > 0 else 0
print(f'    Sharpe(12): {sr:.2f}')

print(f'    By year:')
for yr in sorted(best_picks['year'].unique()):
    yr_data = best_picks[best_picks['year'] == yr]
    print(f'      {int(yr)}: {len(yr_data):,} sigs, WR={yr_data["forward_ret"].gt(0).mean():.1%}, '
          f'Avg={yr_data["forward_ret"].mean():.2%}')

# Save all
all_trades.to_csv('data/models/backtest_all_signals.csv', index=False)
best_picks.to_csv('data/models/backtest_best_signals.csv', index=False)
summary = {
    'score_distribution': {
        'min': float(ensemble_scores.min()),
        'max': float(ensemble_scores.max()),
        'p50': float(np.percentile(ensemble_scores, 50)),
        'p75': float(np.percentile(ensemble_scores, 75)),
        'p90': float(np.percentile(ensemble_scores, 90)),
        'p95': float(np.percentile(ensemble_scores, 95)),
        'p99': float(np.percentile(ensemble_scores, 99)),
    },
    'best_threshold': float(best_th),
    'best_summary': {
        'signals': len(best_picks),
        'win_rate': float(best_picks['forward_ret'].gt(0).mean()),
        'avg_return': float(best_picks['forward_ret'].mean()),
        'cumulative': float(cum),
        'max_drawdown': float(best_picks['forward_ret'].min()),
        'sharpe': float(sr),
    },
    'all_thresholds': all_summaries,
}
with open('data/models/backtest_full_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f'\n  Saved:')
print(f'    data/models/backtest_all_signals.csv ({len(all_trades):,} rows)')
print(f'    data/models/backtest_best_signals.csv ({len(best_picks):,} rows)')
print(f'    data/models/backtest_full_summary.json')

total_time = time.time() - t0
print(f'\nTotal: {total_time:.0f}s ({total_time/60:.1f} min)')
print('DONE')
