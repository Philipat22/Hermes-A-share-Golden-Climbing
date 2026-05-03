"""
Full Factor Generation + Backtest Pipeline
-------------------------------------------
Loads extended price data (2019-2026, 497 stocks, ~800k rows),
computes all 160+ alpha factors, creates 20d forward-return labels,
scores with trained LightGBM ensemble model,
runs trade-by-trade backtest.

Usage: python scripts/run_full_backtest.py
"""
import sys, os, pickle, time, json, warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'

ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
import polars as pl

# ----------------------------------------------------------------
# 0. Load configuration
# ----------------------------------------------------------------
with open('src/surge/params.json', 'r') as f:
    params = json.load(f)
TOP_40 = params['selected_features']
THRESHOLD = params.get('ml_threshold', 0.40)
ens_threshold = params.get('ensemble', {}).get('threshold', 0.40)

print('=' * 60)
print(f'Top-40 features: {len(TOP_40)}')
print(f'ML threshold: {THRESHOLD}')
print(f'Ensemble threshold: {ens_threshold}')
print('=' * 60)

# ----------------------------------------------------------------
# 1. Load extended price data
# ----------------------------------------------------------------
print('\n[1/6] Loading extended price data...')
t0 = time.time()
with open('data/cache/backtest_prices_extended.pkl', 'rb') as f:
    prices = pickle.load(f)

# Process: rename columns, add adj_factor, sort
for sc in prices:
    df = prices[sc].copy()
    # Standardize: ensure all required columns exist
    if 'turnover' in df.columns:
        df = df.rename(columns={'turnover': 'amount'})
    if 'amount' not in df.columns:
        df['amount'] = 0.0
    if 'adj_factor' not in df.columns:
        df['adj_factor'] = 1.0
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    prices[sc] = df

stock_codes = sorted(prices.keys())
print(f'  Stocks: {len(stock_codes)}')
total_rows = sum(len(d) for d in prices.values())
print(f'  Rows: {total_rows:,}')
print(f'  Time: {time.time()-t0:.1f}s')

# ----------------------------------------------------------------
# 2. Build Polars DataFrame + Compute Factors
# ----------------------------------------------------------------
print('\n[2/6] Building Polars DataFrame...')
t0 = time.time()
frames = []
for code in stock_codes:
    pdf = prices[code]
    if pdf.empty or len(pdf) < 80:
        continue
    pldf = pl.from_pandas(pdf.reset_index(drop=True))
    rename = {c: 'datetime' for c in ['date', 'Date'] if c in pldf.columns}
    pldf = pldf.rename(rename) if rename else pldf
    if 'vt_symbol' not in pldf.columns:
        pldf = pldf.with_columns(pl.lit(code).alias('vt_symbol'))
    frames.append(pldf)

polars_df = pl.concat(frames)
print(f'  Polars: {polars_df.shape[0]:,} rows, {polars_df["vt_symbol"].n_unique()} stocks')
print(f'  Columns: {list(polars_df.columns)}')
print(f'  Time: {time.time()-t0:.1f}s')

# Compute factors
print('\n[3/6] Computing 160+ Alpha factors...')
t1 = time.time()
from src.features.feature_generator import FeatureGenerator
fg = FeatureGenerator(max_workers=1)  # Sequential for stability on Windows
factor_df = fg.compute_all(polars_df)
base_cols = {'datetime', 'vt_symbol', 'open', 'high', 'low', 'close',
             'volume', 'amount', 'adj_factor', 'vwap'}
all_factor_cols = [c for c in factor_df.columns if c not in base_cols]
print(f'  Factors computed: {len(all_factor_cols)}')
print(f'  Time: {time.time()-t1:.1f}s')

# Verify top-40 are available
missing = [f for f in TOP_40 if f not in factor_df.columns]
if missing:
    print(f'  WARNING: Missing top-40 factors: {missing}')

# ----------------------------------------------------------------
# 3. Create labels (20-day forward return)
# ----------------------------------------------------------------
print('\n[4/6] Creating 20-day forward return labels...')
t2 = time.time()
pdf = factor_df.to_pandas()
pdf['forward_ret'] = np.nan
pdf = pdf.sort_values(['vt_symbol', 'datetime'])

HORIZON = 20
for _, group in pdf.groupby('vt_symbol'):
    closes = group['close'].values
    rets = np.full(len(group), np.nan)
    for i in range(len(group)):
        j = i + HORIZON
        if j < len(group):
            rets[i] = (closes[j] - closes[i]) / closes[i]
    pdf.loc[group.index, 'forward_ret'] = rets

# Binary label
pdf['label'] = (pdf['forward_ret'] >= 0.10).astype(int)
print(f'  Total samples: {len(pdf):,}')
print(f'  Surge rate: {pdf["label"].mean():.2%}')
print(f'  Time: {time.time()-t2:.1f}s')

# ----------------------------------------------------------------
# 4. Score with trained model ensemble
# ----------------------------------------------------------------
print('\n[5/6] Scoring with trained ensemble...')
t3 = time.time()

# Load models
import lightgbm as lgb
import xgboost as xgb
import pickle

with open('data/models/surge_lgbm.pkl', 'rb') as f:
    lgb_model = pickle.load(f)

xgb_model = xgb.Booster()
xgb_model.load_model('data/models/surge_xgboost.json')

# Prepare features - ensure exactly top-40 ordered
valid_columns = [f for f in TOP_40 if f in pdf.columns]
if len(valid_columns) < len(TOP_40):
    print(f'  Only {len(valid_components)}/{len(TOP_40)} top factors available')

# Score only rows that have ALL needed factors
score_mask = pdf[valid_columns].notna().all(axis=1)
score_df = pdf[score_mask].copy()

X_score = score_df[valid_columns].astype(np.float32).values

# LightGBM score (probability)
lgb_score = lgb_model.predict(X_score, raw_score=False)
# XGBoost score (need DMatrix for Booster API)
xgb_dmatrix = xgb.DMatrix(X_score)
xgb_score = xgb_model.predict(xgb_dmatrix)

# Ensemble (average)
ensemble_score = (lgb_score + xgb_score) / 2

score_df['lgb_score'] = lgb_score
score_df['xgb_score'] = xgb_score
score_df['ensemble_score'] = ensemble_score

print(f'  Scored samples: {len(score_df):,}')
print(f'  Score range: {ensemble_score.min():.4f} ~ {ensemble_score.max():.4f}')

# At various thresholds
for th in [0.30, 0.35, 0.40, 0.45, 0.50]:
    n_picks = (ensemble_score >= th).sum()
    print(f'  Threshold {th:.2f}: {n_picks} picks ({n_picks/len(X_score)*100:.1f}%)')
print(f'  Time: {time.time()-t3:.1f}s')

# ----------------------------------------------------------------
# 5. Backtest
# ----------------------------------------------------------------
print('\n[6/6] Running backtest...')
t4 = time.time()

# Build portfolio: at each date, hold stocks with score >= threshold for 20 days
# Track PnL
results = []
for sc in score_df['vt_symbol'].unique():
    sub = score_df[score_df['vt_symbol'] == sc].sort_values('datetime').copy()
    for idx, row in sub.iterrows():
        if row['ensemble_score'] >= ens_threshold and not pd.isna(row['forward_ret']):
            entry_date = row['datetime']
            forward_ret = row['forward_ret']
            results.append({
                'stock': sc,
                'entry_date': entry_date,
                'forward_ret': forward_ret,
                'score': row['ensemble_score'],
                'label': row['label'],
            })

trades = pd.DataFrame(results)
print(f'  Total trades: {len(trades)}')

if len(trades) > 0:
    trades['win'] = trades['forward_ret'] >= 0
    win_rate = trades['win'].mean()
    avg_ret = trades['forward_ret'].mean()
    cum_pnl = (1 + trades['forward_ret']).prod() - 1
    max_dd = trades['forward_ret'].min() if len(trades) > 0 else 0

    print(f'  Win rate: {win_rate:.1%}')
    print(f'  Avg return: {avg_ret:.2%}')
    print(f'  Cumulative PnL: {cum_pnl:.2%}')
    print(f'  Max drawdown: {max_dd:.2%}')

    # By year
    trades['year'] = pd.to_datetime(trades['entry_date']).dt.year
    print(f'\n  By year:')
    for yr in sorted(trades['year'].unique()):
        yr_data = trades[trades['year'] == yr]
        print(f'    {yr}: {len(yr_data)} trades, WR={yr_data["win"].mean():.1%}, '
              f'AvgRet={yr_data["forward_ret"].mean():.1%}')

    # Save results
    trades.to_csv('data/models/backtest_full_results.csv', index=False)
    print(f'\n  Saved: data/models/backtest_full_results.csv')
else:
    print('  WARNING: No trades!')

print(f'\nTotal time: {time.time()-t0:.1f}s ({time.time()-t0:.1f}/60 min)')
print('DONE')
