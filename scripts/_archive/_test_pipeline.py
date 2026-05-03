"""Test: Run pipeline on extended data for 1 subset year, measure time."""
import sys, os, pickle, time, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# Load extended prices
print('Loading extended prices...')
with open('data/cache/backtest_prices_extended.pkl', 'rb') as f:
    prices = pickle.load(f)

# Filter to only data up to 2025-03 (first full year) to test speed
# Take a subset of stocks for speed test
stock_sample = list(prices.keys())[:50]  # 50 stocks
filtered = {}
for sc in stock_sample:
    df = prices[sc].copy()
    df = df[df['date'] <= '2025-03-31']  # limit date range
    filtered[sc] = df

print(f'Test: {len(filtered)} stocks, ~{sum(len(d) for d in filtered.values())} rows')

# Quick test: Convert to Polars format and try factor computation
import polars as pl
from src.features.feature_generator import FeatureGenerator

# Add required columns (turnover -> amount, adj_factor = 1)
for sc in filtered:
    df = filtered[sc]
    df = df.rename(columns={'turnover': 'amount'})
    df['adj_factor'] = 1.0
    df['date'] = pd.to_datetime(df['date'])
    filtered[sc] = df

t0 = time.time()

# Convert all stocks to one Polars DataFrame
frames = []
for code, pdf in filtered.items():
    if pdf.empty or len(pdf) < 80:
        continue
    pldf = pl.from_pandas(pdf.reset_index(drop=True))
    rename = {}
    for col in ['date', 'Date']:
        if col in pldf.columns:
            rename[col] = 'datetime'
    pldf = pldf.rename(rename) if rename else pldf
    if 'vt_symbol' not in pldf.columns:
        pldf = pldf.with_columns(pl.lit(code).alias('vt_symbol'))
    frames.append(pldf)

polars_df = pl.concat(frames)
print(f'Polars: {polars_df.shape[0]} rows, {polars_df["vt_symbol"].n_unique()} stocks')
t1 = time.time()
print(f'  Convert: {t1-t0:.1f}s')

# Compute factors (sequential, low memory)
fg = FeatureGenerator(max_workers=1)
factor_df = fg.compute_all(polars_df)
t2 = time.time()
print(f'  Factors: {t2-t1:.1f}s')

base_cols = {'datetime', 'vt_symbol', 'open', 'high', 'low', 'close',
              'volume', 'amount', 'adj_factor', 'vwap'}
factor_cols = [c for c in factor_df.columns if c not in base_cols]
print(f'  {len(factor_cols)} factors computed')
print(f'  Total: {t2-t0:.1f}s')

# Memory/logging
print(f'\nSample factor names: {factor_cols[:5]}')
