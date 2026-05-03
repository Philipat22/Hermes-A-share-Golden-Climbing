"""Step 1: Download extended price data from Tushare and combine with existing cache.
Covers 2019-01-01 to 2026-04-29 for all 497 stocks."""
import sys, os, pickle, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# Load existing prices to get stock list
print('Loading existing backtest_prices.pkl...')
with open('data/cache/backtest_prices.pkl', 'rb') as f:
    prices_old = pickle.load(f)

stock_codes = sorted(prices_old.keys())
print(f'Stocks: {len(stock_codes)}')

# Check date range of existing data
min_dates = []
for sc in stock_codes:
    d = prices_old[sc]['date'].min()
    min_dates.append(str(d)[:10])
old_start = min(min_dates)
print(f'Existing data starts: {old_start}')

# Download older data from Tushare
import tushare as ts
from concurrent.futures import ThreadPoolExecutor, as_completed

TOKEN = '5243de737c1a25110583352fde4458266314877dd0c342cae1a9f4c7'
pro = ts.pro_api(TOKEN)

OUT_PATH = 'data/cache/backtest_prices_extended.pkl'
NEW_START = '20190101'
CUTOFF = '20241008'  # download up to 2024-10-08 (existing data starts 2024-10-08)

def fetch_stock(ts_code):
    """Download daily data for one stock from 2019 to 2024-10."""
    try:
        df = pro.daily(ts_code=ts_code, start_date=NEW_START, end_date=CUTOFF)
        if df is None or len(df) == 0:
            return None
        # Standardize: rename columns, sort by date
        df = df.rename(columns={
            'trade_date': 'date', 'vol': 'volume', 'amount': 'turnover'
        })
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        # Keep only needed columns
        cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        for c in cols:
            if c not in df.columns:
                return None
        df = df[cols]
        return ts_code, df
    except Exception as e:
        print(f'  FAIL: {ts_code} - {e}')
        return None

print(f'\nDownloading {len(stock_codes)} stocks from Tushare (2019-01 to 2024-10)...')
t0 = time.time()
new_prices = {}  # {ts_code: df}

# Parallel download
with ThreadPoolExecutor(max_workers=10) as pool:
    futures = {pool.submit(fetch_stock, sc): sc for sc in stock_codes}
    done, n = 0, len(stock_codes)
    for f in as_completed(futures):
        done += 1
        result = f.result()
        if result is not None:
            sc, df = result
            new_prices[sc] = df
        if done % 50 == 0:
            elapsed = time.time() - t0
            print(f'  {done}/{n} stocks fetched ({elapsed:.0f}s)')

t1 = time.time()
print(f'Download complete: {len(new_prices)}/{n} stocks in {t1-t0:.0f}s')

# Combine with existing data
print('\nMerging with existing data...')
merged = {}
for sc in stock_codes:
    old_df = prices_old[sc].copy()
    old_df['date'] = pd.to_datetime(old_df['date'])
    if sc in new_prices:
        new_df = new_prices[sc].copy()
        new_df['date'] = pd.to_datetime(new_df['date'])
        combined = pd.concat([new_df, old_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date'], keep='last')
        combined = combined.sort_values('date').reset_index(drop=True)
    else:
        combined = old_df.copy()
    merged[sc] = combined

# Check date range
all_dates = set()
for sc, df in merged.items():
    for d in df['date']:
        all_dates.add(str(d)[:10])
dates_sorted = sorted(all_dates)
print(f'Merged date range: {dates_sorted[0]} ~ {dates_sorted[-1]}')
print(f'Total trading days: {len(dates_sorted)}')
print(f'Total rows across all stocks: {sum(len(df) for df in merged.values())}')

# Save
with open(OUT_PATH, 'wb') as f:
    pickle.dump(merged, f)
print(f'\nSaved: {OUT_PATH}')
