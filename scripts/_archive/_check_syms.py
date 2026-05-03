import pandas as pd, os
ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
FACTOR_DIR = os.path.join(ROOT, 'data', 'cache', 'factors_batched')
pdf = pd.read_parquet(os.path.join(FACTOR_DIR, sorted(os.listdir(FACTOR_DIR))[0]))
price_dict = pd.read_pickle(os.path.join(ROOT, 'data', 'cache', 'backtest_prices_extended.pkl'))

pdf_syms = set(pdf['vt_symbol'].unique())
price_syms = set(price_dict.keys())

common = pdf_syms & price_syms
print(f"PDF symbols: {len(pdf_syms)}")
print(f"Price symbols: {len(price_syms)}")
print(f"Common: {len(common)}")
print(f"PDF only: {list(pdf_syms - price_syms)[:5]}")
print(f"Price only: {list(price_syms - pdf_syms)[:5]}")
print(f"Sample PDF sym: {sorted(pdf_syms)[0]}")
print(f"Sample Price sym: {sorted(price_syms)[0]}")
