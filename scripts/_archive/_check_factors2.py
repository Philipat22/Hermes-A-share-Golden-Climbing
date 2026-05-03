import pandas as pd, os
ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
FACTOR_DIR = os.path.join(ROOT, 'data', 'cache', 'factors_batched')
pdf = pd.read_parquet(os.path.join(FACTOR_DIR, sorted(os.listdir(FACTOR_DIR))[0]))
print(f"Columns: {list(pdf.columns[:20])}...")
print(f"Has 'close': {'close' in pdf.columns}")
print(f"Has 'close_': {'close_' in pdf.columns}")
print(f"Has 'datetime': {'datetime' in pdf.columns}")
print(f"First row datetime: {pdf['datetime'].iloc[0] if 'datetime' in pdf.columns else 'N/A'}")
print(f"Type of datetime: {type(pdf['datetime'].iloc[0]) if 'datetime' in pdf.columns else 'N/A'}")
