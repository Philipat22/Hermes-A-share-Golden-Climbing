import pandas as pd, os
ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
FACTOR_DIR = os.path.join(ROOT, 'data', 'cache', 'factors_batched')
pdf = pd.read_parquet(os.path.join(FACTOR_DIR, sorted(os.listdir(FACTOR_DIR))[0]))
alpha_cols = [c for c in pdf.columns if c.startswith('alpha')]
other_cols = [c for c in pdf.columns if c.startswith(('rsi','macd','bb_','klen','rsqr','slope','std','vma','vosc'))]
print(f"Alpha factors count: {len(alpha_cols)}")
if alpha_cols:
    print(f"First: {alpha_cols[0]}, Last: {alpha_cols[-1]}")
print(f"Other factors count: {len(other_cols)}")
print(f"Total feature cols: {len(alpha_cols)+len(other_cols)}")
print(f"Other sample: {other_cols[:3]} ... {other_cols[-3:]}")
