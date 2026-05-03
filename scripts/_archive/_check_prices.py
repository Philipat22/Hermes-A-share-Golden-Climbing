import pandas as pd, os
ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
prices = pd.read_pickle(os.path.join(ROOT, 'data', 'cache', 'backtest_prices_extended.pkl'))
print(f"Type: {type(prices)}")
if isinstance(prices, dict):
    print(f"Keys: {list(prices.keys())[:5]}...")
    for k in list(prices.keys())[:2]:
        v = prices[k]
        print(f"  {k}: type={type(v).__name__}, shape={v.shape if hasattr(v, 'shape') else 'N/A'}")
        print(f"    columns: {list(v.columns[:10]) if hasattr(v, 'columns') else 'N/A'}")
elif isinstance(prices, list):
    print(f"Length: {len(prices)}, First: {type(prices[0])}")
else:
    print(f"First rows: {prices.head(3) if hasattr(prices, 'head') else prices}")
