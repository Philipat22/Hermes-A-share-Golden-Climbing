import pandas as pd, os
ROOT = r'D:\AIHedgeFund\ai-hedge-fund-main'
price_dict = pd.read_pickle(os.path.join(ROOT, 'data', 'cache', 'backtest_prices_extended.pkl'))
sample = list(price_dict.values())[0]
print(f"date type: {type(sample['date'].iloc[0])}")
print(f"First 3 dates: {list(sample['date'].head(3))}")
print(f"open type: {type(sample['open'].iloc[0])}")
print(f"Sample stock: {len(price_dict)} stocks, ~{len(sample)} rows each")
