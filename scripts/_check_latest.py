"""检查最新数据日期"""
import pickle, pandas as pd

with open(r'D:\AIHedgeFund\ai-hedge-fund-main\data\cache\prices_full.pkl', 'rb') as f:
    prices = pickle.load(f)

dates = []
for code in list(prices.keys())[:100]:
    df = prices[code]
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    dates.append(df['trade_date'].max())

print(f"Latest date across 100 stocks: {max(dates).date()}")
print(f"Total stocks: {len(prices)}")

# Check a known semiconductor
for code in ['603986.SH', '600171.SH', '002049.SZ']:
    if code in prices:
        df = prices[code]
        df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
        print(f"{code}: {len(df)} rows, latest={df['trade_date'].max().date()}")
