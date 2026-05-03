"""Debug column widths across stocks"""
import pickle, pandas as pd

with open('data/cache/backtest_prices_extended.pkl', 'rb') as f:
    prices = pickle.load(f)

widths = {}
for sc, df in prices.items():
    cols = tuple(sorted(df.columns))
    width = len(cols)
    if width not in widths:
        widths[width] = {'count': 0, 'example': sc, 'columns': cols}
    widths[width]['count'] += 1

print('Column width distribution:')
for w, info in sorted(widths.items()):
    print(f'  {w} cols ({info["count"]} stocks): {info["columns"]}')
