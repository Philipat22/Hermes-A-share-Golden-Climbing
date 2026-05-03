#!/usr/bin/env python3
"""Debug scanner - 看看为什么没信号"""
import sys, os, pandas as pd
sys.path.insert(0, r'D:\AIHedgeFund\ai-hedge-fund-main')

from src.tools.a_stock_api import get_16_sector_stocks, get_prices
from datetime import datetime, timedelta

# 展平
pool = get_16_sector_stocks()
all_stocks = []
for s, stocks in pool.items():
    all_stocks.extend(stocks[:30])

print(f"Total pool: {len(all_stocks)}")

# 测试前20只看看什么情况
end = datetime.now().strftime('%Y-%m-%d')
start = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

good = 0
bad_data = 0
bad_price = 0
errors = 0

for i, code in enumerate(all_stocks[:50]):
    try:
        prices = get_prices(code, start, end)
        if not prices or len(prices) < 40:
            bad_data += 1
            if i < 5: print(f"[{code}] data: {len(prices) if prices else 0}")
            continue
        
        df = pd.DataFrame([{'close': p.close, 'vol': p.volume} for p in prices])
        close = float(df['close'].iloc[-1])
        
        if close < 3 or close > 200:
            bad_price += 1
            if i < 5: print(f"[{code}] price={close:.2f} filtered")
            continue
        
        good += 1
        if i < 5: print(f"[{code}] OK: price={close:.2f} days={len(prices)}")
        
    except Exception as e:
        errors += 1
        if i < 5: print(f"[{code}] ERROR: {e}")

print(f"\n前50只: good={good} bad_data={bad_data} bad_price={bad_price} errors={errors}")
