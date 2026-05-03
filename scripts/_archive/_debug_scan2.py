#!/usr/bin/env python3
"""Debug scanner step by step - with error logging"""
import sys, os, pandas as pd, traceback
sys.path.insert(0, r'D:\AIHedgeFund\ai-hedge-fund-main')

from src.tools.a_stock_api import get_16_sector_stocks, get_prices
from src.surge.engine import analyze_stock, classify_signal, load_params
from datetime import datetime, timedelta

pool = get_16_sector_stocks()
all_stocks = []
for s, stocks in pool.items():
    all_stocks.extend(stocks[:30])

print(f"Pool: {len(all_stocks)}")

params = load_params()
end = datetime.now().strftime('%Y-%m-%d')
start = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

all_signals = {}
detected_count = 0
error_count = 0
no_data = 0
price_filtered = 0

for i, code in enumerate(all_stocks):
    try:
        prices = get_prices(code, start, end)
        if not prices or len(prices) < 40:
            no_data += 1
            continue
        
        df = pd.DataFrame([{
            'close': p.close, 'high': p.high, 'low': p.low,
            'open': p.open, 'vol': p.volume,
        } for p in prices])
        
        close = float(df['close'].iloc[-1])
        if close < 3 or close > 200:
            price_filtered += 1
            continue
        
        signal = analyze_stock(code, df, all_signals, params)
        
        if signal["detected"]:
            detected_count += 1
            grade = classify_signal(signal, params)
            signal["signal_grade"] = grade
            signal["name"] = code
            signal["industry"] = ""
            signal["close"] = close
            all_signals[code] = signal
        
        if (i + 1) % 50 == 0:
            print(f"  progress: {i+1}/{len(all_stocks)} detected={detected_count} errors={error_count}")
        
    except Exception as e:
        error_count += 1
        if error_count <= 3:
            print(f"  ERROR [{code}]: {e}")

print(f"\nResults:")
print(f"  Total: {len(all_stocks)}")
print(f"  No data: {no_data}")
print(f"  Price filtered: {price_filtered}")
print(f"  Errors: {error_count}")
print(f"  Detected: {detected_count}")
print(f"  In all_signals: {len(all_signals)}")

if all_signals:
    grades = {}
    for s in all_signals.values():
        g = s.get('signal_grade', 'NONE')
        grades[g] = grades.get(g, 0) + 1
    print(f"  Grades: {grades}")
    
    top = sorted(all_signals.values(), key=lambda x: x.get('final_score', 0), reverse=True)[:5]
    for s in top:
        print(f"  TOP: {s['ts_code']} score={s['final_score']} type={s['pattern_type']} grade={s.get('signal_grade','')}")
