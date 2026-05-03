#!/usr/bin/env python3
"""跟踪扫描过程——看看每个阶段出了什么"""
import sys, os, pandas as pd
sys.path.insert(0, r'D:\AIHedgeFund\ai-hedge-fund-main')

from src.tools.a_stock_api import get_16_sector_stocks, get_prices
from src.surge.engine import analyze_stock, classify_signal, load_params
from datetime import datetime, timedelta

pool = get_16_sector_stocks()
all_stocks = []
for s, stocks in pool.items():
    all_stocks.extend(stocks[:30])

print(f"池子: {len(all_stocks)}")

params = load_params()
end = datetime.now().strftime('%Y-%m-%d')
start = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

# 只测试前20只，打log
all_signals = {}
hits = 0
total = 0

for i, code in enumerate(all_stocks[:30]):
    total += 1
    try:
        prices = get_prices(code, start, end)
        if not prices or len(prices) < 40:
            continue
        
        df = pd.DataFrame([{
            'close': p.close, 'high': p.high, 'low': p.low,
            'open': p.open, 'vol': p.volume,
        } for p in prices])
        
        close = float(df['close'].iloc[-1])
        if close < 3 or close > 200:
            continue
        
        signal = analyze_stock(code, df, all_signals, params)
        
        if signal["detected"]:
            hits += 1
            grade = classify_signal(signal, params)
            signal["signal_grade"] = grade
            print(f"[{i+1}] {code} DETECTED: type={signal['pattern_type']} score={signal['final_score']} grade={grade}")
            all_signals[code] = signal
        else:
            # 打印为何未检测到
            reason = ""
            if signal["pattern_type"]:
                reason = f"pattern={signal['pattern_type']}({signal['pattern_score']}) but final_score={signal['final_score']}<{params['weak_signal']}"
            else:
                reason = "no pattern detected"
            print(f"[{i+1}] {code} SKIPPED: {reason} total={signal['total_score']} final={signal['final_score']}")
        
    except Exception as e:
        import traceback
        print(f"[{i+1}] {code} ERROR: {e}")
        traceback.print_exc()

print(f"\n结果: {total} processed, {hits} detected")
print(f"all_signals 大小: {len(all_signals)}")
