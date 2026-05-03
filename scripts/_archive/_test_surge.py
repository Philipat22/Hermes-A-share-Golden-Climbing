#!/usr/bin/env python3
"""简明测试 Surge 引擎"""
import sys, os
sys.path.insert(0, r'D:\AIHedgeFund\ai-hedge-fund-main')
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.tools.a_stock_api import get_prices
from src.surge.engine import analyze_stock, classify_signal, load_params
import pandas as pd
from datetime import datetime, timedelta

test_codes = ['603501.SH', '002245.SZ', '002371.SZ']
end = datetime.now().strftime('%Y-%m-%d')
start = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

print("SURGE 引擎测试")
for code in test_codes:
    try:
        prices = get_prices(code, start, end)
        if not prices or len(prices) < 40:
            print(f"[{code}] 数据不足"); continue
        df = pd.DataFrame([{
            'close': p.close, 'high': p.high, 'low': p.low,
            'open': p.open, 'vol': p.volume,
        } for p in prices])
        signal = analyze_stock(code, df, {})
        grade = classify_signal(signal)
        print(f"[{code}] 价格={df.close.iloc[-1]:.0f} | "
              f"总分={signal['total_score']:3d} 扣={signal['fake_score']:2d} "
              f"最终={signal['final_score']:3d} | "
              f"形态={signal['pattern_type'] or '-':>6s}({signal['pattern_score']}) | "
              f"量={signal['volume_score']:2d} 加速={signal['accel_score']:2d} "
              f"板块={signal['sector_score']:2d} | "
              f"等级={grade}")
    except Exception as e:
        print(f"[{code}] ERROR: {e}")

print("=== 测试完成 ===")
