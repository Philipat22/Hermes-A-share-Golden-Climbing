#!/usr/bin/env python3
"""Quick diagnostic of current split state"""
import sys, os
sys.path.insert(0, r'D:\AIHedgeFund\ai-hedge-fund-main')

results = []

# 1. Check data_classes
try:
    from src.tools.data_classes import FinancialMetrics, FinancialLineItem, PriceBar, NewsItem, MarginTrade, LimitUpData, AStockInfo
    results.append(f"[OK] data_classes: all 7 classes importable")
    # Check if classes have __init__
    import inspect
    for cls in [FinancialMetrics, FinancialLineItem, PriceBar]:
        src = inspect.getsource(cls)
        lines = src.count('\n') + 1
        results.append(f"     {cls.__name__}: ~{lines} lines in source")
except ImportError as e:
    results.append(f"[MISSING] data_classes: {e}")
except Exception as e:
    results.append(f"[ERR] data_classes: {e}")

# 2. Check data_cleaner
try:
    from src.tools.data_cleaner import _safe_float
    results.append(f"[OK] data_cleaner: _safe_float importable")
except ImportError as e:
    results.append(f"[MISSING] data_cleaner: {e}")

# 3. Check data_fetcher
try:
    from src.tools.data_fetcher import get_market_context, TUSHARE_TOKEN, SECTOR_POOL
    results.append(f"[OK] data_fetcher: {len(SECTOR_POOL)} sectors, TOKEN={TUSHARE_TOKEN[:8]}...")
except ImportError as e:
    results.append(f"[MISSING] data_fetcher: {e}")

# 4. Check facade
try:
    from src.tools.a_stock_api import get_market_context
    results.append(f"[OK] a_stock_api facade: backward-compatible exports OK")
except ImportError as e:
    results.append(f"[MISSING] a_stock_api: {e}")

# 5. File sizes
import os
for fname in ['data_classes.py', 'data_cleaner.py', 'data_fetcher.py', 'a_stock_api.py']:
    fpath = rf'D:\AIHedgeFund\ai-hedge-fund-main\src\tools\{fname}'
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        results.append(f"[FILE] {fname}: {size:,} bytes")
    else:
        results.append(f"[FILE] {fname}: NOT FOUND")

for r in results:
    print(r)
