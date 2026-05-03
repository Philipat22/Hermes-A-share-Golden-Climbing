#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"D:\AIHedgeFund\ai-hedge-fund-main")

print("[1] Testing data_classes import...")
from src.tools.data_classes import FinancialMetrics, FinancialLineItem, PriceBar, NewsItem
print("  FinancialMetrics OK")
print("  FinancialLineItem OK")
print("  PriceBar OK")
print("  NewsItem OK")

print("[2] Testing data_cleaner import...")
from src.tools.data_cleaner import _safe_float
result = _safe_float("3.14159")
print(f"  _safe_float('3.14159') = {result}  OK")

print("[3] Testing data_fetcher import (no network)...")
from src.tools.data_fetcher import get_pro_api, normalize_ts_code
print(f"  get_pro_api callable: {callable(get_pro_api)}")
print(f"  normalize_ts_code('600519') = {normalize_ts_code('600519')}  OK")

print("[4] Testing backward compat (a_stock_api.py)...")
from src.tools.a_stock_api import (
    FinancialMetrics, PriceBar, get_prices, get_financial_metrics,
    filter_st_stocks, get_market_cap, get_market_context
)
print("  All backward compat imports OK")

print("\n[PASS] All split tests passed!")
