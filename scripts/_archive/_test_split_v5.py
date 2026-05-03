#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"D:\AIHedgeFund\ai-hedge-fund-main")

print("[1] data_classes...")
from src.tools.data_classes import FinancialMetrics, FinancialLineItem, PriceBar, NewsItem
fm = FinancialMetrics(date="2026-03-31", return_on_equity=0.15)
print(f"  FinancialMetrics(date=..., ROE=0.15) -> OK")
li = FinancialLineItem(date="2026-03-31", revenue=1e9)
print(f"  FinancialLineItem(date=..., revenue=1e9) -> OK")
pb = PriceBar(date="2026-04-30", open=100, high=105, low=99, close=103, volume=1e6)
print(f"  PriceBar created -> OK")

print("[2] data_cleaner...")
from src.tools.data_cleaner import _safe_float
print(f"  _safe_float('3.14159') = {_safe_float('3.14159')}  OK")

print("[3] data_fetcher (no network)...")
from src.tools.data_fetcher import normalize_ts_code, get_pro_api
print(f"  normalize_ts_code('600519') = {normalize_ts_code('600519')}  OK")
print(f"  get_pro_api callable = {callable(get_pro_api)}  OK")

print("[4] backward compat (a_stock_api.py)...")
from src.tools.a_stock_api import (
    FinancialMetrics as FM, PriceBar as PB, get_prices,
    get_financial_metrics, filter_st_stocks, get_market_cap,
    get_market_context, normalize_ts_code as nts
)
print(f"  All backward compat imports OK")
print(f"  normalize_ts_code('000858') = {nts('000858')}  OK")

print("\n[PASS] All import tests passed!")
