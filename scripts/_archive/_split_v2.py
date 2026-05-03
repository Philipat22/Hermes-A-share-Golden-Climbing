#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract sections from a_stock_api.py by line number ranges.
Run: python _split_v2.py
"""

SRC = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\a_stock_api.py"
OUT1 = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_classes.py"
OUT2 = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_cleaner.py"
OUT3 = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_fetcher.py"
FACADE = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\a_stock_api.py"

with open(SRC, "r", encoding="utf-8") as f:
    lines = f.readlines()

total = len(lines)
print(f"Total lines: {total}")

# Scan for class/function boundaries
for i, line in enumerate(lines):
    if line.startswith("class ") or line.startswith("def ") or line.startswith("# ===="):
        print(f"{i+1:4d}: {line.rstrip()}")
