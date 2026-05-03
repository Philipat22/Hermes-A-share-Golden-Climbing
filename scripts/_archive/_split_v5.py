#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correct split of data_fetcher.py into 3 layers.
Extracts classes, cleans up data_fetcher, generates facade.
Run: python _split_v5.py
"""

SRC = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_fetcher.py"
OUT_CLASSES = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_classes.py"
OUT_CLEANER = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_cleaner.py"
OUT_FETCHER = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_fetcher.py"
FACADE = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\a_stock_api.py"

with open(SRC, "r", encoding="utf-8") as f:
    lines = f.readlines()

total = len(lines)
print(f"Total lines in data_fetcher.py: {total}")

# 1-indexed ranges to EXCLUDE from fetcher (keep everything else)
# Classes: 205-390 (FinancialMetrics..PriceBar) + 627-990 (NewsItem..AStockInfo)
# Tool funcs: 1296-1319 (_safe_float + _calc_date + blank after)
EXCLUDE = [
    (205, 390),   # classes FinancialMetrics..PriceBar
    (627, 991),   # NewsItem + get_stock_name + get_company_news + _simple_sentiment + MarginTrade + get_insider_trades + LimitUpData + get_limit_list + north_money + AStockInfo
    (1296, 1320), # _safe_float + _calc_date (tool funcs)
]
# Convert to 0-indexed
EX0 = [(s-1, e-1) for s, e in EXCLUDE]

def in_excl(idx):
    for s, e in EX0:
        if s <= idx < e:
            return True
    return False

# Extract classes (1-indexed → 0-indexed: 204-389, 626-990, etc.)
# FinancialMetrics: 205-273, FinancialLineItem: 274-364, PriceBar: 365-389
# NewsItem: 627-755, MarginTrade: 756-850, LimitUpData: 851-967, AStockInfo: 968-990
CLASS_EXTRACT = [
    (204, 273),   # FinancialMetrics
    (273, 364),   # FinancialLineItem
    (365, 389),   # PriceBar (up to before def get_financial_metrics at 391)
    (626, 755),   # NewsItem
    (755, 850),   # MarginTrade
    (850, 967),   # LimitUpData
    (967, 990),   # AStockInfo
]

# 1-indexed _safe_float at 1296, _calc_date at 1310, ends at ~1319
TOOL_EXTRACT = (1295, 1319)  # 0-indexed

# ============================================================
# 1. Build data_classes.py
# ============================================================
classes_header = (
    "#!/usr/bin/env python3\n"
    "# -*- coding: utf-8 -*-\n"
    '"""\n'
    "A股数据模型 - FinancialMetrics, FinancialLineItem, PriceBar,\n"
    "NewsItem, MarginTrade, LimitUpData, AStockInfo\n"
    "自动生成 by _split_v5.py\n"
    '"""\n'
    "from __future__ import annotations\n"
    "from datetime import datetime\n"
    "from typing import Optional, Any\n"
    "import math\n"
    "\n"
)

parts = []
for s, e in CLASS_EXTRACT:
    seg = "".join(lines[s:e]).rstrip()
    parts.append(seg)

with open(OUT_CLASSES, "w", encoding="utf-8") as f:
    f.write(classes_header + "\n\n".join(parts) + "\n")
lc = classes_header.count("\n") + sum(p.count("\n") for p in parts) + len(parts)
print(f"[OK] data_classes.py: {lc} lines -> {OUT_CLASSES}")

# ============================================================
# 2. Build data_cleaner.py
# ============================================================
cleaner_header = (
    "#!/usr/bin/env python3\n"
    "# -*- coding: utf-8 -*-\n"
    '"""\n'
    "数据清洗工具 - _safe_float, _calc_date\n"
    "自动生成 by _split_v5.py\n"
    '"""\n'
    "from typing import Any\n"
    "from datetime import datetime\n"
    "import math\n"
    "\n"
)
cleaner_content = cleaner_header + "".join(lines[TOOL_EXTRACT[0]:TOOL_EXTRACT[1]])

with open(OUT_CLEANER, "w", encoding="utf-8") as f:
    f.write(cleaner_content)
print(f"[OK] data_cleaner.py: {cleaner_content.count(chr(10))} lines -> {OUT_CLEANER}")

# ============================================================
# 3. Build data_fetcher.py (keep all non-excluded lines)
# ============================================================
fetcher_lines = []
for i in range(total):
    if in_excl(i):
        continue
    fetcher_lines.append(lines[i])

# Prepend import for data_classes and data_cleaner
import_pragma = (
    "# [Classes & functions migrated to data_classes.py and data_cleaner.py]\n"
    "from src.tools.data_classes import (\n"
    "    FinancialMetrics, FinancialLineItem, PriceBar,\n"
    "    NewsItem, MarginTrade, LimitUpData, AStockInfo,\n"
    ")\n"
    "from src.tools.data_cleaner import _safe_float\n"
    "\n"
)

fetcher_text = import_pragma + "".join(fetcher_lines)
with open(OUT_FETCHER, "w", encoding="utf-8") as f:
    f.write(fetcher_text)
print(f"[OK] data_fetcher.py: {fetcher_text.count(chr(10))} lines -> {OUT_FETCHER}")

# ============================================================
# 4. Build backward-compatible facade
# ============================================================
facade = (
    "#!/usr/bin/env python3\n"
    "# -*- coding: utf-8 -*-\n"
    '"""\n'
    "A股数据接口 - Facade (backward compatibility)\n"
    "架构:\n"
    "  data_classes.py  - 数据模型\n"
    "  data_cleaner.py  - 数据清洗\n"
    "  data_fetcher.py  - 数据获取\n"
    "自动生成 by _split_v5.py\n"
    '"""\n'
    "from src.tools.data_classes import (\n"
    "    FinancialMetrics, FinancialLineItem, PriceBar,\n"
    "    NewsItem, MarginTrade, LimitUpData, AStockInfo,\n"
    ")\n"
    "from src.tools.data_cleaner import _safe_float\n"
    "from src.tools.data_fetcher import (\n"
    "    get_pro_api, normalize_ts_code, parse_ts_code,\n"
    "    get_prices, prices_to_df, get_price_data,\n"
    "    get_financial_metrics, _get_financial_from_income,\n"
    "    get_stock_name_for_ticker, get_company_news, _simple_sentiment,\n"
    "    get_insider_trades, get_limit_list, get_north_money,\n"
    "    get_stock_info, filter_st_stocks,\n"
    "    get_market_cap, get_float_market_cap,\n"
    "    get_stocks_by_sector, get_16_sector_stocks,\n"
    "    search_line_items, get_market_context,\n"
    "    TUSHARE_TOKEN, SECTOR_POOL,\n"
    ")\n"
    "\n"
    "__all__ = [\n"
    '    "FinancialMetrics", "FinancialLineItem", "PriceBar",\n'
    '    "NewsItem", "MarginTrade", "LimitUpData", "AStockInfo",\n'
    '    "_safe_float",\n'
    '    "get_pro_api", "normalize_ts_code", "parse_ts_code",\n'
    '    "get_prices", "prices_to_df", "get_price_data",\n'
    '    "get_financial_metrics", "_get_financial_from_income",\n'
    '    "get_stock_name_for_ticker", "get_company_news", "_simple_sentiment",\n'
    '    "get_insider_trades", "get_limit_list", "get_north_money",\n'
    '    "get_stock_info", "filter_st_stocks",\n'
    '    "get_market_cap", "get_float_market_cap",\n'
    '    "get_stocks_by_sector", "get_16_sector_stocks",\n'
    '    "search_line_items", "get_market_context",\n'
    '    "TUSHARE_TOKEN", "SECTOR_POOL",\n'
    "]\n"
)
with open(FACADE, "w", encoding="utf-8") as f:
    f.write(facade)
print(f"[OK] a_stock_api.py facade: {facade.count(chr(10))} lines -> {FACADE}")

print("\n[DONE] 3-layer split complete!")
