#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correct split of data_fetcher.py into 3 layers.
Classes extracted by exact line ranges (1-indexed).
"""
import re

SRC = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_fetcher.py"
OUT_CLASSES = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_classes.py"
OUT_CLEANER = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_cleaner.py"
OUT_FETCHER = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_fetcher.py"
FACADE = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\a_stock_api.py"

with open(SRC, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 0-indexed ranges for extraction
CLASSES = [
    (204, 273),   # FinancialMetrics
    (273, 364),   # FinancialLineItem
    (364, 389),   # PriceBar
    (626, 755),   # NewsItem
    (755, 850),   # MarginTrade
    (850, 967),   # LimitUpData
    (967, 990),   # AStockInfo
]

CLEANER_RANGE = (1295, 1319)   # _safe_float + _calc_date

# --- Build data_classes.py ---
header = (
    "#!/usr/bin/env python3\n"
    "# -*- coding: utf-8 -*-\n"
    '"""\n'
    "A股数据模型 - FinancialMetrics, FinancialLineItem, PriceBar,\n"
    "NewsItem, MarginTrade, LimitUpData, AStockInfo\n"
    "自动生成 by _split_v4.py\n"
    '"""\n'
    "from __future__ import annotations\n"
    "from datetime import datetime\n"
    "from typing import Optional, Any\n"
    "import math\n"
    "\n"
)

class_parts = []
for (start, end) in CLASSES:
    # exclude the last blank line if it is one
    seg = "".join(lines[start:end])
    # trim trailing blank lines
    seg = seg.rstrip() + "\n"
    class_parts.append(seg)

with open(OUT_CLASSES, "w", encoding="utf-8") as f:
    f.write(header + "\n".join(class_parts))
print(f"[OK] data_classes.py: {header.count(chr(10)) + sum(p.count(chr(10)) for p in class_parts)} lines")

# --- Build data_cleaner.py ---
cleaner_header = (
    "#!/usr/bin/env python3\n"
    "# -*- coding: utf-8 -*-\n"
    '"""\n'
    "数据清洗工具 - _safe_float, _calc_date\n"
    "自动生成 by _split_v4.py\n"
    '"""\n'
    "from typing import Any\n"
    "from datetime import datetime\n"
    "import math\n"
    "\n"
)
cleaner_content = cleaner_header + "".join(lines[CLEANER_RANGE[0]:CLEANER_RANGE[1]])
with open(OUT_CLEANER, "w", encoding="utf-8") as f:
    f.write(cleaner_content)
print(f"[OK] data_cleaner.py: {cleaner_content.count(chr(10))} lines")

# --- Build data_fetcher.py (everything except classes and cleaner funcs) ---
# Ranges to EXCLUDE (1-indexed start, end)
exclude_ranges = [
    (205, 390),   # classes FinancialMetrics..PriceBar
    (627, 991),   # classes NewsItem..AStockInfo
    (1296, 1320), # _safe_float + _calc_date
]
# Convert to 0-indexed
exclude_0 = [(s-1, e-1) for (s, e) in exclude_ranges]

def in_excluded(idx):
    for s, e in exclude_0:
        if s <= idx < e:
            return True
    return False

fetcher_lines = []
# Keep lines 0..205 (imports, constants, get_pro_api, normalize, parse)
# up to just before class FinancialMetrics
for i in range(0, 205):
    fetcher_lines.append(lines[i])

# Add the class block placeholder
fetcher_lines.append(
    "# [Classes moved to src/tools/data_classes.py]\n"
)
fetcher_lines.append(
    "\n# [Classes moved to src/tools/data_classes.py]\n"
)
fetcher_lines.append("\n")

# Keep lines 390..627 (get_financial_metrics + helpers)
for i in range(390, 627):
    fetcher_lines.append(lines[i])

# Add NewsItem..AStockInfo placeholder
fetcher_lines.append("\n# [Classes moved to src/tools/data_classes.py]\n")

# Keep lines 991..1296 (get_stock_info through search_line_items)
for i in range(991, 1296):
    fetcher_lines.append(lines[i])

# Add _safe_float + _calc_date placeholder and import
fetcher_lines.extend([
    "\n# [Functions moved to src/tools/data_cleaner.py]\n",
    "from src.tools.data_cleaner import _safe_float\n",
    "from src.tools.data_cleaner import _calc_date as _calc_date_orig\n",
    "\n",
    "def _calc_date(end_date: str, years: int = 1) -> str:\n",
    '    """Date calculation wrapper - delegates to data_cleaner."""\n',
    "    return _calc_date_orig(end_date, years)\n",
])

fetcher_text = "".join(fetcher_lines)
with open(OUT_FETCHER, "w", encoding="utf-8") as f:
    f.write(fetcher_text)
print(f"[OK] data_fetcher.py: {fetcher_text.count(chr(10))} lines")

# --- Build backward-compatible facade ---
facade = (
    "#!/usr/bin/env python3\n"
    "# -*- coding: utf-8 -*-\n"
    '"""\n'
    "A股数据接口 - Facade (backward compatibility)\n"
    "Architecture:\n"
    "  data_classes.py  - 数据模型\n"
    "  data_cleaner.py  - 数据清洗\n"
    "  data_fetcher.py  - 数据获取\n"
    "自动生成 by _split_v4.py\n"
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
print(f"[OK] a_stock_api.py facade: {facade.count(chr(10))} lines")

print("\n[DONE] Split v4 complete!")
