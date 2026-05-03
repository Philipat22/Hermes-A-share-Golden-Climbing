#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precise class extraction from data_fetcher.py by scanning for class/def keywords.
Run: python _extract_classes.py
"""
import re

SRC = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_fetcher.py"
OUT_CLASSES = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_classes.py"
OUT_CLEANER = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_cleaner.py"
OUT_FETCHER = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_fetcher.py"
FACADE = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\a_stock_api.py"

with open(SRC, "r", encoding="utf-8") as f:
    content = f.read()
    lines = content.split("\n")

total = len(lines)
print(f"Total lines: {total}")

# ============================================================
# STEP 1: Find all class and def boundaries
# ============================================================
class_boundaries = []  # (start_line_idx, end_line_idx, name)
func_boundaries = []   # (start_line_idx, end_line_idx, name)

current_class_start = None
current_class_name = None
current_class_brace_depth = 0
in_class = False

for i, line in enumerate(lines):
    stripped = line.strip()
    
    # Detect class def
    if stripped.startswith("class ") and not stripped.startswith("class _"):
        if in_class and current_class_start is not None:
            # Close previous class
            class_boundaries.append((current_class_start, i, current_class_name))
        m = re.match(r"class (\w+)", stripped)
        current_class_name = m.group(1)
        current_class_start = i
        in_class = True
        current_class_brace_depth = 0
    elif in_class:
        # Track indent level (heuristic: count leading spaces/tabs)
        if stripped and not stripped.startswith("#"):
            # Increase on open paren/bracket, decrease on close
            for ch in stripped:
                if ch == '(' or ch == '[':
                    current_class_brace_depth += 1
                elif ch == ')' or ch == ']':
                    current_class_brace_depth -= 1
            # If we see another class or top-level def at depth 0, close this class
            if (stripped.startswith("class ") or (stripped.startswith("def ") and not line.startswith("    "))) and current_class_brace_depth <= 0:
                class_boundaries.append((current_class_start, i, current_class_name))
                if stripped.startswith("class "):
                    m = re.match(r"class (\w+)", stripped)
                    current_class_name = m.group(1)
                    current_class_start = i
                else:
                    in_class = False
                    current_class_start = None

# Close last class
if in_class and current_class_start is not None:
    class_boundaries.append((current_class_start, total, current_class_name))

print("\nClass boundaries found:")
for s, e, n in class_boundaries:
    print(f"  lines {s+1}-{e} ({e-s} lines): {n}")

# ============================================================
# STEP 2: Find tool function boundaries (_safe_float, _calc_date)
# ============================================================
tool_funcs = {}
for i, line in enumerate(lines):
    if line.startswith("def _safe_float") or line.startswith("def _calc_date"):
        name = line.split("(")[0].replace("def ", "").strip()
        # Find end of function (next top-level def or class or end)
        end = i + 1
        while end < total and (lines[end].startswith("    ") or lines[end].strip() == "" or lines[end].startswith("#")):
            if lines[end].startswith("def ") and not lines[end].startswith("    "):
                break
            end += 1
        tool_funcs[name] = (i, end)
        print(f"Tool func: lines {i+1}-{end}: {name}")

# ============================================================
# STEP 3: Extract classes
# ============================================================
TARGET_CLASSES = {"FinancialMetrics", "FinancialLineItem", "PriceBar",
                  "NewsItem", "MarginTrade", "LimitUpData", "AStockInfo"}

class_texts = {}
for s, e, n in class_boundaries:
    if n in TARGET_CLASSES:
        # trim leading blank lines
        seg = "\n".join(lines[s:e]).rstrip()
        class_texts[n] = seg

print("\nClasses extracted:")
for n in TARGET_CLASSES:
    if n in class_texts:
        print(f"  {n}: {len(class_texts[n])} chars")
    else:
        print(f"  {n}: MISSING")

# ============================================================
# STEP 4: Build data_classes.py
# ============================================================
header = (
    "#!/usr/bin/env python3\n"
    "# -*- coding: utf-8 -*-\n"
    '"""\n'
    "A股数据模型 - FinancialMetrics, FinancialLineItem, PriceBar,\n"
    "NewsItem, MarginTrade, LimitUpData, AStockInfo\n"
    "自动生成 by _extract_classes.py\n"
    '"""\n'
    "from __future__ import annotations\n"
    "from datetime import datetime\n"
    "from typing import Optional, Any\n"
    "import math\n"
    "\n"
)

with open(OUT_CLASSES, "w", encoding="utf-8") as f:
    f.write(header + "\n\n".join(class_texts.values()) + "\n")
print(f"\n[OK] data_classes.py written")

# ============================================================
# STEP 5: Extract tool funcs -> data_cleaner.py
# ============================================================
tool_header = (
    "#!/usr/bin/env python3\n"
    "# -*- coding: utf-8 -*-\n"
    '"""\n'
    "数据清洗工具 - _safe_float, _calc_date\n"
    "自动生成 by _extract_classes.py\n"
    '"""\n'
    "from typing import Any\n"
    "from datetime import datetime\n"
    "import math\n"
    "\n"
)

tool_parts = []
for name, (s, e) in sorted(tool_funcs.items()):
    tool_parts.append("\n".join(lines[s:e]).rstrip())

with open(OUT_CLEANER, "w", encoding="utf-8") as f:
    f.write(tool_header + "\n\n".join(tool_parts) + "\n")
print(f"[OK] data_cleaner.py written")

# ============================================================
# STEP 6: Build data_fetcher.py (everything except classes and tool funcs)
# ============================================================
# Collect excluded line indices
excluded = set()
for n in TARGET_CLASSES:
    if n in class_boundaries:
        for i in range(class_boundaries[n][0], class_boundaries[n][1]):
            excluded.add(i)
for name, (s, e) in tool_funcs.items():
    for i in range(s, e):
        excluded.add(i)

fetcher_lines = []
# Also exclude the duplicate class header comments
for i in range(total):
    if i in excluded:
        continue
    # Skip lines that are section headers for removed content
    l = lines[i].strip()
    if l.startswith("# ====") and any(x in content[content.find(lines[i]):content.find(lines[i])+200] 
                                        for x in ["数据模型", "数据清洗", "财务指标接口", "新闻接口", "融资融券", "涨跌停", "北向资金", "股票基本信息"]):
        continue
    fetcher_lines.append(lines[i])

# Add imports at the top
fetcher_import = (
    "#!/usr/bin/env python3\n"
    "# -*- coding: utf-8 -*-\n"
    '"""\n'
    "A股数据获取层 - Tushare/AKShare API 调用\n"
    "自动生成 by _extract_classes.py\n"
    '"""\n'
    "from __future__ import annotations\n"
    "import os\n"
    "import sys\n"
    "import io\n"
    "from datetime import datetime, timedelta\n"
    "from typing import Optional, Any\n"
    "\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "\n"
    "import tushare as ts\n"
    "import akshare as ak\n"
    "\n"
    "from src.tools.data_classes import (\n"
    "    FinancialMetrics, FinancialLineItem, PriceBar,\n"
    "    NewsItem, MarginTrade, LimitUpData, AStockInfo,\n"
    ")\n"
    "from src.tools.data_cleaner import _safe_float\n"
    "from src.utils.sector_map import (\n"
    "    SECTOR_INDUSTRY_MAP,\n"
    "    get_stocks_by_sector as _sector_get_stocks,\n"
    "    filter_stocks as _sector_filter,\n"
    "    get_stocks_by_sector as _sector_get_all,\n"
    "    get_stock_info as _sector_get_info,\n"
    ")\n"
    "\n"
    "if sys.platform == 'win32' and __name__ == '__main__':\n"
    "    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')\n"
    "    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')\n"
    "\n"
    "TUSHARE_TOKEN = os.getenv('TUSHARE_PRO_TOKEN') or os.getenv('TUSHARE_PRO')\n"
    "if not TUSHARE_TOKEN:\n"
    "    TUSHARE_TOKEN = '5243de737c1a25110583352fde4458266314877dd0c342cae1a9f4c7'\n"
    "\n"
    "SECTOR_POOL: dict[str, list[str]] = SECTOR_INDUSTRY_MAP\n"
    "\n"
    "_pro_api = None\n"
    "\n"
    "\n"
)

fetcher_text = fetcher_import + "\n".join(fetcher_lines)
with open(OUT_FETCHER, "w", encoding="utf-8") as f:
    f.write(fetcher_text)
print(f"[OK] data_fetcher.py written ({len(fetcher_text.split(chr(10)))} lines)")

# ============================================================
# STEP 7: Facade
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
    "自动生成 by _extract_classes.py\n"
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
print(f"[OK] a_stock_api.py facade written")

print("\n[DONE] Precise class extraction complete!")
