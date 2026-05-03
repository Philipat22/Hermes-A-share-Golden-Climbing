#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 a_stock_api.py 拆分为三层模块
执行: python _split_api.py
"""

import re

SRC = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\a_stock_api.py"
DST_DATA = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_classes.py"
DST_FETCH = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_fetcher.py"
DST_CLEAN = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\data_cleaner.py"
DST_FACADE = r"D:\AIHedgeFund\ai-hedge-fund-main\src\tools\a_stock_api.py"

with open(SRC, "r", encoding="utf-8") as f:
    content = f.read()

lines = content.split("\n")

# ============================================================
# 策略：按 # ==== 注释段落分割
# ============================================================

# 定义每个模块负责的段落
data_classes_sections = [
    "FinancialMetrics",
    "FinancialLineItem",
    "PriceBar",
    "NewsItem",
    "MarginTrade",
    "LimitUpData",
    "AStockInfo",
]

data_cleaner_items = [
    "_safe_float",
    "_pct",
]

fetcher_start_markers = [
    "def get_pro_api",
    "def normalize_ts_code",
    "def parse_ts_code",
    "def get_prices",
    "def prices_to_df",
    "def get_price_data",
    "def get_financial_metrics",
    "def _get_financial_from_income",
    "def get_company_news",
    "def _simple_sentiment",
    "def get_insider_trades",
    "def get_limit_list",
    "def get_north_money",
    "def get_stock_info",
    "def filter_st_stocks",
    "def get_market_cap",
    "def get_float_market_cap",
    "def get_stocks_by_sector",
    "def get_16_sector_stocks",
    "def search_line_items",
    "def get_market_context",
    "def _calc_date",
]

# ============================================================
# 构建各模块内容
# ============================================================

module_lines = {"data_classes": [], "data_cleaner": [], "data_fetcher": [], "facade": []}
current_module = None

# Header lines (shebang, imports, constants) -> data_fetcher
header_done = False
for i, line in enumerate(lines):
    if not header_done:
        if line.startswith("def ") or line.startswith("class "):
            header_done = True
    if not header_done:
        module_lines["data_fetcher"].append(line)
    elif line.startswith("class "):
        cls_name = line.split("class ")[1].split("(")[0].split(":")[0].strip()
        if cls_name in data_classes_sections:
            current_module = "data_classes"
        else:
            current_module = "data_fetcher"
        module_lines[current_module].append(line)
    elif line.startswith("def _safe_float") or line.startswith("def _calc_date") or line.startswith("def _pct"):
        if "_safe_float" in line or "_pct" in line:
            current_module = "data_cleaner"
        else:
            current_module = "data_fetcher"
        module_lines[current_module].append(line)
    elif line.startswith("def get_pro_api") or line.startswith("def normalize_ts_code") or line.startswith("def parse_ts_code"):
        current_module = "data_fetcher"
        module_lines[current_module].append(line)
    elif line.startswith("def "):
        # Check if it's a fetcher function
        fn = line.split("(")[0].split("def ")[1].strip()
        if fn in ["get_prices", "get_price_data", "prices_to_df", "get_financial_metrics",
                  "_get_financial_from_income", "get_company_news", "_simple_sentiment",
                  "get_insider_trades", "get_limit_list", "get_north_money",
                  "get_stock_info", "filter_st_stocks", "get_market_cap",
                  "get_float_market_cap", "get_stocks_by_sector", "get_16_sector_stocks",
                  "search_line_items", "get_market_context"]:
            current_module = "data_fetcher"
        else:
            current_module = "data_fetcher"  # fallback
        module_lines[current_module].append(line)
    else:
        module_lines[current_module].append(line)

def write_module(path, lines, module_name, description, imports=""):
    header = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
{description}
自动生成自 a_stock_api.py - 由 _split_api.py 创建
"""
{imports}
'''
    text = header + "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[OK] {module_name}: {len(lines)} lines -> {path}")

# ============================================================
# 写 data_classes.py
# ============================================================
# 找头部 import（只取和 classes 相关的部分）
class_imports = [
    "from __future__ import annotations",
    "import pandas as pd",
    "import numpy as np",
    "from datetime import datetime, timedelta",
    "from typing import Optional, Any",
]

# 找所有类和它们的完整定义
class_texts = {}
current_class = None
class_buffer = []

for line in lines:
    if line.startswith("class ") and "class " in line:
        if current_class and class_buffer:
            class_texts[current_class] = "\n".join(class_buffer)
        mt = re.match(r"class (\w+)", line)
        current_class = mt.group(1) if mt else None
        class_buffer = [line]
    elif current_class:
        class_buffer.append(line)

if current_class and class_buffer:
    class_texts[current_class] = "\n".join(class_buffer)

# 写 data_classes.py
dc_out = [
    "#!/usr/bin/env python3",
    "# -*- coding: utf-8 -*-",
    '"""',
    "A股数据模型 - FinancialMetrics, FinancialLineItem, PriceBar, NewsItem 等",
    "自动生成自 a_stock_api.py - 由 _split_api.py 创建",
    '"""',
    "",
    "from __future__ import annotations",
    "import math",
    "from datetime import datetime",
    "from typing import Optional, Any",
    "",
]
for cls, cls_text in class_texts.items():
    dc_out.append(cls_text)
    dc_out.append("")

write_module(DST_DATA, dc_out, "data_classes", "A股数据模型", "")

# ============================================================
# 写 data_cleaner.py
# ============================================================
cleaner_out = [
    "#!/usr/bin/env python3",
    "# -*- coding: utf-8 -*-",
    '"""',
    "数据清洗工具 - _safe_float, 百分数归一化, 字段映射",
    "自动生成自 a_stock_api.py - 由 _split_api.py 创建",
    '"""',
    "",
    "from typing import Any",
    "import math",
    "",
]

# 提取 _safe_float, _pct 函数
in_func = None
func_buf = []
for line in lines:
    if line.startswith("def _safe_float") or line.startswith("def _pct"):
        if in_func and func_buf:
            cleaner_out.append("\n".join(func_buf))
            cleaner_out.append("")
        in_func = line.split("(")[0].replace("def ", "")
        func_buf = [line]
    elif in_func and line.startswith("    "):
        func_buf.append(line)
    elif in_func and not line.startswith("    ") and line.strip() != "":
        cleaner_out.append("\n".join(func_buf))
        cleaner_out.append("")
        in_func = None
        func_buf = []

if in_func and func_buf:
    cleaner_out.append("\n".join(func_buf))
    cleaner_out.append("")

write_module(DST_CLEAN, cleaner_out, "data_cleaner", "数据清洗工具", "")

# ============================================================
# 写 data_fetcher.py（保留所有 API 调用逻辑）
# ============================================================
fetcher_out = [
    "#!/usr/bin/env python3",
    "# -*- coding: utf-8 -*-",
    '"""',
    "A股数据获取层 - 所有 Tushare/AKShare API 调用",
    "自动生成自 a_stock_api.py - 由 _split_api.py 创建",
    '"""',
    "",
    "from __future__ import annotations",
    "import os",
    "import sys",
    "import io",
    "from datetime import datetime, timedelta",
    "from typing import Optional, Any",
    "",
    "import pandas as pd",
    "import numpy as np",
    "",
    "import tushare as ts",
    "import akshare as ak",
    "",
    "from src.tools.data_classes import (",
    "    FinancialMetrics, FinancialLineItem, PriceBar,",
    "    NewsItem, MarginTrade, LimitUpData, AStockInfo,",
    ")",
    "from src.tools.data_cleaner import _safe_float",
    "from src.utils.sector_map import (",
    "    SECTOR_INDUSTRY_MAP,",
    "    get_stocks_by_sector as _sector_get_stocks,",
    "    filter_stocks as _sector_filter,",
    "    get_stocks_by_sector as _sector_get_all,",
    "    get_stock_info as _sector_get_info,",
    ")",
    "",
    "# Windows 编码修复",
    "if sys.platform == 'win32' and __name__ == '__main__':",
    "    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')",
    "    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')",
    "",
    "# Token 配置",
    "TUSHARE_TOKEN = os.getenv('TUSHARE_PRO_TOKEN') or os.getenv('TUSHARE_PRO')",
    "if not TUSHARE_TOKEN:",
    "    TUSHARE_TOKEN = '5243de737c1a25110583352fde4458266314877dd0c342cae1a9f4c7'",
    "",
    "# 板块池",
    "SECTOR_POOL: dict[str, list[str]] = SECTOR_INDUSTRY_MAP",
    "",
    "_pro_api = None",
    "",
]

# 找所有 def 函数定义（从 get_pro_api 开始）
in_function = False
func_buf = []
for i, line in enumerate(lines):
    if line.startswith("def get_pro_api") or line.startswith("def normalize_ts_code"):
        in_function = True
        func_buf = [line]
    elif in_function:
        if line.startswith("def ") and not line.startswith("    "):
            fetcher_out.extend(func_buf)
            fetcher_out.append("")
            func_buf = [line]
        else:
            func_buf.append(line)

if func_buf:
    fetcher_out.extend(func_buf)

write_module(DST_FETCH, fetcher_out, "data_fetcher", "数据获取层", "")

# ============================================================
# 写 facade (a_stock_api.py 向后兼容)
# ============================================================
facade = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股数据接口 - Facade（向后兼容）
本文件已重构为三层结构:
  src/tools/data_classes.py  - 数据模型
  src/tools/data_cleaner.py  - 数据清洗
  src/tools/data_fetcher.py  - 数据获取
本文件保留用于向后兼容，所有符号从子模块重新导出。
"""

# 重新导出全部符号（向后兼容已有代码）
from src.tools.data_classes import (
    FinancialMetrics,
    FinancialLineItem,
    PriceBar,
    NewsItem,
    MarginTrade,
    LimitUpData,
    AStockInfo,
)
from src.tools.data_cleaner import _safe_float
from src.tools.data_fetcher import (
    get_pro_api,
    normalize_ts_code,
    parse_ts_code,
    get_prices,
    prices_to_df,
    get_price_data,
    get_financial_metrics,
    get_company_news,
    get_insider_trades,
    get_limit_list,
    get_north_money,
    get_stock_info,
    filter_st_stocks,
    get_market_cap,
    get_float_market_cap,
    get_stocks_by_sector,
    get_16_sector_stocks,
    search_line_items,
    get_market_context,
    TUSHARE_TOKEN,
    SECTOR_POOL,
)

__all__ = [
    "FinancialMetrics", "FinancialLineItem", "PriceBar",
    "NewsItem", "MarginTrade", "LimitUpData", "AStockInfo",
    "_safe_float",
    "get_pro_api", "normalize_ts_code", "parse_ts_code",
    "get_prices", "prices_to_df", "get_price_data",
    "get_financial_metrics", "get_company_news",
    "get_insider_trades", "get_limit_list", "get_north_money",
    "get_stock_info", "filter_st_stocks",
    "get_market_cap", "get_float_market_cap",
    "get_stocks_by_sector", "get_16_sector_stocks",
    "search_line_items", "get_market_context",
    "TUSHARE_TOKEN", "SECTOR_POOL",
]
'''

with open(DST_FACADE, "w", encoding="utf-8") as f:
    f.write(facade)
print(f"[OK] facade (backward compat): -> {DST_FACADE}")

print("\n[DONE] Split complete!")
