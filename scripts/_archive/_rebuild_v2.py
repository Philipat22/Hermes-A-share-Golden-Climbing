#!/usr/bin/env python3
"""
Complete rebuild of the split files.
1. Trim garbage header from data_fetcher.py
2. Reconstruct data_classes.py (full class definitions)
3. Reconstruct data_cleaner.py (safe_float, calc_date)
4. Append missing function stubs to data_fetcher.py
5. Fix a_stock_api facade
"""
import os, ast

BASE = r'D:\AIHedgeFund\ai-hedge-fund-main\src\tools'
FETCHER = os.path.join(BASE, 'data_fetcher.py')

# ============================================================
# STEP 1: Trim garbage header from data_fetcher.py
# ============================================================
print('STEP 1: Trim garbage from data_fetcher.py')
with open(FETCHER, 'r', encoding='utf-8') as f:
    lines = f.readlines()
# Real code starts at line 67 (index 66) - the 'from __future__ import annotations'
new_lines = lines[66:]
with open(FETCHER, 'w', encoding='utf-8', newline='\n') as f:
    f.writelines(new_lines)
print(f'  Written {len(new_lines)} lines (trimmed 66 garbage lines)')

# ============================================================
# STEP 2: Syntax check
# ============================================================
print('STEP 2: Syntax check')
with open(FETCHER, 'r', encoding='utf-8') as f:
    src = f.read()
try:
    ast.parse(src)
    print('  [OK] data_fetcher.py syntax valid')
except SyntaxError as e:
    print(f'  [ERR] {e}')

# ============================================================
# STEP 3: data_classes.py
# ============================================================
print('STEP 3: Write data_classes.py')
data_classes = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股数据模型 - 7个核心数据结构
重建自 a_stock_api.py 使用模式分析 (2026-04-30)
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional, Any
import math


class PriceBar:
    """K线(OHLCV)数据结构"""
    def __init__(self, date: str, open: float = 0.0, high: float = 0.0,
                 low: float = 0.0, close: float = 0.0, volume: float = 0.0):
        self.date = date
        self.open = float(open)
        self.high = float(high)
        self.low = float(low)
        self.close = float(close)
        self.volume = float(volume)
    def __repr__(self):
        return f"PriceBar({self.date} O={self.open:.2f} H={self.high:.2f} L={self.low:.2f} C={self.close:.2f} V={self.volume:.0f})"


class AStockInfo:
    """A股公司基本信息"""
    def __init__(self, ts_code: str, name: str, industry: str = '',
                 market: str = '', list_date: str = '', is_st: bool = False, status: str = "L"):
        self.ts_code = ts_code
        self.name = name
        self.industry = industry
        self.market = market
        self.list_date = list_date
        self.is_st = is_st
        self.status = status
    def __repr__(self):
        return f"AStockInfo({self.ts_code} {self.name} [{self.industry}])"


class FinancialLineItem:
    """
    单期财务科目数据（从年报/季报提取，支持动态字段赋值）。
    """
    def __init__(self, date: str):
        self.date = date
        # 损益表
        self.revenue = None
        self.n_income = None
        self.gross_profit = None
        self.oper_exp = None
        self.operate_profit = None
        self.rd_exp = None
        self.basic_eps = None
        self.da = None
        self.ebit = None
        self.ebitda = None
        # 现金流
        self.free_cashflow = None
        self.n_cashflow_act = None
        self.c_pay_acq_const_fiolta = None
        # 资产负债表
        self.total_assets = None
        self.total_liab = None
        self.total_hldr_eqy_exc_min_int = None
        self.total_cur_assets = None
        self.total_cur_liab = None
        self.total_share = None
        self.money_cap = None

    def recalculate_derived(self):
        if self.ebit is None and self.operate_profit is not None:
            self.ebit = self.operate_profit
        if self.gross_profit is not None and self.revenue and self.revenue > 0:
            self.gross_margin = self.gross_profit / self.revenue
        else:
            self.gross_margin = None
        if self.n_income is not None and self.total_hldr_eqy_exc_min_int and self.total_hldr_eqy_exc_min_int > 0:
            self.roic = self.n_income / self.total_hldr_eqy_exc_min_int
        else:
            self.roic = None

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class FinancialMetrics:
    """
    公司综合财务指标（年化/TTM）。
    """
    def __init__(self):
        self.ts_code = ""; self.name = ""; self.date = ""; self.price = None
        self.market_cap = None; self.pe = None; self.pb = None; self.ps = None; self.pcf = None
        self.roe = None; self.roa = None; self.roic = None
        self.gross_margin = None; self.net_margin = None; self.net_profit_margin = None
        self.revenue = None; self.revenue_growth = None
        self.n_income = None; self.profit_growth = None
        self.operate_profit = None; self.ebit = None; self.ebitda = None
        self.total_assets = None; self.total_liab = None
        self.total_hldr_eqy_exc_min_int = None
        self.total_cur_assets = None; self.total_cur_liab = None
        self.intangible_asset = None; self.rd_exp = None
        self.free_cashflow = None; self.operating_cashflow = None; self.cash_and_equivalents = None
        self.total_shares = None; self.float_shares = None
        self.esp = None; self.cps = None
        self.debt_to_equity = None; self.current_ratio = None; self.quick_ratio = None
        self.days_sales_outstanding = None; self.days_sales_inventory = None
        self.days_payables_outstanding = None; self.cash_conversion_cycle = None
        self.asset_turnover = None; self.equity_multiplier = None; self.levered_beta = None
        self.ev = None; self.ev_ebitda = None; self.peg = None; self.debt_to_assets = None

    def __repr__(self):
        return f"FinancialMetrics({self.ts_code} {self.name}, PE={self.pe}, ROE={self.roe})"


class NewsItem:
    """公司新闻条目"""
    def __init__(self, date: str, title: str, content: str = "",
                 source: str = "", url: str = "", sentiment: float = None):
        self.date = date; self.title = title; self.content = content
        self.source = source; self.url = url; self.sentiment = sentiment

    def __repr__(self):
        return f"NewsItem({self.date} [{self.source}]: {self.title[:50]})"


class MarginTrade:
    """融资融券/资金流向数据"""
    def __init__(self, date: str, buy_amount: float = 0.0, sell_amount: float = 0.0,
                 net_amount: float = 0.0, balance: float = 0.0, description: str = ""):
        self.date = date; self.buy_amount = buy_amount; self.sell_amount = sell_amount
        self.net_amount = net_amount; self.balance = balance; self.description = description

    def __repr__(self):
        return f"MarginTrade({self.date} net={self.net_amount:.0f} balance={self.balance:.0f})"


class LimitUpData:
    """涨跌停数据"""
    def __init__(self, ts_code: str, trade_date: str, close: float = 0.0,
                 pct_change: float = 0.0, turnover_ratio: float = 0.0,
                 volume_ratio: float = 0.0, lead_signal: int = 0, nature: str = ""):
        self.ts_code = ts_code; self.trade_date = trade_date; self.close = close
        self.pct_change = pct_change; self.turnover_ratio = turnover_ratio
        self.volume_ratio = volume_ratio; self.lead_signal = lead_signal; self.nature = nature

    def __repr__(self):
        return f"LimitUpData({self.ts_code} {self.trade_date} pct={self.pct_change:.1f}%)"
'''

with open(os.path.join(BASE, 'data_classes.py'), 'w', encoding='utf-8', newline='\n') as f:
    f.write(data_classes)
print(f'  [OK] data_classes.py: {len(data_classes.splitlines())} lines')

# ============================================================
# STEP 4: data_cleaner.py
# ============================================================
print('STEP 4: Write data_cleaner.py')
data_cleaner = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗工具函数
"""
from datetime import datetime, timedelta
from typing import Any


def _safe_float(value: Any, default: float = None) -> float:
    """安全地将值转换为 float。处理 None、字符串、pandas NA 等情况。"""
    if value is None:
        return default if default is not None else 0.0
    import math
    if isinstance(value, float):
        return 0.0 if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if value in ('', 'nan', 'None', 'NA', '--', '-'):
            return default if default is not None else 0.0
        try:
            result = float(value)
            return 0.0 if (math.isnan(result) or math.isinf(result)) else result
        except (ValueError, TypeError):
            return default if default is not None else 0.0
    try:
        result = float(value)
        import math
        return 0.0 if (math.isnan(result) or math.isinf(result)) else result
    except (ValueError, TypeError, AttributeError):
        return default if default is not None else 0.0


def _calc_date(days: int = 0, fmt: str = '%Y%m%d') -> str:
    """计算相对日期。days > 0 未来，days < 0 过去。"""
    return (datetime.now() + timedelta(days=days)).strftime(fmt)
'''

with open(os.path.join(BASE, 'data_cleaner.py'), 'w', encoding='utf-8', newline='\n') as f:
    f.write(data_cleaner)
print(f'  [OK] data_cleaner.py: {len(data_cleaner.splitlines())} lines')

# ============================================================
# STEP 5: Append missing stubs to data_fetcher.py
# ============================================================
print('STEP 5: Append missing function stubs to data_fetcher.py')
stubs = '''

# ============================================================
# MISSING FUNCTION STUBS (reconstructed 2026-04-30)
# ============================================================

def get_pro_api():
    """Get or create Tushare Pro API instance (lazy singleton)."""
    global _pro_api
    if _pro_api is None:
        import tushare as ts
        token = os.getenv('TUSHARE_PRO_TOKEN') or os.getenv('TUSHARE_PRO') or TUSHARE_TOKEN
        _pro_api = ts.pro_api(token)
    return _pro_api


def get_stock_name_for_ticker(ticker: str) -> str:
    """Get stock name from ticker symbol."""
    info = get_stock_info(ticker)
    return info.name if info else ticker


def get_company_news(ticker: str, limit: int = 20, skip_sensitive: bool = True) -> list:
    """Get company news from Tushare major_news."""
    pro = get_pro_api()
    try:
        df = pro.major_news(ts_code=normalize_ts_code(ticker), limit=limit)
        if df is None or df.empty:
            return []
        results = []
        SENSITIVE = ['违规', '调查', '处罚', '立案']
        for _, row in df.iterrows():
            content = str(row.get('content', ''))
            title = str(row.get('title', ''))
            if skip_sensitive and any(k in content for k in SENSITIVE):
                continue
            results.append(NewsItem(
                date=str(row.get('datetime', ''))[:10],
                title=title[:200],
                content=content[:1000],
                source=str(row.get('src', '')),
                url='',
                sentiment=None,
            ))
        return results
    except Exception as e:
        print(f"[WARN] major_news failed for {ticker}: {e}")
        return []


def _simple_sentiment(text: str) -> float:
    """
    Simple keyword-based sentiment scoring.
    Returns: -1.0 (negative) to 1.0 (positive)
    """
    pos_kw = ['超预期', '突破', '增长', '创新', '领先', '盈利', '订单', '签约', '扩产', '新品', '获批', '战略', '合作']
    neg_kw = ['亏损', '下滑', '减持', '违规', '诉讼', '债务', '产能过剩', '降价', '竞争加剧', '风险', '调查', '处罚', '警告']
    pos = sum(1 for kw in pos_kw if kw in text)
    neg = sum(1 for kw in neg_kw if kw in text)
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0.0


def get_insider_trades(ticker: str, limit: int = 30) -> list:
    """Get margin trade data (融资融券) as proxy for institutional activity."""
    pro = get_pro_api()
    try:
        df = pro.margin_detail(ts_code=normalize_ts_code(ticker), limit=limit)
        if df is None or df.empty:
            return []
        results = []
        for _, row in df.iterrows():
            td = str(row.get('trade_date', ''))
            if len(td) == 8:
                td = f"{td[:4]}-{td[4:6]}-{td[6:8]}"
            results.append(MarginTrade(
                date=td,
                buy_amount=_safe_float(row.get('buy_amount')),
                sell_amount=_safe_float(row.get('sell_amount')),
                net_amount=_safe_float(row.get('net_amount')),
                balance=_safe_float(row.get('margin_balance')),
                description=f"Margin {row.get('trade_date', '')}",
            ))
        return results
    except Exception as e:
        print(f"[WARN] margin_detail failed for {ticker}: {e}")
        return []


def get_limit_list(date: str = None) -> list:
    """Get limit-up/limit-down list for given date."""
    pro = get_pro_api()
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    else:
        date = date.replace('-', '')
    try:
        df = pro.limit_list(trade_date=date)
        if df is None or df.empty:
            return []
        results = []
        for _, row in df.iterrows():
            td = str(row.get('trade_date', ''))
            if len(td) == 8:
                td = f"{td[:4]}-{td[4:6]}-{td[6:8]}"
            results.append(LimitUpData(
                ts_code=str(row.get('ts_code', '')),
                trade_date=td,
                close=_safe_float(row.get('close')),
                pct_change=_safe_float(row.get('pct_change')),
                turnover_ratio=_safe_float(row.get('turnover_ratio')),
                volume_ratio=_safe_float(row.get('volume_ratio')),
                lead_signal=int(row.get('fd_flag', 0)),
                nature='',
            ))
        return results
    except Exception as e:
        print(f"[WARN] limit_list: {e}")
        return []


def get_north_money(days: int = 10) -> list:
    """Get northbound money flow (沪深港通北向资金)."""
    pro = get_pro_api()
    results = []
    try:
        dfs = pro.moneyflow_hsgt()
        if dfs is not None and not dfs.empty:
            for _, row in dfs.iterrows().head(days):
                results.append({
                    'date': str(row.get('trade_date', '')),
                    'hgt_net': _safe_float(row.get('hgt_net')),
                    'sgt_net': _safe_float(row.get('sgt_net')),
                    'north_net': _safe_float(row.get('north_net')),
                    'north_amount': _safe_float(row.get('north_amount')),
                })
    except Exception as e:
        print(f"[WARN] moneyflow_hsgt: {e}")
    return results[:days]


def get_market_context(trade_date: str = None) -> dict:
    """
    Get market-wide context for 大师 analysis.
    Returns: {index_prices, sector_performance, market_pe, north_flow}
    """
    if trade_date is None:
        trade_date = datetime.now().strftime('%Y%m%d')
    pro = get_pro_api()
    context = {
        'trade_date': trade_date,
        'index_prices': {},
        'sector_performance': {},
        'market_pe': None,
        'north_flow': 0.0,
    }
    indices = {
        '000001.SH': '上证指数',
        '399001.SZ': '深证成指',
        '399006.SZ': '创业板指',
        '000688.SH': '科创50',
    }
    for ts_code, name in indices.items():
        try:
            df = pro.daily(ts_code=ts_code, start_date=trade_date, end_date=trade_date)
            if df is not None and not df.empty:
                row = df.iloc[0]
                context['index_prices'][name] = {
                    'close': _safe_float(row.get('close')),
                    'pct_change': _safe_float(row.get('pct_chg')),
                    'volume': _safe_float(row.get('vol')),
                }
        except:
            pass
    try:
        north = get_north_money(days=5)
        context['north_flow'] = sum(n.get('north_net', 0) or 0 for n in north)
    except:
        pass
    return context
'''

with open(FETCHER, 'a', encoding='utf-8') as f:
    f.write(stubs)
print('  [OK] Appended stubs')

# ============================================================
# STEP 6: Rewrite a_stock_api.py facade (clean)
# ============================================================
print('STEP 6: Rewrite a_stock_api.py facade')
facade = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股数据接口 - Facade (backward compatibility layer)
Delegates to: data_classes, data_cleaner, data_fetcher
Generated 2026-04-30
"""
from src.tools.data_classes import (
    FinancialMetrics, FinancialLineItem, PriceBar,
    NewsItem, MarginTrade, LimitUpData, AStockInfo,
)
from src.tools.data_cleaner import _safe_float
from src.tools.data_fetcher import (
    # Functions present in data_fetcher
    normalize_ts_code, parse_ts_code,
    get_prices, prices_to_df, get_price_data,
    get_financial_metrics, _get_financial_from_income,
    get_stock_info, filter_st_stocks,
    get_market_cap, get_float_market_cap,
    get_stocks_by_sector, get_16_sector_stocks,
    search_line_items,
    # Reconstructed stubs
    get_pro_api, get_stock_name_for_ticker, get_company_news,
    _simple_sentiment, get_insider_trades, get_limit_list,
    get_north_money, get_market_context,
    TUSHARE_TOKEN, SECTOR_POOL,
)

__all__ = [
    # Classes
    "FinancialMetrics", "FinancialLineItem", "PriceBar",
    "NewsItem", "MarginTrade", "LimitUpData", "AStockInfo",
    # Utils
    "_safe_float",
    # Fetcher functions
    "get_pro_api", "normalize_ts_code", "parse_ts_code",
    "get_prices", "prices_to_df", "get_price_data",
    "get_financial_metrics", "_get_financial_from_income",
    "get_stock_name_for_ticker", "get_company_news", "_simple_sentiment",
    "get_insider_trades", "get_limit_list", "get_north_money",
    "get_stock_info", "filter_st_stocks",
    "get_market_cap", "get_float_market_cap",
    "get_stocks_by_sector", "get_16_sector_stocks",
    "search_line_items", "get_market_context",
    "TUSHARE_TOKEN", "SECTOR_POOL",
]
'''

with open(os.path.join(BASE, 'a_stock_api.py'), 'w', encoding='utf-8', newline='\n') as f:
    f.write(facade)
print('  [OK] a_stock_api.py facade rewritten')

# ============================================================
# STEP 7: Final verification
# ============================================================
print()
print('STEP 7: Final verification')
import importlib.util, sys
sys.path.insert(0, r'D:\AIHedgeFund\ai-hedge-fund-main')

try:
    from src.tools.data_classes import FinancialMetrics, FinancialLineItem, PriceBar, NewsItem, MarginTrade, LimitUpData, AStockInfo
    print('  [OK] data_classes: all 7 classes importable')
except ImportError as e:
    print(f'  [ERR] data_classes: {e}')

try:
    from src.tools.data_cleaner import _safe_float
    print('  [OK] data_cleaner: _safe_float importable')
except ImportError as e:
    print(f'  [ERR] data_cleaner: {e}')

try:
    from src.tools.a_stock_api import get_market_context, get_pro_api, get_company_news
    print('  [OK] a_stock_api: facade imports all working')
except ImportError as e:
    print(f'  [ERR] a_stock_api: {e}')

print()
print('DONE. All split files rebuilt.')
