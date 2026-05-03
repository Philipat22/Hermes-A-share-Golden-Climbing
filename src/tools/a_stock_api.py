#!/usr/bin/env python3
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
    get_north_money, get_market_context, get_sector_index_perf,
    get_index_data,
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
    "get_insider_trades", "get_limit_list", "get_north_money", "get_sector_index_perf",
    "get_stock_info", "filter_st_stocks",
    "get_market_cap", "get_float_market_cap",
    "get_stocks_by_sector", "get_16_sector_stocks",
    "search_line_items", "get_market_context",
    "TUSHARE_TOKEN", "SECTOR_POOL",
]
