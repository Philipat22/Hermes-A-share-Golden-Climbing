#!/usr/bin/env python3
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
