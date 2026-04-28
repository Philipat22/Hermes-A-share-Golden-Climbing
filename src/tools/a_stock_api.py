#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股数据接口 - Tushare Pro + AKShare
替换原 api.py 的 Financial Datasets API

作者: JoJo (华尔街金融大咖)
适配: virattt/ai-hedge-fund 项目 A股版本
"""

from __future__ import annotations

import os
import sys
import io
import math
import json
from datetime import datetime, timedelta
from typing import Optional, Any

import pandas as pd
import numpy as np

# Tushare Pro
import tushare as ts

# AKShare (补充数据源)
import akshare as ak

# 申万行业板块系统
from src.utils.sector_map import (
    SECTOR_INDUSTRY_MAP,
    get_stocks_by_sector as _sector_get_stocks,
    filter_stocks as _sector_filter,
    get_all_stocks as _sector_get_all,
    get_stock_info as _sector_get_info,
)

# ============================================================
# 编码修复（Windows，只在直接运行时生效，import 时跳过）
# ============================================================
if sys.platform == 'win32' and __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

# ============================================================
# 常量配置
# ============================================================

# Tushare Pro Token (从环境变量读取)
# Tushare Pro Token（优先级：环境变量 > 配置文件）
TUSHARE_TOKEN = os.getenv("TUSHARE_PRO_TOKEN") or os.getenv("TUSHARE_PRO")
if not TUSHARE_TOKEN:
    # Token 2026-04-27 更新
    TUSHARE_TOKEN = "5243de737c1a25110583352fde4458266314877dd0c342cae1a9f4c7"

# A股交易所后缀
EXCHANGE_MAP = {
    "SH": ".SH",  # 上海
    "SZ": ".SZ",  # 深圳
    "BJ": ".BJ",  # 北京
}

# 14个核心板块（基于申万行业分类，与 sector_map 同步）
SECTOR_POOL: dict[str, list[str]] = SECTOR_INDUSTRY_MAP

# ============================================================
# Tushare Pro 初始化
# ============================================================
_pro_api = None


def get_pro_api():
    """获取 Tushare Pro API 实例（单例）"""
    global _pro_api
    if _pro_api is None:
        try:
            _pro_api = ts.pro_api(TUSHARE_TOKEN)
        except Exception as e:
            print(f"[WARN] Tushare API 初始化失败: {e}")
            _pro_api = None
    return _pro_api


# ============================================================
# 核心数据接口
# ============================================================

def normalize_ts_code(ts_code: str) -> str:
    """
    标准化股票代码格式
    输入: "600519" / "600519.SH" / "贵州茅台"
    输出: "600519.SH"
    """
    ts_code = str(ts_code).strip().upper()
    # 已经是标准格式
    if ".SH" in ts_code or ".SZ" in ts_code or ".BJ" in ts_code:
        return ts_code
    # 纯数字代码，推断交易所
    if ts_code.isdigit():
        if len(ts_code) == 6:
            if ts_code.startswith(("6", "5", "9")):
                return f"{ts_code}.SH"
            elif ts_code.startswith(("0", "1", "2", "3")):
                return f"{ts_code}.SZ"
            elif ts_code.startswith(("4", "8")):
                return f"{ts_code}.BJ"
    return ts_code


def parse_ts_code(ts_code: str) -> tuple[str, str]:
    """解析股票代码，返回 (code, exchange)"""
    ts_code = normalize_ts_code(ts_code)
    if "." in ts_code:
        code, ex = ts_code.split(".")
        return code, ex
    return ts_code, "SH"


def get_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None,
) -> list[Any]:
    """
    获取股票历史价格数据（OHLCV）
    返回: list[PriceBar] 格式，与原 api.py 兼容

    Args:
        ticker: A股代码，如 "600519.SH" 或 "600519"
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        api_key: 忽略，保留接口兼容性
    """
    pro = get_pro_api()
    if pro is None:
        return []

    ts_code = normalize_ts_code(ticker)

    # 转换日期格式
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Tushare 日线数据
    try:
        # pro.daily 返回: ts_code, trade_date, open, high, low, close, vol, amount
        df = pro.daily(
            ts_code=ts_code,
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
        )
    except Exception as e:
        print(f"[WARN] Tushare daily failed for {ts_code}: {e}")
        return []

    if df is None or df.empty:
        return []

    # 按日期正序排列
    df = df.sort_values("trade_date").reset_index(drop=True)

    # 转换为 PriceBar 列表（兼容原接口）
    results = []
    for _, row in df.iterrows():
        trade_date = str(row["trade_date"])
        if len(trade_date) == 8:
            trade_date = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"

        # 处理列名兼容（原接口用 vol，但数据中可能用 amount 或 vol）
        open_price = float(row.get("open", 0))
        high_price = float(row.get("high", 0))
        low_price = float(row.get("low", 0))
        close_price = float(row.get("close", 0))
        volume = float(row.get("vol", row.get("amount", 0)))

        results.append(
            PriceBar(
                date=trade_date,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
            )
        )

    return results


def prices_to_df(prices: list[Any]) -> pd.DataFrame:
    """
    将 PriceBar 列表转换为 pandas DataFrame
    列: date, open, high, low, close, volume
    """
    if not prices:
        return pd.DataFrame()

    data = []
    for p in prices:
        if hasattr(p, "__dict__"):
            data.append(p.__dict__)
        elif isinstance(p, dict):
            data.append(p)
        else:
            data.append({"date": str(p.date) if hasattr(p, "date") else str(p), "open": 0, "high": 0, "low": 0, "close": 0, "volume": 0})

    df = pd.DataFrame(data)

    # 确保列类型正确
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def get_price_data(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """获取 DataFrame 格式的价格数据"""
    prices = get_prices(ticker, start_date, end_date, api_key)
    return prices_to_df(prices)


# ============================================================
# 财务数据接口
# ============================================================

class FinancialMetrics:
    """
    财务指标数据类，与原 api.py 接口兼容
    适配 Warren Buffett Agent 的所有字段访问
    """

    def __init__(
        self,
        date: str,
        revenue: Optional[float] = None,
        net_margin: Optional[float] = None,
        operating_margin: Optional[float] = None,
        return_on_equity: Optional[float] = None,
        debt_to_equity: Optional[float] = None,
        earnings_per_share: Optional[float] = None,
        free_cash_flow_per_share: Optional[float] = None,
        price_to_earnings_ratio: Optional[float] = None,
        price_to_book_ratio: Optional[float] = None,
        price_to_sales_ratio: Optional[float] = None,
        revenue_growth: Optional[float] = None,
        earnings_growth: Optional[float] = None,
        book_value_growth: Optional[float] = None,
        current_ratio: Optional[float] = None,
        # --- A股扩展字段 ---
        gross_margin: Optional[float] = None,
        return_on_invested_capital: Optional[float] = None,
        asset_turnover: Optional[float] = None,
        price_to_free_cash_flow: Optional[float] = None,
        earnings_per_share_growth: Optional[float] = None,
        free_cash_flow_growth: Optional[float] = None,
        peg_ratio: Optional[float] = None,
        enterprise_value_to_ebitda_ratio: Optional[float] = None,
        interest_coverage: Optional[float] = None,
        market_cap: Optional[float] = None,
        enterprise_value: Optional[float] = None,
    ):
        self.date = date
        self.revenue = revenue
        self.net_margin = net_margin
        self.operating_margin = operating_margin
        self.return_on_equity = return_on_equity
        self.debt_to_equity = debt_to_equity
        self.earnings_per_share = earnings_per_share
        self.free_cash_flow_per_share = free_cash_flow_per_share
        self.price_to_earnings_ratio = price_to_earnings_ratio
        self.price_to_book_ratio = price_to_book_ratio
        self.price_to_sales_ratio = price_to_sales_ratio
        self.revenue_growth = revenue_growth
        self.earnings_growth = earnings_growth
        self.book_value_growth = book_value_growth
        self.current_ratio = current_ratio
        # A股扩展
        self.gross_margin = gross_margin
        self.return_on_invested_capital = return_on_invested_capital
        self.asset_turnover = asset_turnover
        self.price_to_free_cash_flow = price_to_free_cash_flow
        self.earnings_per_share_growth = earnings_per_share_growth
        self.free_cash_flow_growth = free_cash_flow_growth
        self.peg_ratio = peg_ratio
        self.enterprise_value_to_ebitda_ratio = enterprise_value_to_ebitda_ratio
        self.interest_coverage = interest_coverage
        self.market_cap = market_cap
        self.enterprise_value = enterprise_value

    def model_dump(self) -> dict:
        """Pydantic v2 兼容方法（Warren Agent 调用 .model_dump()）"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class FinancialLineItem:
    """
    财务科目数据类，Warren Buffett Agent 使用
    对应 Tushare 财务报表接口
    """

    def __init__(
        self,
        date: str,
        revenue: Optional[float] = None,
        net_income: Optional[float] = None,
        gross_profit: Optional[float] = None,
        capital_expenditure: Optional[float] = None,
        depreciation_and_amortization: Optional[float] = None,
        outstanding_shares: Optional[float] = None,
        total_assets: Optional[float] = None,
        total_liabilities: Optional[float] = None,
        shareholders_equity: Optional[float] = None,
        book_value_per_share: Optional[float] = None,
        dividends_and_other_cash_distributions: Optional[float] = None,
        issuance_or_purchase_of_equity_shares: Optional[float] = None,
        free_cash_flow: Optional[float] = None,
        operating_cash_flow: Optional[float] = None,
        current_assets: Optional[float] = None,
        current_liabilities: Optional[float] = None,
        cash_and_equivalents: Optional[float] = None,
        debt_to_equity: Optional[float] = None,
        earnings_per_share: Optional[float] = None,
        goodwill_and_intangible_assets: Optional[float] = None,
        gross_margin: Optional[float] = None,
        operating_expense: Optional[float] = None,
        operating_income: Optional[float] = None,
        operating_margin: Optional[float] = None,
        research_and_development: Optional[float] = None,
        return_on_invested_capital: Optional[float] = None,
        total_debt: Optional[float] = None,
        ebit: Optional[float] = None,
        ebitda: Optional[float] = None,
    ):
        self.date = date
        self.revenue = revenue
        self.net_income = net_income
        self.gross_profit = gross_profit
        self.capital_expenditure = capital_expenditure
        self.depreciation_and_amortization = depreciation_and_amortization
        self.outstanding_shares = outstanding_shares
        self.total_assets = total_assets
        self.total_liabilities = total_liabilities
        self.shareholders_equity = shareholders_equity
        self.book_value_per_share = book_value_per_share or (shareholders_equity / outstanding_shares if shareholders_equity and outstanding_shares else None)
        self.dividends_and_other_cash_distributions = dividends_and_other_cash_distributions
        self.issuance_or_purchase_of_equity_shares = issuance_or_purchase_of_equity_shares
        self.free_cash_flow = free_cash_flow
        self.operating_cash_flow = operating_cash_flow
        self.current_assets = current_assets
        self.current_liabilities = current_liabilities
        self.cash_and_equivalents = cash_and_equivalents
        self.debt_to_equity = debt_to_equity
        self.earnings_per_share = earnings_per_share or ((net_income or 0) / (outstanding_shares or 1) if outstanding_shares and outstanding_shares != 0 else None)
        self.goodwill_and_intangible_assets = goodwill_and_intangible_assets
        self.gross_margin = gross_margin or ((gross_profit or 0) / (revenue or 1) if revenue and revenue != 0 else None)
        self.operating_expense = operating_expense
        self.operating_income = operating_income
        self.operating_margin = operating_margin or ((operating_income or 0) / (revenue or 1) if revenue and revenue != 0 else None)
        self.research_and_development = research_and_development
        self.return_on_invested_capital = return_on_invested_capital
        self.total_debt = total_debt
        self.working_capital = (current_assets or 0) - (current_liabilities or 0) if (current_assets is not None and current_liabilities is not None) else None
        self.ebit = ebit
        self.ebitda = ebitda

    def model_dump(self) -> dict:
        """兼容 Pydantic model_dump 接口"""
        return self.__dict__

    def recalculate_derived(self):
        """
        在 _set_fields 填充完原始数据后，重新计算所有派生字段
        """
        if self.book_value_per_share is None and self.shareholders_equity is not None and self.outstanding_shares and self.outstanding_shares != 0:
            self.book_value_per_share = self.shareholders_equity / self.outstanding_shares
        if self.earnings_per_share is None and self.net_income is not None and self.outstanding_shares and self.outstanding_shares != 0:
            self.earnings_per_share = self.net_income / self.outstanding_shares
        if self.gross_margin is None and self.gross_profit is not None and self.revenue and self.revenue != 0:
            self.gross_margin = self.gross_profit / self.revenue
        if self.operating_margin is None and self.operating_income is not None and self.revenue and self.revenue != 0:
            self.operating_margin = self.operating_income / self.revenue
        if self.working_capital is None and self.current_assets is not None and self.current_liabilities is not None:
            self.working_capital = self.current_assets - self.current_liabilities


class PriceBar:
    """K线数据类，与原 api.py 接口兼容"""

    def __init__(
        self,
        date: str,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ):
        self.date = date
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    @property
    def time(self):
        """兼容原 api.py 的 time 属性"""
        return self.date


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 8,
    api_key: Optional[str] = None,
) -> list[FinancialMetrics]:
    """
    获取财务指标（替代原 Financial Datasets API）
    使用 Tushare 财务指标接口

    Args:
        ticker: A股代码，如 "600519.SH"
        end_date: 截止日期
        period: 财报周期 ("q1", "q2", "q3", "q4", "annual", "ttm")
        limit: 返回数量
        api_key: 忽略，保留接口兼容性
    """
    pro = get_pro_api()
    if pro is None:
        return []

    ts_code = normalize_ts_code(ticker)

    def _pct(v):
        """Tushare百分比字段：值>1说明是百分数形式(如10.57)，需除100"""
        f = _safe_float(v)
        if f is not None and abs(f) > 1:
            return f / 100
        return f

    # 尝试从 daily_basic 获取最新 PE/PB（含 fallback）
    pe_ttm = None
    pb_ratio = None
    try:
        trade_date = end_date.replace("-", "")
        basic_df = pro.daily_basic(ts_code=ts_code, trade_date=trade_date)
        if basic_df is None or basic_df.empty:
            # fallback: 取最新可用数据
            basic_df = pro.daily_basic(ts_code=ts_code)
        if basic_df is not None and not basic_df.empty:
            pe_ttm = _safe_float(basic_df.iloc[0].get("pe_ttm"))
            pb_ratio = _safe_float(basic_df.iloc[0].get("pb"))
    except Exception:
        pass

    # 获取利润表数据（用来填 revenue 和计算 eps_growth）
    income_rows = {}  # end_date(YYYYMMDD) -> dict
    try:
        inc_df = pro.income(
            ts_code=ts_code,
            start_date=_calc_date(end_date, years=2),
            end_date=end_date.replace("-", ""),
        )
        if inc_df is not None and not inc_df.empty:
            for _, r in inc_df.iterrows():
                ed = str(r.get("end_date", ""))
                income_rows[ed] = {
                    "revenue": _safe_float(r.get("revenue", 0.0)),
                    "net_income": _safe_float(r.get("n_income", 0.0)),
                }
    except Exception:
        pass

    # 获取资产负债表数据（总股本 total_share）
    shares_map = {}  # end_date(YYYYMMDD) -> total_share
    try:
        bs_df = pro.balancesheet(
            ts_code=ts_code,
            start_date=_calc_date(end_date, years=3),
            end_date=end_date.replace("-", ""),
        )
        if bs_df is not None and not bs_df.empty:
            for _, r in bs_df.iterrows():
                ed = str(r.get("end_date", ""))
                shares_map[ed] = _safe_float(r.get("total_share"))
    except Exception:
        pass

    # 获取现金流量表数据（自由现金流 free_cashflow）
    cf_map = {}  # end_date(YYYYMMDD) -> free_cashflow
    try:
        cf_df = pro.cashflow(
            ts_code=ts_code,
            start_date=_calc_date(end_date, years=3),
            end_date=end_date.replace("-", ""),
        )
        if cf_df is not None and not cf_df.empty:
            for _, r in cf_df.iterrows():
                ed = str(r.get("end_date", ""))
                val = _safe_float(r.get("free_cashflow"))
                # Tushare 有重复行(一行实值一行NaN)，保留绝对值更大的那个
                if ed not in cf_map or abs(val) > abs(cf_map[ed]):
                    cf_map[ed] = val
    except Exception:
        pass

    # 获取每日行情数据（收盘价 close）
    close_price = None
    try:
        trade_date = end_date.replace("-", "")
        dly = pro.daily(ts_code=ts_code, start_date=trade_date, end_date=trade_date)
        if dly is not None and not dly.empty:
            close_price = _safe_float(dly.iloc[0].get("close"))
    except Exception:
        pass

    # Tushare fina_indicator（无 period_type 参数，静默忽略）
    try:
        df = pro.fina_indicator(
            ts_code=ts_code,
            start_date=_calc_date(end_date, years=3),
            end_date=end_date.replace("-", ""),
        )
        if df is not None and not df.empty:
            df = df.sort_values("end_date", ascending=False).head(limit)
        else:
            df = pd.DataFrame()
    except Exception as e:
        print(f"[WARN] fina_indicator failed for {ts_code}: {e}")
        df = pd.DataFrame()

    results = []
    if df is not None and not df.empty:
        for _, row in df.iterrows():
            end_date_str = str(row.get("end_date", ""))
            if len(end_date_str) == 8:
                end_date_str = f"{end_date_str[:4]}-{end_date_str[4:6]}-{end_date_str[6:8]}"

            ed_raw = str(row.get("end_date", ""))  # YYYYMMDD
            # 从利润表取 revenue
            inc_row = income_rows.get(ed_raw, {})
            rev = inc_row.get("revenue", 0.0) or 0.0
            ni = inc_row.get("net_income", 0.0) or 0.0

            # debt_to_assets（资产负债率）≠ D/E，转换: D/E = D/A / (1 - D/A)
            d_a = _pct(row.get("debt_to_assets"))
            if d_a is not None and d_a < 1.0:
                de_val = d_a / max(1.0 - d_a, 0.001)
            else:
                de_val = d_a

            rev_growth = _pct(row.get("tr_yoy"))
            eps = _safe_float(row.get("eps"))

            # FCFPS / PTFCF 计算
            fcf = cf_map.get(ed_raw)
            total_shares = shares_map.get(ed_raw)
            fcfps = (fcf / total_shares) if (fcf is not None and total_shares and total_shares != 0) else None
            price_est = (pe_ttm * eps) if (pe_ttm is not None and eps is not None) else close_price
            ptfcf = (price_est / fcfps) if (price_est is not None and fcfps is not None and fcfps != 0) else None

            # PEG ratio 暂放 None，等循环后计算

            results.append(
                FinancialMetrics(
                    date=end_date_str,
                    return_on_equity=_pct(row.get("roe")),
                    debt_to_equity=de_val,
                    net_margin=_pct(row.get("netprofit_margin")),
                    operating_margin=_pct(row.get("op_of_gr")),
                    current_ratio=_safe_float(row.get("current_ratio")),
                    earnings_per_share=eps,
                    price_to_earnings_ratio=pe_ttm,
                    price_to_book_ratio=pb_ratio,
                    revenue_growth=rev_growth,
                    earnings_growth=_pct(row.get("netprofit_yoy")),
                    book_value_growth=_pct(row.get("bps_yoy")),
                    price_to_sales_ratio=None,
                    revenue=rev,
                    # --- 新增字段 ---
                    gross_margin=_pct(row.get("grossprofit_margin")),
                    return_on_invested_capital=_pct(row.get("roic")),
                    asset_turnover=_safe_float(row.get("asset_turnover")),
                    # --- 现金流量字段 ---
                    free_cash_flow_per_share=fcfps,
                    price_to_free_cash_flow=ptfcf,
                )
            )

        # 后处理：计算 eps_growth、PEG、FCF 增长率
        for curr in results:
            # earnings_per_share_growth ≈ netprofit_yoy（已有，不需要跨期对比）
            if curr.earnings_growth is not None:
                curr.earnings_per_share_growth = curr.earnings_growth
            # PEG = PE / (earnings_growth_pct * 100 转为百分比)
            if curr.price_to_earnings_ratio is not None:
                if curr.earnings_growth is not None and curr.earnings_growth != 0:
                    curr.peg_ratio = curr.price_to_earnings_ratio / (curr.earnings_growth * 100)
                elif curr.revenue_growth is not None and curr.revenue_growth != 0:
                    curr.peg_ratio = curr.price_to_earnings_ratio / (curr.revenue_growth * 100)

        # 后处理：计算 FCF 增长率（同季度 YoY）
        for i in range(len(results)):
            curr = results[i]
            if curr.free_cash_flow_per_share is None or curr.free_cash_flow_per_share == 0:
                continue
            curr_date = curr.date  # YYYY-MM-DD
            for j in range(i + 1, len(results)):
                prev = results[j]
                if prev.free_cash_flow_per_share is None or prev.free_cash_flow_per_share == 0:
                    continue
                # 跨年比较同月同日
                if curr_date[5:] == prev.date[5:]:  # 同月同日
                    if prev.date < curr_date:
                        curr.free_cash_flow_growth = (
                            curr.free_cash_flow_per_share - prev.free_cash_flow_per_share
                        ) / abs(prev.free_cash_flow_per_share)
                        break

    return results


def _get_financial_from_income(ts_code: str, end_date: str, limit: int):
    """从利润表数据推导财务指标"""
    pro = get_pro_api()
    if pro is None:
        return pd.DataFrame()

    try:
        df = pro.income(
            ts_code=ts_code,
            start_date=_calc_date(end_date, years=2),
            end_date=end_date.replace("-", ""),
            period_type="q",
        )
        return df
    except Exception:
        return pd.DataFrame()


# ============================================================
# 新闻接口（中文舆情）
# ============================================================

class NewsItem:
    """新闻数据类"""

    def __init__(
        self,
        title: str,
        url: str,
        publisher: str,
        sentiment: str = "neutral",
        published_date: str = "",
    ):
        self.title = title
        self.url = url
        self.publisher = publisher
        self.sentiment = sentiment
        self.published_date = published_date


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: Optional[str] = None,
    limit: int = 100,
    period: str = "ttm",
    api_key: Optional[str] = None,
) -> list[NewsItem]:
    """
    获取公司新闻（中文舆情）
    使用 AKShare 财经新闻接口

    Args:
        ticker: A股代码，如 "600519.SH" 或简称 "贵州茅台"
        end_date: 截止日期
        limit: 返回数量
        api_key: 忽略，保留接口兼容性
    """
    # 解析股票代码
    code, _ = parse_ts_code(ticker)
    stock_name = ticker

    # 尝试从 Tushare 获取新闻
    news_list = []

    # 方法1: 使用 Tushare 公告接口（如果有权限）
    pro = get_pro_api()
    if pro is not None:
        try:
            # 获取近期新闻/公告
            news_df = pro.news(
                start_date=(datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y%m%d"),
                end_date=end_date.replace("-", ""),
                market="cns",
            )
            if news_df is not None and not news_df.empty:
                for _, row in news_df.head(limit).iterrows():
                    # 简单情感判断（关键词法）
                    title = str(row.get("title", ""))
                    content = str(row.get("content", ""))[:200]
                    sentiment = _simple_sentiment(title + content)
                    news_list.append(
                        NewsItem(
                            title=title,
                            url=str(row.get("url", "")),
                            publisher=str(row.get("source", "未知")),
                            sentiment=sentiment,
                            published_date=str(row.get("datetime", "")),
                        )
                    )
        except Exception:
            pass

    # 方法2: 使用 AKShare 财经新闻
    if len(news_list) < 5:
        try:
            df = ak.stock_news_em(symbol=code)
            if df is not None and not df.empty:
                for _, row in df.head(limit).iterrows():
                    title = str(row.get("新闻标题", row.get("title", "")))
                    content = str(row.get("新闻内容", row.get("content", "")))[:200]
                    sentiment = _simple_sentiment(title + content)
                    news_list.append(
                        NewsItem(
                            title=title,
                            url=str(row.get("链接", row.get("url", ""))),
                            publisher=str(row.get("文章来源", "财经网")),
                            sentiment=sentiment,
                            published_date=str(row.get("发布时间", "")),
                        )
                    )
        except Exception:
            pass

    return news_list[:limit]


def _simple_sentiment(text: str) -> str:
    """简单的中文情感判断（关键词法）"""
    text = text.lower()
    positive = ["涨", "突破", "增长", "盈利", "业绩", "看好", "买入", "增持", "超预期", "景气", "复苏", "创新高", "加速"]
    negative = ["跌", "亏损", "风险", "减持", "降级", "预警", "危机", "违约", "造假", "处罚", "下修", "衰退"]

    pos_count = sum(1 for w in positive if w in text)
    neg_count = sum(1 for w in negative if w in text)

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"


# ============================================================
# 融资融券数据（替代美股 Insider Trades）
# ============================================================

class MarginTrade:
    """融资融券数据"""

    def __init__(
        self,
        trade_date: str,
        margin_balance: float,
        short_balance: float,
        net_balance: float,
        change_pct: float,
    ):
        self.trade_date = trade_date
        self.margin_balance = margin_balance  # 融资余额
        self.short_balance = short_balance  # 融券余额
        self.net_balance = net_balance
        self.change_pct = change_pct
        self.transaction_shares = margin_balance  # 兼容接口
        self.transaction_value = net_balance  # 兼容接口（净额≈交易价值）
        self.transaction_type = 'margin' if margin_balance > 0 else 'short'


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: Optional[str] = None,
    limit: int = 1000,
    api_key: Optional[str] = None,
) -> list[MarginTrade]:
    """
    获取融资融券数据（A股特色，替代美股 Insider Trades）
    融资余额增加 → 杠杆做多情绪上升
    融券余额增加 → 做空情绪/看跌情绪

    Args:
        ticker: A股代码
        end_date: 截止日期
        start_date: 开始日期
        limit: 返回数量
        api_key: 忽略
    """
    pro = get_pro_api()
    if pro is None:
        return []

    ts_code = normalize_ts_code(ticker)

    if start_date is None:
        start_date = _calc_date(end_date, years=1)

    try:
        df = pro.margin_detail(
            ts_code=ts_code,
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
        )
    except Exception:
        try:
            df = pro.margin(
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )
            df = df[df["ts_code"] == ts_code]
        except Exception as e:
            print(f"[WARN] Margin data failed for {ts_code}: {e}")
            return []

    if df is None or df.empty:
        return []

    df = df.sort_values("trade_date", ascending=False).head(limit)

    results = []
    for _, row in df.iterrows():
        trade_date = str(row.get("trade_date", ""))
        if len(trade_date) == 8:
            trade_date = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"

        results.append(
            MarginTrade(
                trade_date=trade_date,
                margin_balance=_safe_float(row.get("rzye")),  # 融资余额
                short_balance=_safe_float(row.get("rqye")),  # 融券余额
                net_balance=_safe_float(row.get("rqye")) - _safe_float(row.get("rzye")),
                change_pct=_safe_float(row.get("rzmre")),  # 融资净买入
            )
        )

    return results


# ============================================================
# 涨跌停数据（A股特色）
# ============================================================

class LimitUpData:
    """涨跌停数据"""

    def __init__(
        self,
        ts_code: str,
        name: str,
        trade_date: str,
        close: float,
        pct_chg: float,
        limit_up: bool,
        limit_down: bool,
        turn_rate: float,
    ):
        self.ts_code = ts_code
        self.name = name
        self.trade_date = trade_date
        self.close = close
        self.pct_chg = pct_chg
        self.limit_up = limit_up
        self.limit_down = limit_down
        self.turn_rate = turn_rate


def get_limit_list(trade_date: str) -> list[LimitUpData]:
    """
    获取涨跌停统计（全部A股）
    用于识别涨停敢死队、跌停风险股

    Args:
        trade_date: YYYY-MM-DD
    """
    pro = get_pro_api()
    if pro is None:
        return []

    try:
        df = pro.limit_list_d(
            trade_date=trade_date.replace("-", ""),
            limit_type="U",  # 涨停
        )
    except Exception:
        try:
            df = pro.limit_list(
                trade_date=trade_date.replace("-", ""),
            )
        except Exception as e:
            print(f"[WARN] Limit list failed: {e}")
            return []

    if df is None or df.empty:
        return []

    results = []
    for _, row in df.iterrows():
        trade_date_str = str(row.get("trade_date", ""))
        if len(trade_date_str) == 8:
            trade_date_str = f"{trade_date_str[:4]}-{trade_date_str[4:6]}-{trade_date_str[6:8]}"

        pct = _safe_float(row.get("pct_chg", 0))
        results.append(
            LimitUpData(
                ts_code=str(row.get("ts_code", "")),
                name=str(row.get("name", "")),
                trade_date=trade_date_str,
                close=_safe_float(row.get("close", 0)),
                pct_chg=pct,
                limit_up=pct >= 9.9,
                limit_down=pct <= -9.9,
                turn_rate=_safe_float(row.get("turn_rate", 0)),
            )
        )

    return results


# ============================================================
# 北向资金（外资流向）
# ============================================================

def get_north_money(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 20,
) -> pd.DataFrame:
    """
    获取北向资金流向（沪深港通）
    外资情绪参考指标
    """
    pro = get_pro_api()
    if pro is None:
        return pd.DataFrame()

    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

    try:
        df = pro.moneyflow_hsgt(
            start_date=start_date,
            end_date=end_date,
        )
        if df is not None:
            df = df.sort_values("trade_date", ascending=False).head(limit)
        return df
    except Exception as e:
        print(f"[WARN] North money data failed: {e}")
        return pd.DataFrame()


# ============================================================
# 股票基本信息（ST过滤、退市风险）
# ============================================================

class AStockInfo:
    """A股股票基本信息（兼容接口）"""

    def __init__(
        self,
        ts_code: str,
        name: str,
        industry: str,
        market: str,
        list_date: str,
        is_st: bool,
        status: str = "L",
    ):
        self.ts_code = ts_code
        self.name = name
        self.industry = industry
        self.market = market
        self.list_date = list_date
        self.is_st = is_st
        self.status = status


def get_stock_info(ticker: str) -> Optional[AStockInfo]:
    """
    获取股票基本信息（使用缓存，速度快）
    委托 sector_map.get_stock_info
    """
    info = _sector_get_info(ticker)
    if info is None:
        return None
    return AStockInfo(
        ts_code=info["ts_code"],
        name=info["name"],
        industry=info["industry"],
        market=info["market"],
        list_date=info["list_date"],
        is_st=info["is_st"],
        status="L",
    )


def filter_st_stocks(
    tickers: list[str],
    min_market_cap: float = 5.0,
    exclude_st: bool = True,
    exclude_new: bool = True,
) -> list[str]:
    """
    过滤 ST/*ST + 新股 + 市值门槛（使用批量缓存，极速）
    委托 sector_map.filter_stocks

    Args:
        tickers: 股票列表
        min_market_cap: 最低流通市值（亿元，默认5亿）
        exclude_st: 剔除 ST/*ST
        exclude_new: 剔除新股（<90天）

    Returns:
        过滤后股票列表
    """
    return _sector_filter(
        tickers,
        min_market_cap=min_market_cap,
        exclude_st=exclude_st,
        exclude_new=exclude_new,
    )


def get_market_cap(
    ticker: str,
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
) -> float:
    """获取总市值（亿元）
    兼容 Warren Agent 调用签名：get_market_cap(ticker, end_date, api_key)
    """
    pro = get_pro_api()
    if pro is None:
        return 0.0

    ts_code = normalize_ts_code(ticker)
    trade_date = end_date.replace("-", "") if end_date else datetime.now().strftime("%Y%m%d")

    try:
        df = pro.daily_basic(ts_code=ts_code, trade_date=trade_date)
        if df is None or df.empty:
            # 尝试取最近交易日
            df = pro.daily_basic(ts_code=ts_code)
        if df is None or df.empty:
            return 0.0
        # total_mv 单位：万元 → 除10000 → 亿元
        return _safe_float(df.iloc[0].get("total_mv", 0)) / 10000
    except Exception:
        return 0.0


def get_float_market_cap(ticker: str) -> float:
    """获取流通市值（亿元）"""
    pro = get_pro_api()
    if pro is None:
        return 0.0

    ts_code = normalize_ts_code(ticker)

    try:
        df = pro.daily_basic(
            ts_code=ts_code,
            trade_date=datetime.now().strftime("%Y%m%d"),
        )
        if df is None or df.empty:
            return 0.0
        return _safe_float(df.iloc[0].get("circ_mv", 0)) / 10000
    except Exception:
        return 0.0


# ============================================================
# 行业/板块选股
# ============================================================

def get_stocks_by_sector(sector: str) -> list[str]:
    """
    按板块名获取成分股（使用申万行业分类系统）
    内部委托 sector_map.get_stocks_by_sector

    Args:
        sector: 板块名，如 "白酒", "半导体", "医药", "新能源"

    Returns:
        排序后的股票代码列表，如 ["600519.SH", "000858.SZ", ...]
    """
    return _sector_get_stocks(sector)


def get_16_sector_stocks() -> dict[str, list[str]]:
    """
    获取14个核心板块的成分股
    返回: {板块名: [股票代码列表]}
    """
    return {sector: get_stocks_by_sector(sector) for sector in SECTOR_POOL.keys()}


# ============================================================
# 搜索财务科目（利润表、资产负债表）
# ============================================================

def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "annual",
    limit: int = 10,
    api_key: Optional[str] = None,
) -> list[FinancialLineItem]:
    """
    搜索财务会计科目（替代原 Financial Datasets API）
    使用 Tushare 财务报表接口

    Warren Agent 传入英文字段名，直接映射到 Tushare 英文列名

    Args:
        ticker: A股代码
        line_items: 要查询的科目名称列表（Warren 传英文）
        end_date: 截止日期
        period: 财报周期 ("annual" / "q1" / "q2" / "q3" / "q4")
        limit: 返回期数
        api_key: 忽略

    Returns:
        list[FinancialLineItem]: 每期一个对象，支持属性访问
    """
    pro = get_pro_api()
    if pro is None:
        return []

    ts_code = normalize_ts_code(ticker)
    start_date = _calc_date(end_date, years=3)

    # Warren 传入的英文字段名 → Tushare 列名
    field_map = {
        # 利润表
        "revenue": "revenue",
        "net_income": "n_income",
        "gross_profit": "gross_profit",
        "depreciation_and_amortization": "da",
        "operating_expense": "oper_exp",
        "operating_income": "operate_profit",
        "research_and_development": "rd_exp",
        "ebit": "ebit",
        "ebitda": "ebitda",
        # 资产负债表
        "total_assets": "total_assets",
        "total_liabilities": "total_liab",
        "shareholders_equity": "total_hldr_eqy_exc_min_int",
        "current_assets": "total_cur_assets",
        "current_liabilities": "total_cur_liab",
        "outstanding_shares": "total_share",
        "total_debt": "total_liab",  # 近似
        "debt_to_equity": None,  # 后续从财务指标计算
        "cash_and_equivalents": "money_cap",
        "goodwill_and_intangible_assets": None,  # 后续从 intan_assets+goodwill 计算
        # 现金流量表
        "free_cash_flow": "free_cashflow",
        "operating_cash_flow": "n_cashflow_act",
        "capital_expenditure": "c_pay_acq_const_fiolta",
        # 特殊（利润表基本EPS）
        "earnings_per_share": "basic_eps",
        "dividends_and_other_cash_distributions": "div_dist",
        "issuance_or_purchase_of_equity_shares": "buy_back",
    }
    # 反向映射：Tushare列名 → FinancialLineItem 属性名
    reverse_map = {v: k for k, v in field_map.items() if v is not None}

    # 收集需要的 Tushare 列（排除 None 映射的字段）
    needed_ts = set()
    for item in line_items:
        if item in field_map and field_map[item] is not None:
            needed_ts.add(field_map[item])

    results: list[FinancialLineItem] = []

    def _set_fields(obj, row, ts_cols: list[str]):
        """将Tushare数据行的列赋值到对象的正确属性名"""
        for ts_col in ts_cols:
            attr = reverse_map.get(ts_col, ts_col)
            val = _safe_float(row.get(ts_col))
            setattr(obj, attr, val)

    def _date_str(row) -> str:
        s = str(row.get("end_date", ""))
        if len(s) == 8:
            s = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return s

    # 1) 利润表（收入、净利润、毛利、研发费用、营业利润等）
    income_ts = ["revenue", "n_income", "gross_profit", "oper_exp", "operate_profit", "rd_exp", "basic_eps", "da", "ebit", "ebitda"]
    avail_income = [c for c in income_ts if c in needed_ts]
    if avail_income:
        try:
            df = pro.income(
                ts_code=ts_code,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )
            if df is not None and not df.empty:
                df = df.sort_values("end_date", ascending=False).head(limit)
                for _, row in df.iterrows():
                    ds = _date_str(row)
                    existing = next((r for r in results if r.date == ds), None)
                    if existing is None:
                        existing = FinancialLineItem(date=ds)
                        results.append(existing)
                    _set_fields(existing, row, avail_income)
        except Exception as e:
            print(f"[WARN] Income: {e}")

    # 2) 资产负债表（全部可用列）
    balance_ts = [
        "total_assets", "total_liab", "total_hldr_eqy_exc_min_int",
        "total_cur_assets", "total_cur_liab", "total_share",
        "money_cap",
    ]
    avail_balance = [c for c in balance_ts if c in needed_ts]
    if avail_balance:
        try:
            df = pro.balancesheet(
                ts_code=ts_code,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )
            if df is not None and not df.empty:
                df = df.sort_values("end_date", ascending=False).head(limit)
                for _, row in df.iterrows():
                    ds = _date_str(row)
                    existing = next((r for r in results if r.date == ds), None)
                    if existing is None:
                        existing = FinancialLineItem(date=ds)
                        results.append(existing)
                    _set_fields(existing, row, avail_balance)
        except Exception as e:
            print(f"[WARN] Balance sheet: {e}")

    # 3) 现金流量表（全部可用列）
    cash_ts = ["free_cashflow", "n_cashflow_act", "c_pay_acq_const_fiolta"]
    avail_cash = [c for c in cash_ts if c in needed_ts]
    if avail_cash:
        try:
            df = pro.cashflow(
                ts_code=ts_code,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )
            if df is not None and not df.empty:
                df = df.sort_values("end_date", ascending=False).head(limit)
                # 去重：同日期多行时保留非零 free_cashflow 的行
                if "free_cashflow" in df.columns:
                    df = df.sort_values("free_cashflow", key=abs, ascending=False).drop_duplicates(subset=["end_date"])
                for _, row in df.iterrows():
                    ds = _date_str(row)
                    existing = next((r for r in results if r.date == ds), None)
                    if existing is None:
                        existing = FinancialLineItem(date=ds)
                        results.append(existing)
                    _set_fields(existing, row, avail_cash)
        except Exception as e:
            print(f"[WARN] Cash flow: {e}")

    # 4) 后处理：重算派生字段
    for item in results:
        item.recalculate_derived()

    # 按日期排序（最新在前）
    results.sort(key=lambda x: x.date, reverse=True)
    return results


# ============================================================
# 工具函数
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    """安全转换为 float"""
    if value is None:
        return default
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (ValueError, TypeError):
        return default


def _calc_date(end_date: str, years: int = 1) -> str:
    """计算起始日期"""
    try:
        dt = datetime.strptime(end_date, "%Y-%m-%d")
        dt = dt.replace(year=dt.year - years)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return end_date


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("A股数据接口测试")
    print("=" * 60)

    # 测试 Tushare 连接
    print("\n[1] Tushare 连接测试...")
    pro = get_pro_api()
    if pro:
        print("  ✅ Tushare 连接成功")
    else:
        print("  ❌ Tushare 连接失败")
        sys.exit(1)

    # 测试价格数据（贵州茅台）
    print("\n[2] 价格数据测试 (600519.SH)...")
    prices = get_prices("600519.SH", "2026-03-01", "2026-04-25")
    print(f"  获取 {len(prices)} 条数据")
    if prices:
        latest = prices[-1]
        print(f"  最新: {latest.date} 收盘价={latest.close}")

    # 测试财务数据
    print("\n[3] 财务数据测试 (600519.SH)...")
    metrics = get_financial_metrics("600519.SH", "2026-03-31", limit=4)
    print(f"  获取 {len(metrics)} 期数据")
    if metrics:
        m = metrics[0]
        print(f"  最新: ROE={m.return_on_equity}, PE={m.price_to_earnings_ratio}, 净利率={m.net_margin}")

    # 测试新闻
    print("\n[4] 新闻测试 (600519.SH)...")
    news = get_company_news("600519.SH", "2026-04-25", limit=3)
    print(f"  获取 {len(news)} 条新闻")

    # 测试 ST 过滤
    print("\n[5] ST 过滤测试...")
    test_stocks = ["600519.SH", "000858.SZ", "600687.SH"]  # 600687是ST股
    filtered = filter_st_stocks(test_stocks)
    print(f"  原始: {test_stocks}")
    print(f"  过滤后: {filtered}")

    # 测试市值
    print("\n[6] 流通市值测试 (600519.SH)...")
    cap = get_float_market_cap("600519.SH")
    print(f"  流通市值: {cap:.2f} 亿元")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
