from __future__ import annotations
import os
import sys
import io
from datetime import datetime, timedelta
from typing import Optional, Any

import pandas as pd
import numpy as np

import tushare as ts
import akshare as ak

from src.tools.data_classes import (
    FinancialMetrics, FinancialLineItem, PriceBar,
    NewsItem, MarginTrade, LimitUpData, AStockInfo,
)
from src.tools.data_cleaner import _safe_float
from src.utils.sector_map import (
    SECTOR_INDUSTRY_MAP,
    get_stocks_by_sector as _sector_get_stocks,
    filter_stocks as _sector_filter,
    get_stocks_by_sector as _sector_get_all,
    get_stock_info as _sector_get_info,
)

# Windows 编码修复
if sys.platform == 'win32' and __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

# Token 配置
TUSHARE_TOKEN = os.getenv('TUSHARE_PRO_TOKEN') or os.getenv('TUSHARE_PRO')
if not TUSHARE_TOKEN:
    raise RuntimeError(
        "请在 .env 文件中设置 TUSHARE_PRO_TOKEN，或执行: export TUSHARE_PRO_TOKEN=你的token"
    )

# 板块池
SECTOR_POOL: dict[str, list[str]] = SECTOR_INDUSTRY_MAP

_pro_api = None

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
            adj='qfq',  # 前复权 — 技术指标不因除权失真
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


# ── 申万行业指数表现（带缓存）──────────────────────
_SECTOR_PERF_CACHE: dict = {}
_SECTOR_INDEX_MAP = {
    "\u534a\u5bfc\u4f53": "801081.SI", "\u94f6\u884c": "801780.SI",
    "\u623f\u5730\u4ea7": "801180.SI", "\u533b\u836f": "801150.SI",
    "\u98df\u54c1\u996e\u6599": "801120.SI", "\u65b0\u80fd\u6e90": "801730.SI",
    "\u519b\u5de5": "801740.SI", "\u8ba1\u7b97\u673a": "801750.SI",
    "\u901a\u4fe1": "801770.SI", "\u5316\u5de5": "801030.SI",
    "\u6709\u8272\u91d1\u5c5e": "801050.SI", "\u673a\u68b0\u8bbe\u5907": "801890.SI",
    "\u5efa\u7b51\u5efa\u6750": "801710.SI", "\u6c7d\u8f66": "801880.SI",
    "\u7535\u529b": "801161.SI", "\u4f20\u5a92": "801760.SI",
    "\u94a2\u94c1": "801040.SI", "\u6d88\u8d39\u7535\u5b50": "801085.SI",
}


def get_sector_index_perf(ts_code: str, cache_minutes: int = 60) -> dict:
    """
    \u83b7\u53d6\u80a1\u7968\u6240\u5c5e\u677f\u5757\u7684\u7533\u4e07\u884c\u4e1a\u6307\u6570\u8fd1\u671f\u8868\u73b0

    \u901a\u8fc7 sw_daily \u63a5\u53e3\u67e5\u7533\u4e07\u884c\u4e1a\u6307\u6570\uff0c\u8ba1\u7b971d/5d/20d\u6da8\u8dcc\u5e45\u5e76\u8f6c\u5206\u6570\u3002
    \u6a21\u5757\u7ea7\u7f13\u5b58\u907f\u514d\u91cd\u590dAPI\u8c03\u7528\u3002

    Returns: {
        sector, index_code,
        perf_1d, perf_5d, perf_20d,
        perf_score,  # 0-100, \u57fa\u4e8e20d\u6da8\u8dcc\u5e45
    }
    """
    from src.utils.sector_map import SECTOR_INDUSTRY_MAP as SMAP
    import time

    result = {"sector": "unknown", "index_code": "",
              "perf_1d": 0, "perf_5d": 0, "perf_20d": 0, "perf_score": 50}

    # \u627e\u677f\u5757
    info = get_stock_info(ts_code)
    if not info or not info.industry:
        return result

    industry = info.industry
    sector_name = "other"
    for name, keywords in SMAP.items():
        if any(kw in industry for kw in keywords):
            sector_name = name
            break
    result["sector"] = sector_name

    # \u7f13\u5b58\u68c0\u67e5
    now = time.time()
    cached = _SECTOR_PERF_CACHE.get(sector_name)
    if cached and (now - cached["_ts"]) < cache_minutes * 60:
        return {k: v for k, v in cached.items() if k != "_ts"}

    # \u67e5\u6307\u6570
    index_code = _SECTOR_INDEX_MAP.get(sector_name, "")
    result["index_code"] = index_code
    if not index_code:
        _SECTOR_PERF_CACHE[sector_name] = {"_ts": now, **result}
        return result

    try:
        pro = get_pro_api()
        df = pro.sw_daily(
            ts_code=index_code,
            start_date=(datetime.now() - timedelta(days=60)).strftime("%Y%m%d"),
            end_date=datetime.now().strftime("%Y%m%d"),
        )
        if df is not None and not df.empty and "close" in df.columns:
            df = df.sort_values("trade_date")
            closes = df["close"].values

            def _chg(n):
                return (closes[-1] - closes[-n]) / closes[-n] * 100 if len(closes) >= n else 0

            result["perf_1d"] = round(_chg(2), 2)
            result["perf_5d"] = round(_chg(6), 2)
            result["perf_20d"] = round(_chg(21), 2)

            p = result["perf_20d"]
            if p > 10: result["perf_score"] = 85
            elif p > 5: result["perf_score"] = 70
            elif p > 2: result["perf_score"] = 60
            elif p > 0: result["perf_score"] = 55
            elif p > -5: result["perf_score"] = 40
            elif p > -10: result["perf_score"] = 25
            else: result["perf_score"] = 10
    except Exception:
        pass

    _SECTOR_PERF_CACHE[sector_name] = {"_ts": now, **result}
    return result


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


def get_index_data(ts_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    获取指数行情数据（通过 Tushare index_daily）

    返回格式与 get_price_data 保持一致：columns = [date, open, high, low, close, vol, amount]
    """
    pro = get_pro_api()
    if not start_date:
        start_date = "20200101"
    if not end_date:
        from datetime import datetime
        end_date = datetime.now().strftime("%Y%m%d")
    try:
        df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"trade_date": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [get_index_data] 错误 {ts_code}: {e}")
        return pd.DataFrame()
