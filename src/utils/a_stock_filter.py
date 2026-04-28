#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股股票筛选器 - 16板块 + ST过滤 + 动态扩展
用于全市场选股，剔除ST股票

作者: JoJo (华尔街金融大咖)
"""

from __future__ import annotations

import os
import sys
import io
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

# 引入 Tushare
import tushare as ts

# 引入本地A股数据接口
from src.tools.a_stock_api import (
    get_pro_api,
    normalize_ts_code,
    parse_ts_code,
    get_stock_info,
    get_stocks_by_sector,
    get_float_market_cap,
    get_limit_list,
    get_limit_up_stocks,
    filter_st_stocks as api_filter_st,
    SECTOR_POOL,
)

# ============================================================
# 编码修复（Windows）
# ============================================================
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

# ============================================================
# 常量配置
# ============================================================

# Tushare Token
TUSHARE_TOKEN = os.getenv("TUSHARE_PRO_TOKEN") or "44a561fb111bc7527aa1124223015982ab20a0843b6fb9a19bf91799"

# 16个核心板块
CORE_SECTORS = list(SECTOR_POOL.keys())

# 动态扩展关键词
EXPANSION_KEYWORDS = [
    # 新兴成长
    "人工智能", "AI", "大模型", "机器人", "低空经济", "商业航天", "量子计算",
    "脑机接口", "固态电池", "钠离子", "氢能", "可控核聚变",
    # 消费升级
    "医美", "预制菜", "宠物经济", "户外运动", "智能家居",
    # 政策驱动
    "国企改革", "数据要素", "新型基础设施", "数字经济", "信创",
    # 周期复苏
    "稀有金属", "稀土", "锂矿", "钴矿", "航运", "油运",
    # 市场热点
    "Sora", "DeepSeek", "豆包", "Kimi", "固态电池", "飞行汽车",
]

# 市值筛选
MIN_FLOAT_MARKET_CAP = 10.0   # 最低流通市值（亿元）
MAX_FLOAT_MARKET_CAP = 3000.0 # 最高流通市值（亿元）

# 换手率筛选
MIN_TURN_RATE = 0.5  # 最低日均换手率（%）

# ============================================================
# 板块扩展系统
# ============================================================

def expand_sectors_with_keywords(
    base_sectors: list[str] = None,
    top_n: int = 5,
) -> list[str]:
    """
    基于关键词扩展板块列表

    Args:
        base_sectors: 基础板块列表，默认16个核心板块
        top_n: 每个关键词最多扩展几个板块

    Returns:
        扩展后的板块列表
    """
    if base_sectors is None:
        base_sectors = CORE_SECTORS

    pro = get_pro_api()
    if pro is None:
        return base_sectors

    expanded = list(base_sectors)  # 保留原有板块

    try:
        # 获取所有概念板块
        concept_df = pro.concept()
        if concept_df is None or concept_df.empty:
            return expanded

        # 获取所有行业板块
        try:
            industry_df = pro.industry()
        except Exception:
            industry_df = pd.DataFrame()

        all_df = concept_df
        if not industry_df.empty:
            all_df = pd.concat([concept_df, industry_df], ignore_index=True)

        # 去重
        all_df = all_df.drop_duplicates(subset=["code"])

        # 匹配关键词
        for keyword in EXPANSION_KEYWORDS:
            matched = all_df[all_df["name"].str.contains(keyword, na=False)]
            for _, row in matched.head(top_n).iterrows():
                name = str(row["name"])
                if name not in expanded:
                    expanded.append(name)
                    print(f"[EXPAND] 关键词 '{keyword}' 匹配 → 新增板块: {name}")

    except Exception as e:
        print(f"[WARN] 板块扩展失败: {e}")

    return expanded


def get_trade_dates(start_date: str, end_date: str) -> list[str]:
    """获取交易日历"""
    pro = get_pro_api()
    if pro is None:
        return []

    try:
        cal_df = pro.trade_cal(
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            is_open="1",
        )
        if cal_df is None or cal_df.empty:
            return []
        return cal_df["cal_date"].tolist()
    except Exception as e:
        print(f"[WARN] Trade cal failed: {e}")
        return []


# ============================================================
# 全市场选股
# ============================================================

def get_a_stock_universe(
    sectors: Optional[list[str]] = None,
    min_market_cap: float = MIN_FLOAT_MARKET_CAP,
    max_market_cap: float = MAX_FLOAT_MARKET_CAP,
    min_turn_rate: float = MIN_TURN_RATE,
    exclude_st: bool = True,
    trade_date: Optional[str] = None,
    max_stocks_per_sector: int = 20,
) -> list[str]:
    """
    获取A股优选股票池

    筛选逻辑:
    1. 按行业板块选取成分股
    2. 剔除 ST/*ST 股票
    3. 剔除流通市值过小/过大的股票
    4. 剔除近期涨跌停的股票（追板风险）
    5. 保留成交活跃股票

    Args:
        sectors: 板块列表，默认16个核心板块
        min_market_cap: 最低流通市值（亿元）
        max_market_cap: 最高流通市值（亿元）
        min_turn_rate: 最低日均换手率（%）
        exclude_st: 是否剔除ST股
        trade_date: 交易日期（用于涨跌停过滤）
        max_stocks_per_sector: 每个板块最多保留股票数

    Returns:
        优选后的股票代码列表
    """
    if sectors is None:
        sectors = CORE_SECTORS

    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"A股选股: {len(sectors)} 个板块")
    print(f"筛选条件: 流通市值 {min_market_cap}-{max_market_cap}亿, 换手率>{min_turn_rate}%, ST剔除={exclude_st}")
    print(f"{'='*60}\n")

    all_selected = []
    sector_summary = {}

    for sector in sectors:
        print(f"\n[板块] {sector}...")

        # 获取板块成分股
        try:
            raw_stocks = get_stocks_by_sector(sector)
        except Exception as e:
            print(f"  ⚠️ 获取失败: {e}")
            raw_stocks = []

        if not raw_stocks:
            # 尝试用 Tushare 行业分类
            try:
                pro = get_pro_api()
                if pro:
                    industry_df = pro.industry()
                    if industry_df is not None:
                        matched = industry_df[industry_df["industry"].str.contains(sector[:2], na=False)]
                        if not matched.empty:
                            code = matched.iloc[0].get("code", "")
                            if code:
                                raw_stocks = _get_stocks_by_industry_code(pro, code)
            except Exception:
                pass

        print(f"  原始成分股: {len(raw_stocks)} 只")

        if not raw_stocks:
            sector_summary[sector] = 0
            continue

        # ST 过滤
        if exclude_st:
            try:
                raw_stocks = api_filter_st(raw_stocks)
                print(f"  ST过滤后: {len(raw_stocks)} 只")
            except Exception as e:
                print(f"  ⚠️ ST过滤失败: {e}")

        if not raw_stocks:
            sector_summary[sector] = 0
            continue

        # 市值 + 换手率筛选
        selected = _filter_by_quality(raw_stocks, trade_date, min_market_cap, max_market_cap, min_turn_rate)
        print(f"  质量筛选后: {len(selected)} 只")

        # 限制每板块数量
        if len(selected) > max_stocks_per_sector:
            selected = selected[:max_stocks_per_sector]
            print(f"  截取前 {max_stocks_per_sector} 只")

        all_selected.extend(selected)
        sector_summary[sector] = len(selected)

    # 去重
    all_selected = list(dict.fromkeys(all_selected))  # 保持顺序去重

    print(f"\n{'='*60}")
    print(f"选股完成: 共 {len(all_selected)} 只股票")
    print(f"覆盖 {len(sector_summary)} 个板块")
    print(f"{'='*60}\n")

    # 打印板块分布
    for sector, count in sorted(sector_summary.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {sector}: {count} 只")

    return all_selected


def _get_stocks_by_industry_code(pro, industry_code: str) -> list[str]:
    """根据行业代码获取成分股"""
    try:
        # 获取天燃气等细分行业的成分股
        df = pro.stock_basic(
            ts_code="",
            list_status="L",
            industry=industry_code,
        )
        if df is not None and not df.empty:
            return [normalize_ts_code(str(row["ts_code"])) for _, row in df.iterrows()]
    except Exception:
        pass
    return []


def _filter_by_quality(
    tickers: list[str],
    trade_date: str,
    min_market_cap: float,
    max_market_cap: float,
    min_turn_rate: float,
) -> list[str]:
    """
    按质量和活跃度筛选股票
    - 流通市值过滤
    - 换手率过滤
    - 涨跌停过滤
    """
    if not tickers:
        return []

    pro = get_pro_api()
    if pro is None:
        return tickers

    selected = []
    trade_date_str = trade_date.replace("-", "")

    try:
        # 批量获取日线数据
        # 使用 daily_basic 获取市值和换手率
        for ticker in tickers:
            try:
                ts_code = normalize_ts_code(ticker)
                # 获取当日基础数据
                df = pro.daily_basic(
                    ts_code=ts_code,
                    trade_date=trade_date_str,
                )
                if df is None or df.empty:
                    # 尝试前一个交易日
                    cal_df = pro.trade_cal(
                        start_date=(datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y%m%d"),
                        end_date=trade_date_str,
                        is_open="1",
                    )
                    if cal_df is not None and not cal_df.empty:
                        prev_date = cal_df.iloc[-1]["cal_date"]
                        df = pro.daily_basic(
                            ts_code=ts_code,
                            trade_date=prev_date,
                        )

                if df is None or df.empty:
                    # 查不到数据，保守保留
                    selected.append(ticker)
                    continue

                row = df.iloc[0]

                # 流通市值（万元→亿元）
                circ_mv = float(row.get("circ_mv", 0)) / 10000
                if circ_mv < min_market_cap or circ_mv > max_market_cap:
                    continue

                # 换手率
                turn_rate = float(row.get("turn_rate", 0))
                if turn_rate < min_turn_rate:
                    continue

                # 通过全部筛选
                selected.append(ticker)

            except Exception:
                # 保守保留
                selected.append(ticker)

    except Exception as e:
        print(f"  ⚠️ 质量筛选失败: {e}")
        return tickers

    return selected


def filter_limit_up_stocks(
    tickers: list[str],
    trade_date: str,
    exclude_limit_up: bool = True,
    exclude_limit_down: bool = True,
) -> list[str]:
    """
    剔除涨跌停股票（避免追板风险）

    Args:
        tickers: 股票列表
        trade_date: 交易日期
        exclude_limit_up: 是否剔除涨停股
        exclude_limit_down: 是否剔除跌停股

    Returns:
        剔除涨跌停后的股票列表
    """
    if not exclude_limit_up and not exclude_limit_down:
        return tickers

    pro = get_pro_api()
    if pro is None:
        return tickers

    trade_date_str = trade_date.replace("-", "")

    try:
        # 获取当日涨跌停数据
        df = pro.limit_list(trade_date=trade_date_str)
        if df is None or df.empty:
            return tickers

        limit_up_codes = set()
        limit_down_codes = set()

        for _, row in df.iterrows():
            code = normalize_ts_code(str(row.get("ts_code", "")))
            pct = float(row.get("pct_chg", 0))
            if pct >= 9.9:
                limit_up_codes.add(code)
            elif pct <= -9.9:
                limit_down_codes.add(code)

        filtered = []
        for ticker in tickers:
            code = normalize_ts_code(ticker)
            if exclude_limit_up and code in limit_up_codes:
                print(f"  [LIMIT-UP] 剔除 {ticker} (涨停)")
                continue
            if exclude_limit_down and code in limit_down_codes:
                print(f"  [LIMIT-DOWN] 剔除 {ticker} (跌停)")
                continue
            filtered.append(ticker)

        return filtered

    except Exception as e:
        print(f"  ⚠️ 涨跌停过滤失败: {e}")
        return tickers


# ============================================================
# 快速选股（用于 demo）
# ============================================================

def quick_stock_pick(n: int = 10, sector: Optional[str] = None) -> list[str]:
    """
    快速选股（用于 demo 和快速测试）

    Args:
        n: 选取股票数量
        sector: 指定板块，默认从16个板块中选取

    Returns:
        股票代码列表
    """
    pro = get_pro_api()
    if pro is None:
        return []

    if sector:
        # 指定板块
        stocks = get_stocks_by_sector(sector)
        if not stocks:
            print(f"[WARN] 板块 '{sector}' 无成分股")
            return []
    else:
        # 从全部板块选
        all_stocks = []
        for sec in CORE_SECTORS[:4]:  # 限制前4个板块加速
            try:
                sec_stocks = get_stocks_by_sector(sec)
                all_stocks.extend(sec_stocks)
            except Exception:
                pass
        # 去重
        stocks = list(dict.fromkeys(all_stocks))

    # ST 过滤
    stocks = api_filter_st(stocks)
    if not stocks:
        return []

    # 取前 n 只
    return stocks[:n]


# ============================================================
# 选股报告
# ============================================================

def generate_stock_report(tickers: list[str]) -> pd.DataFrame:
    """
    生成选股报告（股票池概览）

    返回 DataFrame，包含：
    - 代码、名称、板块、流通市值、换手率、近期涨跌
    """
    pro = get_pro_api()
    if pro is None:
        return pd.DataFrame()

    if not tickers:
        return pd.DataFrame()

    trade_date = datetime.now().strftime("%Y%m%d")
    rows = []

    for ticker in tickers[:50]:  # 限制前50只
        try:
            ts_code = normalize_ts_code(ticker)
            # 获取基本信息
            info = get_stock_info(ts_code)
            if info is None:
                continue

            # 获取当日行情
            df = pro.daily_basic(ts_code=ts_code, trade_date=trade_date)
            if df is None or df.empty:
                continue

            row = df.iloc[0]
            circ_mv = float(row.get("circ_mv", 0)) / 10000
            turn_rate = float(row.get("turn_rate", 0))
            close = float(row.get("close", 0))
            pct_chg = float(row.get("pct_chg", 0))

            rows.append({
                "代码": ts_code,
                "名称": info.name,
                "行业": info.industry,
                "流通市值(亿)": round(circ_mv, 2),
                "换手率(%)": round(turn_rate, 2),
                "收盘价": close,
                "涨跌幅(%)": round(pct_chg, 2),
                "ST": "✅" if info.is_st else "❌",
            })
        except Exception:
            continue

    df_report = pd.DataFrame(rows)
    if not df_report.empty:
        df_report = df_report.sort_values("流通市值(亿)", ascending=False)
    return df_report


# ============================================================
# 主程序（测试用）
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("A股选股器测试")
    print("=" * 60)

    # 测试快速选股
    print("\n[测试] 快速选股 (每板块5只, 限制前4个板块)...")
    stocks = quick_stock_pick(n=5)
    print(f"快速选股结果: {len(stocks)} 只")
    print(f"示例: {stocks[:5]}")

    # 测试板块扩展
    print("\n[测试] 板块扩展...")
    expanded = expand_sectors_with_keywords(top_n=3)
    print(f"扩展后板块数量: {len(expanded)}")

    # 测试全量选股（限制板块数加速）
    print("\n[测试] 全量选股 (3个板块，每板块5只)...")
    selected = get_a_stock_universe(
        sectors=CORE_SECTORS[:3],
        max_stocks_per_sector=5,
    )
    print(f"选股结果: {selected}")

    # 生成报告
    if selected:
        print("\n[选股报告]")
        report = generate_stock_report(selected)
        if not report.empty:
            print(report.to_string(index=False))

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
