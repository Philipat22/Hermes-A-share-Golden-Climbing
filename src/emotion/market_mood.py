#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
market_mood.py — 市场整体情绪指标

数据源：
  - 涨停/跌停比 (Tushare limit_list)
  - 北向资金净流入 (Tushare moneyflow_hsgt)
  - 指数趋势 (上证/深证/创业板)
  - 全市场成交量变化

输出：0-100 情绪分数（>60 偏多, <40 偏空）
      及各项细分指标

接入点：get_market_mood() -> dict
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np

from src.tools.a_stock_api import (
    get_pro_api, get_north_money, get_market_context,
    _safe_float,
)

logger = logging.getLogger(__name__)


def get_limit_ratio(date: str = None) -> dict:
    """
    获取涨停跌停比

    Returns:
        {limit_up_count, limit_down_count, ratio, score}
        ratio > 2 -> 偏多, ratio < 0.5 -> 偏空
        score 0-100
    """
    from src.tools.a_stock_api import get_limit_list
    ups = get_limit_list(date)
    downs = get_limit_list(date)

    # 需要过滤：get_limit_list 同时返回涨停和跌停
    # Tushare limit_list 有 pct_change 和 nature 字段
    up_count = 0
    down_count = 0

    if ups:
        for item in ups:
            # nature: 'U' = 涨停, 'D' = 跌停
            if hasattr(item, 'nature') and item.nature == 'D':
                down_count += 1
            elif hasattr(item, 'pct_change') and (item.pct_change or 0) < 0:
                down_count += 1
            elif hasattr(item, 'pct_change') and (item.pct_change or 0) >= 9.5:
                up_count += 1
            else:
                up_count += 1

    total = up_count + down_count
    if total == 0:
        return {"limit_up_count": 0, "limit_down_count": 0,
                "ratio": 1.0, "score": 50}

    ratio = up_count / down_count if down_count > 0 else float('inf')
    # ratio score: 1.0 -> 50, 2.0 -> 70, 3.0+ -> 85, 0.5 -> 30, 0.25 -> 15
    if ratio >= 5:
        score = 90
    elif ratio >= 3:
        score = 80
    elif ratio >= 2:
        score = 70
    elif ratio >= 1.5:
        score = 60
    elif ratio >= 1:
        score = 50
    elif ratio >= 0.5:
        score = 35
    elif ratio >= 0.25:
        score = 20
    else:
        score = 10

    return {"limit_up_count": up_count, "limit_down_count": down_count,
            "ratio": round(ratio, 2), "score": score}


def get_north_money_score(days: int = 5) -> dict:
    """
    北向资金情绪评分

    连续净流入 -> 偏多, 连续净流出 -> 偏空
    单日大幅流入/流出也计入
    """
    flow = get_north_money(days)
    if not flow:
        return {"north_net_total": 0, "positive_days": 0, "score": 50}

    totals = [f.get("north_net", 0) or 0 for f in flow]
    net_total = sum(totals)
    positive_days = sum(1 for t in totals if t > 0)

    if net_total > 5e9:  # 50亿+
        score = 80
    elif net_total > 1e9:
        score = 65
    elif net_total > 0:
        score = 55
    elif net_total > -1e9:
        score = 40
    elif net_total > -5e9:
        score = 30
    else:
        score = 15

    return {
        "north_net_total": round(net_total, 2),
        "positive_days": positive_days,
        "total_days": len(totals),
        "score": score,
    }


def get_index_momentum() -> dict:
    """
    指数动量评分

    基于上证指数5日和20日涨跌幅
    """
    pro = get_pro_api()
    try:
        df = pro.index_daily(ts_code="000001.SH",
                             start_date=(datetime.now() - timedelta(days=30)).strftime("%Y%m%d"),
                             end_date=datetime.now().strftime("%Y%m%d"))
        if df is None or df.empty:
            return {"pct_5d": 0, "pct_20d": 0, "score": 50}

        df = df.sort_values("trade_date")
        closes = df["close"].values

        pct_5d = (closes[-1] - closes[-min(6, len(closes))]) / closes[-min(6, len(closes))] * 100
        pct_20d = (closes[-1] - closes[-min(21, len(closes))]) / closes[-min(21, len(closes))] * 100

        # 综合评分
        if pct_5d > 3:
            score = 85
        elif pct_5d > 1:
            score = 70
        elif pct_5d > 0:
            score = 60
        elif pct_5d > -2:
            score = 40
        elif pct_5d > -4:
            score = 25
        else:
            score = 10

        return {
            "pct_5d": round(pct_5d, 2),
            "pct_20d": round(pct_20d, 2),
            "score": score,
        }
    except Exception as e:
        logger.warning(f"index_momentum failed: {e}")
        return {"pct_5d": 0, "pct_20d": 0, "score": 50}


def get_market_mood(trade_date: str = None) -> dict:
    """
    市场整体情绪评分（融合接口）

    Returns:
        {
            composite_score: 0-100,
            components: {
                limit_ratio: {score, ratio, ...},
                north_money: {score, ...},
                index_momentum: {score, ...},
            },
            label: "bullish"|"bearish"|"neutral"
        }
    """
    limit = get_limit_ratio(trade_date)
    north = get_north_money_score(days=5)
    index_m = get_index_momentum()

    # 加权融合
    w_limit = 0.35
    w_north = 0.25
    w_index = 0.40

    composite = (
        limit["score"] * w_limit
        + north["score"] * w_north
        + index_m["score"] * w_index
    )

    if composite >= 65:
        label = "bullish"
    elif composite <= 40:
        label = "bearish"
    else:
        label = "neutral"

    return {
        "composite_score": round(composite, 1),
        "label": label,
        "trade_date": trade_date or datetime.now().strftime("%Y-%m-%d"),
        "components": {
            "limit_ratio": limit,
            "north_money": north,
            "index_momentum": index_m,
        },
    }
