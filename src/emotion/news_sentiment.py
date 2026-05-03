#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
news_sentiment.py — 个股新闻情绪评分

数据源：Tushare major_news（近7天内公司新闻）
评分方式：关键词匹配 + 加权聚合
输出：0-100 情绪分数（>60 偏多, <40 偏空）

接入点：get_news_score(ts_code) -> dict
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Optional
from src.tools.a_stock_api import get_company_news, get_pro_api

logger = logging.getLogger(__name__)

# ── 情绪词典（三级权重）─────────────────────────────
# 权重：强(3) 中(2) 弱(1) — 一个新闻命中多个关键词累加

POSITIVE_DICT = {
    # 强利好
    "业绩预增": 3, "净利润增长": 3, "营收增长": 3, "大幅增长": 3,
    "创历史新高": 3, "超预期": 3, "扭亏为盈": 3,
    "签订重大合同": 3, "中标": 3, "获得大订单": 3,
    "研发突破": 3, "技术突破": 3, "新产品": 3,
    "回购": 3, "增持": 3,
    # 中利好
    "签约": 2, "战略合作": 2, "合资": 2,
    "行业复苏": 2, "景气度": 2, "供不应求": 2,
    "产能扩张": 2, "产能释放": 2,
    "补贴": 2, "政策扶持": 2,
    # 弱利好
    "稳定增长": 1, "持续向好": 1, "改善": 1,
    "布局": 1, "拓展": 1, "加速": 1,
}

NEGATIVE_DICT = {
    # 强利空
    "亏损": 3, "净利润下降": 3, "营收下降": 3, "大幅下滑": 3,
    "跌停": 3, "ST": 3, "退市": 3, "被立案": 3,
    "违约": 3, "债务": 3, "资金链": 3,
    "股东减持": 3, "减持": 3,
    # 中利空
    "产能过剩": 2, "库存高企": 2, "需求下滑": 2,
    "罚款": 2, "处罚": 2, "违规": 2, "调查": 2,
    "裁员": 2, "降薪": 2,
    # 弱利空
    "下调": 1, "低于预期": 1, "放缓": 1,
    "压力": 1, "挑战": 1, "不确定": 1,
    "警示": 1, "预警": 1,
}

# ── 噪声过滤词（命中则 skip 该新闻）──
SENSITIVE_FILTER = ["警示", "风险提示", "减持公告", "筹划重大资产重组",
                    "内幕", "泄密", "内部", "举报"]

# 非情感纯事实关键词（不计分）
NEUTRAL_KW = ["除权除息", "分红派息", "股东大会", "停牌", "复牌",
              "分红", "交易", "上证", "深证", "公告"]


def score_news_item(content: str, title: str = "") -> dict:
    """对单条新闻逐级打分，返回 {score, positive_count, negative_count, neutral}"""
    text = (title + " " + content).lower()

    # 中性/噪声过滤
    if any(k in text for k in SENSITIVE_FILTER):
        return {"score": 0, "positive_count": 0, "negative_count": 0,
                "neutral": True, "reason": "filtered"}
    if any(k in text for k in NEUTRAL_KW):
        return {"score": 0, "positive_count": 0, "negative_count": 0,
                "neutral": True, "reason": "neutral"}

    pos_score = 0
    neg_score = 0
    for kw, w in POSITIVE_DICT.items():
        if kw in text:
            pos_score += w
    for kw, w in NEGATIVE_DICT.items():
        if kw in text:
            neg_score += w

    net_score = pos_score - neg_score

    # 归一化到 -100 ~ 100，再映射到 0~100
    raw = max(-100, min(100, net_score * 10))
    normalized = (raw + 100) / 2  # 0-100

    result = {
        "score": round(normalized, 1),
        "raw_net": net_score,
        "positive_count": pos_score,
        "negative_count": neg_score,
        "neutral": False,
        "reason": "positive" if pos_score > neg_score else (
            "negative" if neg_score > pos_score else "mixed"
        ),
    }
    return result


def get_news_score(ts_code: str, days: int = 7) -> dict:
    """
    获取个股新闻情绪评分

    Args:
        ts_code: 股票代码（如 "603501.SH"）
        days: 回溯天数（默认7天）

    Returns:
        {ts_code, score_avg, score_latest, news_count, positive_pct,
         negative_pct, sentiment_label, items: [{date, title, score, reason}]}
    """
    raw_news = get_company_news(ts_code, limit=80)  # 多取一些
    if not raw_news:
        return {
            "ts_code": ts_code,
            "score_avg": 50.0,
            "score_latest": 50.0,
            "news_count": 0,
            "positive_pct": 0,
            "negative_pct": 0,
            "sentiment_label": "neutral",
            "items": [],
        }

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    scored = []
    for item in raw_news:
        if item.date < cutoff:
            continue
        s = score_news_item(item.content, item.title)
        scored.append({
            "date": item.date,
            "title": item.title[:80],
            "score": s["score"],
            "reason": s["reason"],
            "positive_count": s["positive_count"],
            "negative_count": s["negative_count"],
        })

    if not scored:
        return {
            "ts_code": ts_code,
            "score_avg": 50.0,
            "score_latest": 50.0,
            "news_count": 0,
            "positive_pct": 0,
            "negative_pct": 0,
            "sentiment_label": "neutral",
            "items": [],
        }

    scores = [s["score"] for s in scored]
    avg = sum(scores) / len(scores)
    latest = scored[-1]["score"] if scored else 50.0
    pos = sum(1 for s in scored if s["reason"] == "positive")
    neg = sum(1 for s in scored if s["reason"] == "negative")

    # 标签
    if avg >= 60:
        label = "bullish"
    elif avg <= 40:
        label = "bearish"
    else:
        label = "neutral"

    return {
        "ts_code": ts_code,
        "score_avg": round(avg, 1),
        "score_latest": round(latest, 1),
        "news_count": len(scored),
        "positive_pct": round(pos / len(scored) * 100, 1),
        "negative_pct": round(neg / len(scored) * 100, 1),
        "sentiment_label": label,
        "items": scored[-5:],  # 只保留最近5条
    }
