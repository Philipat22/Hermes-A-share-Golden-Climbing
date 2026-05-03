#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
news_fallback.py — 新闻备源降级链路

当 Tushare major_news 额度耗尽时，自动降级到其他免费源。

降级链路：
  1. Tushare major_news（主源，40次/天）
  2. 远程复写 stock_news_em（东方财富，无限制）
  3. 财联社快讯过滤（无个股分离时返回中性）
"""

from __future__ import annotations
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)

# ── 关键词情绪词典（同 news_sentiment.py）─
_BULLISH_KEYWORDS = [
    "增长", "突破", "利好", "创新", "超预期", "放量", "涨停", "反转",
    "买入", "推荐", "看好", "低估值", "高增长", "扭亏", "盈喜", "拉升",
    "资金流入", "主力", "抢筹", "溢价", "新高", "景气", "扩张", "中标",
    "获批复", "产能", "需求旺盛",
]
_BEARISH_KEYWORDS = [
    "下跌", "利空", "亏损", "预警", "减持", "套现", "跌停", "出货",
    "下调", "看空", "卖出", "高估值", "恶化", "暴雷", "st", "退市",
    "资金流出", "解禁", "质押", "违约", "立案", "罚款", "整改",
    "暂停上市", "风险提示",
]


def _simple_sentiment(title: str, content: str = "") -> float:
    """简单的关键词情绪评分"""
    text = (title + " " + content).lower()
    bullish = sum(1 for kw in _BULLISH_KEYWORDS if kw in text)
    bearish = sum(1 for kw in _BEARISH_KEYWORDS if kw in text)
    total = bullish + bearish
    if total == 0:
        return 50.0
    return round(50 + (bullish - bearish) / total * 50, 1)


# ── 备源1：东方财富新闻（修复版 stock_news_em）─
_EM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://so.eastmoney.com/news/s",
    "Accept": "application/json, text/plain, */*",
}

_EM_COOKIES = {
    "qgqp_b_id": "652bf4c98a74e210088f372a17d4e27b",
}


def fetch_em_news(symbol: str, page_size: int = 10) -> list[dict]:
    """
    从东方财富搜索 API 获取个股新闻

    Returns: [{title, date, content_snippet, url}, ...]
    """
    params = {
        'uid': '',
        'keyword': symbol,
        'type': ['cmsArticleWebOld'],
        'client': 'web',
        'clientType': 'web',
        'clientVersion': 'curr',
        'param': {
            'cmsArticleWebOld': {
                'searchScope': 'default',
                'sort': 'default',
                'pageIndex': 1,
                'pageSize': page_size,
            }
        }
    }

    url = "https://search-api-web.eastmoney.com/search/jsonp"
    resp = requests.get(
        url,
        params={"cb": "jQuery", "param": json.dumps(params, ensure_ascii=False)},
        headers=_EM_HEADERS,
        cookies=_EM_COOKIES,
        timeout=10,
    )

    # Extract JSON from JSONP
    m = re.search(r'\{.*\}', resp.text, re.DOTALL)
    if not m:
        logger.warning("EM news: no JSON found in response")
        return []

    data = json.loads(m.group())
    articles = data.get("result", {}).get("cmsArticleWebOld", [])
    if not articles:
        logger.info(f"EM news: 0 articles for {symbol}")
        return []

    results = []
    for a in articles:
        title = a.get("title", "")
        title = re.sub(r"<[^>]+>", "", title)  # strip HTML tags
        # Remove \u3000 (ideographic space) - the AKShare bug fix
        title = title.replace("\u3000", "")
        date = a.get("date", "")[:10]
        content = a.get("content", a.get("summary", ""))[:200]
        code = a.get("art_code", "")
        art_url = f"http://finance.eastmoney.com/a/{code}.html" if code else ""

        results.append({
            "title": title,
            "date": date,
            "content": content,
            "url": art_url,
        })

    return results


def em_news_sentiment(symbol: str, days: int = 7) -> Optional[dict]:
    """
    东方财富新闻情绪分析

    Returns: {score_avg, score_latest, news_count, sentiment_label, source}
    """
    articles = fetch_em_news(symbol, page_size=10)
    if not articles:
        return None

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    filtered = [a for a in articles if a["date"] >= cutoff]
    if not filtered:
        filtered = articles[:5]  # 日期不匹配时取最新的

    scores = []
    for a in filtered:
        s = _simple_sentiment(a["title"], a.get("content", ""))
        scores.append(s)

    score_avg = round(sum(scores) / len(scores), 1)
    score_latest = scores[-1] if scores else 50.0

    if score_avg >= 60:
        label = "positive"
    elif score_avg <= 40:
        label = "negative"
    else:
        label = "neutral"

    return {
        "score_avg": score_avg,
        "score_latest": score_latest,
        "news_count": len(filtered),
        "sentiment_label": label,
        "source": "eastmoney",
    }


# ── 财联社快讯备源（全局新闻，含个股关键词过滤）─
_CLS_CACHE = {"data": None, "ts": 0}


def _fetch_cls_news() -> list[dict]:
    """取财联社快讯"""
    now = time.time()
    if _CLS_CACHE["data"] and now - _CLS_CACHE["ts"] < 60:
        return _CLS_CACHE["data"]

    try:
        import akshare as ak
        df = ak.stock_info_global_cls()
        if df is not None and hasattr(df, "columns") and len(df) > 0:
            items = []
            for _, row in df.iterrows():
                items.append({
                    "title": str(row.iloc[0]) if len(row) > 0 else "",
                    "content": str(row.iloc[1]) if len(row) > 1 else "",
                    "date": str(row.iloc[3]) if len(row) > 3 else "",
                })
            _CLS_CACHE["data"] = items
            _CLS_CACHE["ts"] = now
            return items
    except Exception as e:
        logger.warning(f"CLS fetch failed: {e}")
    return []


def cls_news_sentiment(symbol: str, company_name: str = "") -> Optional[dict]:
    """
    财联社快讯过滤（按股票代码和公司名称匹配）

    Returns: {score_avg, score_latest, news_count, sentiment_label, source}
    """
    articles = _fetch_cls_news()
    if not articles:
        return None

    code_last4 = symbol.split(".")[0][-4:] if "." in symbol else symbol[-6:]
    # 过滤相关新闻
    matched = []
    for a in articles:
        text = a["title"] + " " + a["content"]
        if code_last4 in text or (company_name and company_name in text):
            matched.append(a)

    if not matched:
        return None

    scores = [_simple_sentiment(a["title"], a["content"]) for a in matched]
    score_avg = round(sum(scores) / len(scores), 1)
    score_latest = scores[-1] if scores else 50.0

    label = "positive" if score_avg >= 60 else ("negative" if score_avg <= 40 else "neutral")

    return {
        "score_avg": score_avg,
        "score_latest": score_latest,
        "news_count": len(matched),
        "sentiment_label": label,
        "source": "cls",
    }


# ── 统一入口 ────────────────────────────────────────
def get_fallback_news_score(
    symbol: str,
    company_name: str = "",
    days: int = 7,
) -> dict:
    """
    多级新闻备源降级调用

    顺序：
      1. 财联社（宏观快讯，覆盖有限）
      2. 东方财富（需 curl_cffi，纯requests可能受限）
      3. 中性兜底（major_news 日限40次用完后的默认）

    注：major_news配额耗尽后所有未查股票返回中性50。
    """
    # 1. 财联社快讯过滤
    try:
        result = cls_news_sentiment(symbol, company_name)
        if result and result["news_count"] > 0:
            return result
    except Exception:
        pass

    # 2. 东方财富（try with curl_cffi if available）
    try:
        result = em_news_sentiment(symbol, days)
        if result and result["news_count"] > 0:
            return result
    except Exception:
        pass

    # 3. 兜底中性
    return {
        "score_avg": 50.0,
        "score_latest": 50.0,
        "news_count": 0,
        "sentiment_label": "neutral",
        "source": "fallback_none",
    }


if __name__ == "__main__":
    # quick test
    r1 = get_fallback_news_score("603501")
    print(f"603501 (EM): {r1}")
    r2 = get_fallback_news_score("000001")
    print(f"000001 (EM): {r2}")
    r3 = get_fallback_news_score("999999")
    print(f"999999 (no news): {r3}")
