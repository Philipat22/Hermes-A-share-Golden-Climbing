#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fusion.py — 情绪融合引擎

将多源信号加权融合为综合情绪评分：
  - 形态信号 (surge engine: N字/VCP/V反转/W底/平台突破)
  - 新闻情绪 (news_sentiment)
  - 市场整体情绪 (market_mood)
  - 板块情绪 (sector momentum)

输出：0-100 融合评分 + 置信度 + 分项拆解

使用方式：
    from src.emotion import analyze_emotion
    result = analyze_emotion("603501.SH")
"""
from __future__ import annotations
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np

from src.tools.a_stock_api import get_prices, get_market_context
from src.utils.sector_map import get_stock_info as _get_stock_info
from src.utils.sector_map import SECTOR_INDUSTRY_MAP as _SECTOR_MAP
from src.surge.engine import analyze_stock, load_params

# Lazy imports (avoid circular deps at module load)
_news_module = None
_market_module = None

# ── 全局缓存（每交易日记一次，避免 N 次调用）─
_MARKET_MOOD_CACHE = {}
_SECTOR_MOOD_CACHE = {}


def _get_cached_market_mood(trade_date: str = None) -> dict:
    """市场情绪缓存（全局一致，拍一次）"""
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%d")
    key = f"mood_{trade_date}"
    if key not in _MARKET_MOOD_CACHE:
        mm = _get_market_module()
        _MARKET_MOOD_CACHE[key] = mm.get_market_mood(trade_date)
    return _MARKET_MOOD_CACHE[key]


def _get_cached_sector_mood(ts_code: str) -> dict:
    """板块情绪缓存（同板块不重复查）"""
    sector_name = _find_sector(ts_code)
    key = f"sector_{sector_name}"
    if key not in _SECTOR_MOOD_CACHE:
        _SECTOR_MOOD_CACHE[key] = get_sector_mood(ts_code)
    return _SECTOR_MOOD_CACHE[key]


def _get_news_module():
    global _news_module
    if _news_module is None:
        from src.emotion import news_sentiment as m
        _news_module = m
    return _news_module

def _get_market_module():
    global _market_module
    if _market_module is None:
        from src.emotion import market_mood as m
        _market_module = m
    return _market_module


# ── 默认权重（可从 feedback.py 调整）─────────────────
EMOTION_WEIGHTS = {
    "w_surge": 0.35,       # 形态信号（权重最大，但形态本身已有 vol+accel）
    "w_news": 0.10,        # 新闻情绪（数据较少，权重偏低）
    "w_market": 0.20,      # 市场整体情绪（大盘涨跌影响个股）
    "w_sector": 0.25,      # 板块情绪（板块共振效应）
    "w_diversion": 0.10,   # 分歧惩罚（信号方向不一致时扣分）
}

WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "emotion_weights.json")


def _load_weights() -> dict:
    """加载权重（支持 feedback 动态调整）"""
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return dict(EMOTION_WEIGHTS)


def _save_weights(weights: dict) -> None:
    """保存权重"""
    with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)


def _find_sector(ts_code: str) -> str:
    """根据股票代码查找所属板块"""
    info = _get_stock_info(ts_code)
    if info is None:
        return "unknown"
    industry = info.get("industry", "")
    for sector_name, keywords in _SECTOR_MAP.items():
        if any(kw in industry for kw in keywords):
            return sector_name
    return "unknown"


# ── 申万行业指数映射（sw_daily 接口）─────────────
_SECTOR_INDEX_MAP = {
    "半导体": "801081.SI",  # 半导体(申万)
    "银行": "801780.SI",    # 银行(申万)
    "房地产": "801180.SI",  # 房地产(申万)
    "医药": "801150.SI",    # 医药生物(申万)
    "食品饮料": "801120.SI", # 食品饮料(申万)
    "新能源": "801730.SI",  # 电气设备(申万) —— 含光伏/风电/电池
    "军工": "801740.SI",    # 国防军工(申万)
    "计算机": "801750.SI",  # 计算机(申万)
    "通信": "801770.SI",    # 通信(申万)
    "化工": "801030.SI",    # 化工(申万)
    "有色金属": "801050.SI", # 有色金属(申万)
    "机械设备": "801890.SI", # 机械设备(申万)
    "建筑建材": "801710.SI", # 建筑材料(申万)
    "汽车": "801880.SI",    # 汽车(申万)
    "电力": "801161.SI",    # 电力(申万)
    "传媒": "801760.SI",    # 传媒(申万)
    "钢铁": "801040.SI",    # 钢铁(申万)
    "消费电子": "801085.SI", # 消费电子(申万)
}


def get_sector_mood(ts_code: str) -> dict:
    """
    板块情绪评分

    基于：
    1. 申万行业指数近期涨跌幅（sw_daily 接口）
    2. 板块内个股的形态信号密度（预留）

    Returns: {score, sector_name, sector_perf_20d, signal_density}
    """
    sector_name = _find_sector(ts_code)
    if not sector_name or sector_name == "unknown":
        return {"score": 50, "sector_name": "unknown",
                "sector_perf_20d": 0, "signal_density": 0}

    perf_20d = 0
    index_code = _SECTOR_INDEX_MAP.get(sector_name)

    if index_code:
        try:
            from src.tools.a_stock_api import get_pro_api
            pro = get_pro_api()
            # 申万行业指数用 sw_daily 接口
            df = pro.sw_daily(
                ts_code=index_code,
                start_date=(datetime.now() - timedelta(days=60)).strftime("%Y%m%d"),
                end_date=datetime.now().strftime("%Y%m%d"),
            )
            if df is not None and not df.empty and "close" in df.columns:
                df = df.sort_values("trade_date")
                closes = df["close"].values
                if len(closes) >= 2:
                    # 20 交易日（约一个自然月）涨跌幅
                    n = min(21, len(closes))
                    perf_20d = (closes[-1] - closes[-n]) / closes[-n] * 100
        except Exception:
            pass

    # 板块涨跌幅 -> 分数
    if perf_20d > 10:
        score = 85
    elif perf_20d > 5:
        score = 70
    elif perf_20d > 2:
        score = 60
    elif perf_20d > 0:
        score = 55
    elif perf_20d > -5:
        score = 40
    elif perf_20d > -10:
        score = 25
    else:
        score = 10

    return {
        "score": score,
        "sector_name": sector_name,
        "sector_perf_20d": round(perf_20d, 2),
        "signal_density": 0,
    }


def compute_diversion_penalty(components: dict) -> float:
    """
    计算信号分歧惩罚

    当各源信号方向不一致时（如形态看多但新闻看空），惩罚融合分数
    Returns: 0.0 (一致) ~ 0.3 (严重分歧) 扣分比例
    """
    scores = []
    for k, v in components.items():
        s = v.get("score", 50)
        scores.append(s)

    if not scores:
        return 0.0

    # 标准差衡量分歧度
    std = float(np.std(scores))
    # std=0 -> 0, std=25 -> 0.1, std=40+ -> 0.3
    penalty = min(0.3, std / 150)
    return round(penalty, 3)


def analyze_emotion(
    ts_code: str,
    trade_date: str = None,
    df: Optional[pd.DataFrame] = None,
    surge_params: Optional[dict] = None,
) -> dict:
    """
    多源情绪融合分析

    Args:
        ts_code: 股票代码
        trade_date: 交易日（默认今天）
        df: 价格数据（可选，不传则自动获取）
        surge_params: surge 引擎参数（可选）

    Returns:
        {
            ts_code, fusion_score, confidence, label,
            components: {surge, news, market, sector},
            weights_used: {...}
        }
    """
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%d")

    weights = _load_weights()
    components = {}

    # 1. 形态信号（surge engine）
    try:
        if df is None:
            prices = get_prices(ts_code, "2025-06-01", trade_date)
            if prices and len(prices) > 30:
                df = pd.DataFrame([{
                    "close": p.close, "high": p.high, "low": p.low,
                    "open": p.open, "vol": p.volume,
                } for p in prices])

        if df is not None and len(df) > 30:
            params = surge_params or load_params()
            surge_result = analyze_stock(ts_code, df, {}, params)
            surge_score = surge_result.get("final_score", 50)
            surge_detected = surge_result.get("detected", False)
            components["surge"] = {
                "score": surge_score,
                "detected": surge_detected,
                "pattern_type": surge_result.get("pattern_type", "none"),
                "detail": surge_result.get("detail", ""),
            }
        else:
            components["surge"] = {"score": 50, "detected": False,
                                   "pattern_type": "none", "detail": "no_data"}
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"surge analysis failed for {ts_code}: {e}")
        components["surge"] = {"score": 50, "detected": False,
                               "pattern_type": "error", "detail": str(e)[:60]}

    # 2. 新闻情绪
    try:
        news_module = _get_news_module()
        news_result = news_module.get_news_score(ts_code, days=7)
        components["news"] = {
            "score": news_result["score_avg"],
            "news_count": news_result["news_count"],
            "label": news_result["sentiment_label"],
            "latest_score": news_result["score_latest"],
        }
    except Exception as e:
        components["news"] = {"score": 50, "news_count": 0,
                              "label": "neutral", "error": str(e)[:60]}

    # 3. 市场整体情绪（全局缓存，同交易日只查一次）
    try:
        mood = _get_cached_market_mood(trade_date)
        components["market"] = {
            "score": mood["composite_score"],
            "label": mood["label"],
            "limit_ratio": mood["components"]["limit_ratio"].get("ratio", 1),
            "north_net": mood["components"]["north_money"].get("north_net_total", 0),
            "index_pct_5d": mood["components"]["index_momentum"].get("pct_5d", 0),
        }
    except Exception as e:
        components["market"] = {"score": 50, "label": "neutral",
                                "error": str(e)[:60]}

    # 4. 板块情绪（板块级缓存，同板块共享）
    try:
        sector_mood = _get_cached_sector_mood(ts_code)
        components["sector"] = {
            "score": sector_mood["score"],
            "sector_name": sector_mood["sector_name"],
            "perf_20d": sector_mood["sector_perf_20d"],
        }
    except Exception as e:
        components["sector"] = {"score": 50, "sector_name": "unknown",
                                "error": str(e)[:60]}

    # 5. 分歧惩罚
    diversion = compute_diversion_penalty(components)
    components["_diversion"] = {"penalty": diversion}

    # 6. 加权融合
    fusion = (
        components["surge"]["score"] * weights["w_surge"]
        + components["news"]["score"] * weights["w_news"]
        + components["market"]["score"] * weights["w_market"]
        + components["sector"]["score"] * weights["w_sector"]
    )

    # 应用分歧惩罚（降低分数）
    fusion = fusion * (1 - diversion * weights["w_diversion"] * 2)

    # 7. 置信度（基于数据源完整性）
    data_sources = 0
    for key in ["surge", "news", "market", "sector"]:
        c = components.get(key, {})
        if "error" not in c and c.get("score", 50) != 50:
            # 粗略判断：如果默认值就是50，且没有error，可能是数据缺失
            pass
        data_sources += 1

    # 新闻情绪若无数据，confidence 下降
    news_conf = 1.0 if components["news"].get("news_count", 0) > 0 else 0.6

    # 8. 标签
    if fusion >= 70:
        label = "strong_bullish"
    elif fusion >= 60:
        label = "bullish"
    elif fusion >= 45:
        label = "neutral"
    elif fusion >= 35:
        label = "bearish"
    else:
        label = "strong_bearish"

    # 移除内部字段
    display_components = {k: v for k, v in components.items()
                          if not k.startswith("_")}

    return {
        "ts_code": ts_code,
        "fusion_score": round(fusion, 1),
        "confidence": round(news_conf * 100, 1),
        "label": label,
        "trade_date": trade_date,
        "components": display_components,
        "weights_used": weights,
    }


def batch_emotion_score(
    stock_codes: list[str],
    trade_date: str = None,
) -> list[dict]:
    """
    批量情绪融合评分

    Args:
        stock_codes: 股票代码列表
        trade_date: 交易日

    Returns:
        [{ts_code, fusion_score, confidence, label, ...}, ...]
        按 fusion_score 降序排列
    """
    results = []
    total = len(stock_codes)
    for i, code in enumerate(stock_codes):
        if (i + 1) % 50 == 0:
            print(f"  emotion: {i+1}/{total}")
        try:
            r = analyze_emotion(code, trade_date)
            results.append(r)
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"emotion skipped {code}: {e}")
            results.append({
                "ts_code": code,
                "fusion_score": 50,
                "confidence": 0,
                "label": "error",
                "error": str(e)[:100],
            })

    results.sort(key=lambda x: x.get("fusion_score", 0), reverse=True)
    return results
