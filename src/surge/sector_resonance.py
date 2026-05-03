#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sector_resonance.py — 板块共振检测

核心能力：
  1. 从 signal_log.json 加载信号数据，按板块聚合
  2. 计算各板块共振指数（信号密度、强度、情绪一致性）
  3. 自动识别 "主线板块"（共振最强的板块）
  4. 生成可排序/可视化的共振报告

工作流程：
  scan_resonance() -> 加载信号 -> 按板块分组 -> 共振评分 -> 排序 -> 报告
  
输出：
  resonance_report = {
    timestamp, trade_date,
    main_line: "半导体",           # 最高共振板块
    main_line_stocks: [...],       # 该板块 Top 个股
    sectors: {
      "半导体": { resonance_score, signal_density, strong_count, ... },
      ...
    }
  }
"""
from __future__ import annotations
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Optional

from src.utils.sector_map import SECTOR_INDUSTRY_MAP as SECTOR_MAP
from src.utils.sector_map import get_stock_info

logger = logging.getLogger(__name__)

# ── 路径 ────────────────────────────────────────────
SIGNAL_LOG_PATH = os.path.join(
    os.path.dirname(__file__), "signal_memory", "signal_log.json"
)


# ── 板块映射辅助 ────────────────────────────────────
def _find_sector(ts_code: str) -> str:
    """根据股票代码查找所属板块"""
    info = get_stock_info(ts_code)
    if info is None:
        return "other"
    industry = info.get("industry", "")
    if not industry:
        return "other"
    for sector_name, keywords in SECTOR_MAP.items():
        if any(kw in industry for kw in keywords):
            return sector_name
    return "other"


def _extract_sector_from_detail(detail: str) -> Optional[str]:
    """尝试从 detail 字段提取板块名（备选方案）"""
    m = re.search(r"所属板块:\s*(\S+)", detail)
    if m:
        name = m.group(1).strip()
        # 在 SECTOR_MAP 中找最匹配的
        for sector_name in SECTOR_MAP:
            if sector_name in name or name in sector_name:
                return sector_name
        return name  # 按字面返回
    return None


# ── 信号加载 ────────────────────────────────────────
def load_signals(path: str = None) -> dict:
    """
    加载 signal_log.json

    Returns: {timestamp, trade_date, signals: [{ts_code, score, grade, ...}]}
    """
    path = path or SIGNAL_LOG_PATH
    if not os.path.exists(path):
        logger.warning(f"signal_log not found: {path}")
        return {"timestamp": "", "trade_date": "", "signals": []}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    signals = data.get("signals", [])
    ts = data.get("last_updated", "")
    trade_date = signals[0]["trade_date"] if signals else datetime.now().strftime("%Y-%m-%d")

    return {"timestamp": ts, "trade_date": trade_date, "signals": signals}


# ── 共振计算 ────────────────────────────────────────
def compute_sector_resonance(
    signals: list, trade_date: str = None
) -> dict:
    """
    计算各板块共振指数

    输入: signals = [{ts_code, total_score, signal_grade, components, detail}, ...]
    输出: {
        sectors: { "半导体": { resonance_score, strong_count, weak_count,
                               signal_density, avg_score, sector_perf, ... },
                   ... },
        metadata: { trade_date, total_signals, sectors_with_signals }
    }
    """
    # 按板块分组
    sector_data = defaultdict(lambda: {
        "stocks": [],
        "scores": [],
        "strong_codes": [],
        "weak_codes": [],
        "signal_count": 0,
        "total_sector_score": 0,
    })

    for sig in signals:
        ts_code = sig["ts_code"]
        grade = sig.get("signal_grade", "NONE")

        # 只处理有效信号
        if grade not in ("STRONG", "WEAK"):
            continue

        sector = _find_sector(ts_code)

        sd = sector_data[sector]
        sd["stocks"].append(ts_code)
        sd["scores"].append(sig.get("total_score", 50))
        sd["signal_count"] += 1
        sd["total_sector_score"] += sig.get("total_score", 50)

        if grade == "STRONG":
            sd["strong_codes"].append(ts_code)
        else:
            sd["weak_codes"].append(ts_code)

    # 获取板块内总股票数（用于计算信号密度）
    sector_total_stocks = {}
    for sector_name in SECTOR_MAP:
        try:
            from src.utils.sector_map import get_stocks_by_sector
            stocks = get_stocks_by_sector(sector_name)
            sector_total_stocks[sector_name] = len(stocks)
        except Exception:
            sector_total_stocks[sector_name] = 200  # 估算

    # 计算共振评分
    sectors_result = {}
    for sector_name, sd in sector_data.items():
        n_signals = sd["signal_count"]
        if n_signals == 0:
            continue

        total = sector_total_stocks.get(sector_name, 200)
        density = n_signals / total if total > 0 else 0
        avg_score = sd["total_sector_score"] / n_signals
        strong_count = len(sd["strong_codes"])
        weak_count = len(sd["weak_codes"])

        # === 共振评分算法 ===
        # 1) 信号密度分 (0-40): 密度越高，板块共振越强
        if density >= 0.08:
            density_score = 40
        elif density >= 0.05:
            density_score = 30
        elif density >= 0.03:
            density_score = 20
        elif density >= 0.01:
            density_score = 10
        else:
            density_score = 5

        # 2) 强度分 (0-30): 强信号数量权重
        if strong_count >= 3:
            strength_score = 30
        elif strong_count >= 2:
            strength_score = 22
        elif strong_count >= 1:
            strength_score = 15
        else:
            strength_score = 5

        # 3) 平均质量分 (0-20): 板块内信号的平均分数
        quality = avg_score
        if quality >= 70:
            quality_score = 20
        elif quality >= 60:
            quality_score = 14
        elif quality >= 50:
            quality_score = 8
        else:
            quality_score = 4

        # 4) 板块组件分 (0-10): 来自 engine 的 sector_score 分量
        # 用第一个信号作为参考（所有信号共享板块环境得分）
        sector_component_scores = []
        for sig in signals:
            if sig["ts_code"] in sd["stocks"][:5]:
                comp = sig.get("components", {})
                sc = comp.get("sector_score", 0)
                if sc > 0:
                    sector_component_scores.append(sc)
        avg_sector_comp = (
            sum(sector_component_scores) / len(sector_component_scores)
            if sector_component_scores else 30
        )
        comp_score = min(10, avg_sector_comp / 10)

        resonance = density_score + strength_score + quality_score + comp_score

        # === 共振等级 ===
        if resonance >= 80:
            level = "main_line"
        elif resonance >= 60:
            level = "strong"
        elif resonance >= 40:
            level = "moderate"
        elif resonance >= 20:
            level = "weak"
        else:
            level = "noise"

        # Top 10 个股（按分数排序）
        top_stocks = sorted(
            [
                {"ts_code": code, "score": sd["scores"][i]}
                for i, code in enumerate(sd["stocks"])
            ],
            key=lambda x: x["score"],
            reverse=True,
        )[:10]

        sectors_result[sector_name] = {
            "resonance_score": round(resonance, 1),
            "resonance_level": level,
            "signal_density": round(density * 100, 2),
            "total_in_sector": total,
            "strong_count": strong_count,
            "weak_count": weak_count,
            "avg_signal_score": round(avg_score, 1),
            "top_stocks": top_stocks,
            "detail_scores": {
                "density": density_score,
                "strength": strength_score,
                "quality": quality_score,
                "component": comp_score,
            },
        }

    # 排序（按共振评分降序）
    sorted_sectors = sorted(
        sectors_result.items(),
        key=lambda x: x[1]["resonance_score"],
        reverse=True,
    )

    main_line = sorted_sectors[0][0] if sorted_sectors else None

    return {
        "sectors": dict(sorted_sectors),
        "main_line": main_line,
        "metadata": {
            "trade_date": trade_date or signals[0].get("trade_date", ""),
            "total_signals": len(signals),
            "sectors_with_signals": len(sorted_sectors),
            "signal_log_updated": signals[0].get("timestamp", "") if signals else "",
        },
    }


def get_resonance_report(path: str = None) -> dict:
    """
    一键生成板块共振报告

    从 signal_log 加载 -> 计算共振 -> Top Stocks 情绪融合补全 -> 返回
    """
    loaded = load_signals(path)
    signals = loaded["signals"]
    trade_date = loaded["trade_date"]

    if not signals:
        return {"error": "no signals", "trade_date": trade_date}

    resonance = compute_sector_resonance(signals, trade_date)

    # 主线板块 Top 3 个股补充 emotion 评分
    main_line = resonance.get("main_line")
    if main_line and main_line in resonance["sectors"]:
        top3 = resonance["sectors"][main_line]["top_stocks"][:3]
        try:
            from src.emotion import analyze_emotion
            emotion_results = {}
            for s in top3:
                try:
                    er = analyze_emotion(s["ts_code"], trade_date)
                    emotion_results[s["ts_code"]] = {
                        "fusion_score": er["fusion_score"],
                        "label": er["label"],
                        "surge_score": er["components"]["surge"].get("score", 0),
                        "news_score": er["components"]["news"].get("score", 50),
                        "market_score": er["components"]["market"].get("score", 50),
                        "sector_sentiment": er["components"]["sector"].get("score", 50),
                    }
                except Exception:
                    pass
            resonance["emotion_detail"] = emotion_results
        except ImportError:
            pass

    resonance["trade_date"] = trade_date

    return resonance


def format_report(report: dict) -> str:
    """将共振报告格式化为可读文本"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  板块共振检测报告")
    lines.append(f"  交易日: {report.get('trade_date', 'N/A')}")
    lines.append("=" * 60)

    main_line = report.get("main_line", "N/A")
    meta = report.get("metadata", {})
    lines.append(f"\n[主线板块] >>> {main_line} <<<")
    lines.append(f"  信号总数: {meta.get('total_signals', 0)}")
    lines.append(f"  有信号的板块: {meta.get('sectors_with_signals', 0)}")

    sectors = report.get("sectors", {})
    rank = 0
    for sector_name, info in sectors.items():
        rank += 1
        is_main = sector_name == main_line
        marker = " ★" if is_main else ""
        level_map = {
            "main_line": "主线",
            "strong": "强共振",
            "moderate": "中共振",
            "weak": "弱共振",
            "noise": "噪音",
        }
        level_str = level_map.get(info["resonance_level"], info["resonance_level"])
        lines.append(
            f"\n{rank}. {sector_name}{marker}"
        )
        lines.append(f"   共振评分: {info['resonance_score']} ({level_str})")
        lines.append(f"   信号密度: {info['signal_density']}% ({info['strong_count']}强/{info['weak_count']}弱)")
        lines.append(f"   平均分: {info['avg_signal_score']}")
        top = info.get("top_stocks", [])
        if top:
            top_str = ", ".join([f"{s['ts_code']}({s['score']})" for s in top[:5]])
            lines.append(f"   Top个股: {top_str}")

    # emotion detail
    if "emotion_detail" in report:
        lines.append("\n--- 主线板块情绪融合 ---")
        for code, ed in report["emotion_detail"].items():
            lines.append(
                f"  {code}: 融合={ed['fusion_score']} | "
                f"形态={ed['surge_score']} 新闻={ed['news_score']} | "
                f"板块={ed['sector_sentiment']}"
            )

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def generate_picks(
    resonance_report: dict,
    top_sectors: int = 3,
    per_sector: int = 3,
    min_fusion: float = 50.0,
) -> dict:
    """
    从共振报告生成可操作买入候选列表

    Args:
        resonance_report: get_resonance_report() 输出
        top_sectors: 取前几个板块
        per_sector: 每个板块取几只
        min_fusion: 情绪融合最低门槛

    Returns:
        {
            timestamp, trade_date,
            picks: [{
                ts_code, sector, surge_score, fusion_score,
                label, confidence, suggestion, position_pct, reason
            }, ...],
            market_context: { mood, warning }
        }
    """
    from datetime import datetime

    sectors = resonance_report.get("sectors", {})
    trade_date = resonance_report.get("trade_date", datetime.now().strftime("%Y-%m-%d"))

    # 排序取 Top N
    ranked = sorted(sectors.items(), key=lambda x: x[1]["resonance_score"], reverse=True)
    target_sectors = ranked[:top_sectors]

    picks = []
    seen_codes = set()

    for sector_name, info in target_sectors:
        top_stocks = info.get("top_stocks", [])[:per_sector]

        for s in top_stocks:
            code = s["ts_code"]
            if code in seen_codes:
                continue
            seen_codes.add(code)

            surge_score = s.get("score", 50)

            # 情感融合
            fusion_info = resonance_report.get("emotion_detail", {}).get(code, {})
            fusion_score = fusion_info.get("fusion_score", surge_score)
            label = fusion_info.get("label", "neutral")

            # 如果 emotion_detail 中没有（非主线板块），补充查询
            if not fusion_info and fusion_score == surge_score:
                try:
                    from src.emotion import analyze_emotion
                    er = analyze_emotion(code, trade_date)
                    fusion_score = er["fusion_score"]
                    label = er["label"]
                except Exception:
                    pass

            if fusion_score < min_fusion:
                continue

            # 建议仓位（基于融合评分和板块共振等级）
            if fusion_score >= 70 and info["resonance_level"] in ("main_line", "strong"):
                position_pct = 12
                suggestion = "BULL"
                reason = f"高融合{fusion_score} + {sector_name}主线共振"
            elif fusion_score >= 60:
                position_pct = 8
                suggestion = "BUY"
                reason = f"中融合{fusion_score} + {sector_name}共振{info['resonance_score']}"
            elif fusion_score >= 50:
                position_pct = 5
                suggestion = "WATCH"
                reason = f"低融合{fusion_score}，{sector_name}共振待确认"
            else:
                continue

            picks.append({
                "ts_code": code,
                "sector": sector_name,
                "surge_score": surge_score,
                "fusion_score": fusion_score,
                "label": label,
                "confidence": fusion_info.get("confidence", 60),
                "suggestion": suggestion,
                "position_pct": position_pct,
                "reason": reason,
            })

    # 按 fusion_score 排序
    picks.sort(key=lambda x: x["fusion_score"], reverse=True)

    # 市场上下文
    market_context = {
        "mood": "neutral",
        "warning": "",
    }
    try:
        from src.emotion.market_mood import get_market_mood
        mm = get_market_mood(trade_date)
        market_context["mood"] = mm.get("label", "neutral")
        limit = mm.get("components", {}).get("limit_ratio", {}).get("ratio", 1)
        if limit < 0.5:
            market_context["warning"] = "涨停比过低，短线风险加大"
    except Exception:
        pass

    return {
        "timestamp": datetime.now().isoformat(),
        "trade_date": trade_date,
        "total_picks": len(picks),
        "main_lines": [s for s, _ in target_sectors],
        "picks": picks,
        "market_context": market_context,
    }


def format_picks(picks_report: dict) -> str:
    """格式化候选列表为可读文本"""
    lines = []
    lines.append("=" * 65)
    lines.append(f"  [情绪共振] 买入候选列表")
    lines.append(f"  交易日: {picks_report.get('trade_date', 'N/A')}")
    lines.append(f"  市场情绪: {picks_report.get('market_context', {}).get('mood', 'N/A')}")
    warn = picks_report.get("market_context", {}).get("warning", "")
    if warn:
        lines.append(f"  ⚠ {warn}")
    lines.append("=" * 65)

    picks = picks_report.get("picks", [])
    if not picks:
        lines.append("\n  当前无符合条件的买入候选。")
        lines.append("=" * 65)
        return "\n".join(lines)

    lines.append(f"\n  共 {picks_report['total_picks']} 只候选 (Top {picks_report.get('main_lines', [])})")

    for i, p in enumerate(picks, 1):
        action = p["suggestion"]
        action_map = {"BULL": "强烈买入", "BUY": "买入", "WATCH": "关注"}
        action_cn = action_map.get(action, action)
        lines.append(
            f"\n{i}. {p['ts_code']} [{p['sector']}]"
        )
        lines.append(f"   建议: {action_cn}  |  仓位: {p['position_pct']}%")
        lines.append(f"   形态分: {p['surge_score']}  融合分: {p['fusion_score']}  置信: {p['confidence']}%")
        lines.append(f"   逻辑: {p['reason']}")

    lines.append("\n" + "=" * 65)
    return "\n".join(lines)

