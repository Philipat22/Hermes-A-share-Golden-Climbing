#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
sys.stdout.reconfigure(encoding="utf-8")

"""
板块扫描仪 — 16板块×大师分析管线

两阶段策略：
  Phase 1: 快速计算Agent扫全量候选股（基本面/技术/情绪/成长/估值/新闻）— 秒级
  Phase 2: LLM大师深度分析各板块Top Pick（巴菲特/芒格/Damodaran等）— 需API调用

用法:
    python sector_scan.py                          # 全板块扫描（默认）
    python sector_scan.py --phase 1                 # 只跑Phase 1 (快速计算)
    python sector_scan.py --phase 2                 # 只跑Phase 2 (LLM大师)
    python sector_scan.py --sectors 白酒 半导体      # 指定板块
    python sector_scan.py --output report.md         # 输出到文件
"""

import sys, os
from datetime import datetime
from collections import defaultdict
import argparse
import json

from dotenv import load_dotenv
from src.tools.a_stock_api import get_stock_info, SECTOR_POOL, get_stocks_by_sector, filter_st_stocks
from src.main_astock import run_astock_analysis, print_signal_result
from src.utils.analysts import ANALYST_CONFIG, get_analyst_nodes

load_dotenv()

# ──────────────────────────────────────────────
# Agent 分组
# ──────────────────────────────────────────────

# 快速计算Agent（纯计算，无LLM调用）
FAST_AGENTS = [
    "fundamentals_analyst",
    "technicals_analyst",
    "sentiment_analyst",
    "growth_analyst",
    "valuation_analyst",
    "news_sentiment_analyst",
]

# LLM 大师（调用DeepSeek）
LLM_AGENTS = [
    "warren_buffett",
    "charlie_munger",
    "ben_graham",
    "aswath_damodaran",
    "cathie_wood",
    "peter_lynch",
    "phil_fisher",
    "bill_ackman",
    "michael_burry",
    "stanley_druckenmiller",
    "nassim_taleb",
    "rakesh_jhunjhunwala",
    "mohnish_pabrai",
]

ALL_AGENTS = FAST_AGENTS + LLM_AGENTS

# ──────────────────────────────────────────────
# 评分计算
# ──────────────────────────────────────────────

def compute_composite_score(analyst_signals: dict, ticker: str) -> float:
    """
    对一个股票的所有Agent信号计算综合得分。
    返回 [-100, 100] 区间得分：正=看多，负=看空。
    """
    total_bullish = 0.0
    total_bearish = 0.0

    for agent_key, signals in analyst_signals.items():
        if ticker not in signals:
            continue
        sig = signals[ticker]
        s = sig.get("signal", "neutral")
        c = sig.get("confidence", 0) or 0
        if s == "bullish":
            total_bullish += c
        elif s == "bearish":
            total_bearish += c
        # neutral: contributes 0

    total = total_bullish + total_bearish
    if total == 0:
        return 0.0
    return round((total_bullish - total_bearish) / total * 100, 1)


def compute_score_breakdown(analyst_signals: dict, ticker: str) -> dict:
    """返回每个Agent的单独得分，用于展示"""
    breakdown = {}
    for agent_key, signals in analyst_signals.items():
        if ticker not in signals:
            continue
        sig = signals[ticker]
        cfg = ANALYST_CONFIG.get(agent_key, {})
        breakdown[agent_key] = {
            "display_name": cfg.get("display_name", agent_key),
            "signal": sig.get("signal", "neutral"),
            "confidence": sig.get("confidence", 0),
            "reasoning": sig.get("reasoning", {}),
        }
    return breakdown


# ──────────────────────────────────────────────
# 板块扫描 Phase 1: 快速Agent
# ──────────────────────────────────────────────

def phase1_scan(sectors: list[str], stocks_per_sector: int = 5) -> dict:
    """
    Phase 1: 对所有候选股运行6个快速计算Agent。
    返回 { sector: [(ticker, score, breakdown), ...] }
    """
    print("\n" + "=" * 60)
    print("📡 PHASE 1: 快速Agent扫描")
    print("=" * 60)
    print(f"  板块: {len(sectors)} 个 | 每板块: {stocks_per_sector} 只")

    # 收集所有候选股
    sector_candidates = {}
    all_tickers = []

    for sector in sectors:
        raw = get_stocks_by_sector(sector)
        filtered = filter_st_stocks(raw, min_market_cap=10.0)
        picks = filtered[:stocks_per_sector]
        sector_candidates[sector] = picks
        all_tickers.extend(picks)
        for t in picks:
            info = get_stock_info(t)
            name = info.name if info else "??"
            print(f"    {sector:6s} → {t} {name}")

    print(f"\n  共 {len(all_tickers)} 只候选股，运行 {len(FAST_AGENTS)} 个快速Agent...")

    result = run_astock_analysis(
        tickers=all_tickers,
        selected_analysts=FAST_AGENTS,
        show_reasoning=False,
    )

    # 计算得分并归入各板块
    sector_ranked = {}
    for sector, tickers in sector_candidates.items():
        scored = []
        for t in tickers:
            score = compute_composite_score(result["analyst_signals"], t)
            breakdown = compute_score_breakdown(result["analyst_signals"], t)
            scored.append((t, score, breakdown))
        # 按得分降序
        scored.sort(key=lambda x: x[1], reverse=True)
        sector_ranked[sector] = scored

    return sector_ranked, result["analyst_signals"]


# ──────────────────────────────────────────────
# 板块扫描 Phase 2: LLM大师
# ──────────────────────────────────────────────

def phase2_deep_dive(sector_ranked: dict, top_n: int = 1) -> dict:
    """
    Phase 2: 对每板块Top N股票运行13位LLM大师。
    返回 { sector: [(ticker, score, full_breakdown), ...] }
    """
    print("\n" + "=" * 60)
    print("🧠 PHASE 2: LLM大师深度分析")
    print("=" * 60)

    # 提取top picks
    top_tickers = []
    top_info = {}  # ticker -> (sector, phase1_score)
    for sector, scored in sector_ranked.items():
        for ticker, score, _ in scored[:top_n]:
            top_tickers.append(ticker)
            top_info[ticker] = (sector, score)

    print(f"  每板块Top {top_n}: {len(top_tickers)} 只 → {len(LLM_AGENTS)} 位LLM大师")

    for t in top_tickers:
        info = get_stock_info(t)
        name = info.name if info else t
        sec, sc = top_info[t]
        print(f"    {sec:6s} → {t} {name} (Phase1得分: {sc:+.1f})")

    print(f"\n  LLM大师分析中（需调用DeepSeek API，预计2-4分钟）...")

    result = run_astock_analysis(
        tickers=top_tickers,
        selected_analysts=LLM_AGENTS,
        show_reasoning=False,
    )

    # 合并phase1 + phase2结果
    full = {}
    for sector, scored in sector_ranked.items():
        full_scored = []
        for ticker, score, bd1 in scored:
            # 合并phase2信号（如果有）
            bd2 = compute_score_breakdown(result["analyst_signals"], ticker)
            full_bd = {**bd1, **bd2}
            full_score = compute_composite_score(result["analyst_signals"], ticker)
            # 综合得分：phase1 占40% + phase2 占60%
            if full_score != 0 and score != 0:
                composite = round(score * 0.4 + full_score * 0.6, 1)
            else:
                composite = score if score != 0 else full_score
            full_scored.append((ticker, composite, full_bd))
        full_scored.sort(key=lambda x: x[1], reverse=True)
        full[sector] = full_scored

    return full, result["analyst_signals"]


# ──────────────────────────────────────────────
# 生成报告
# ──────────────────────────────────────────────

def generate_report(sector_ranked: dict, phase1_signals: dict = None,
                    deep_dive_result: dict = None, phase2_signals: dict = None,
                    timestamp: str = None) -> str:
    """生成Markdown格式板块扫描报告"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []
    lines.append(f"# 📊 A股板块扫描报告")
    lines.append(f"")
    lines.append(f"**生成时间**: {timestamp}")
    lines.append(f"")
    lines.append(f"## 一、板块排行总览")
    lines.append(f"")
    lines.append(f"| 板块 | 最佳推荐 | 得分 | 信号 | 操作 |")
    lines.append(f"|------|---------|:----:|:----:|:----:|")

    # 板块排名：按top1得分降序
    sector_top_scores = []
    for sector, scored in sector_ranked.items():
        if scored:
            sector_top_scores.append((sector, scored[0][1], scored))
        else:
            sector_top_scores.append((sector, 0, scored))
    sector_top_scores.sort(key=lambda x: x[1], reverse=True)

    for sector, top_score, scored in sector_top_scores:
        if not scored:
            lines.append(f"| {sector} | — | — | ⚪ | — |")
            continue
        ticker = scored[0][0]
        info = get_stock_info(ticker)
        name = info.name if info else ticker
        score = scored[0][1]
        emoji = "🟢" if score > 15 else ("🔴" if score < -15 else "🟡")
        action = "✅ 重点关注" if score > 20 else ("⚠️ 观望" if score > -10 else "❌ 回避")
        lines.append(f"| {sector} | {name} ({ticker}) | {score:+.1f} | {emoji} | {action} |")

    lines.append(f"")
    lines.append(f"## 二、各板块详细分析")
    lines.append(f"")

    for sector, top_score, scored in sector_top_scores:
        lines.append(f"### {sector}")
        lines.append(f"")

        if not scored:
            lines.append(f"*无候选股*")
            lines.append(f"")
            continue

        for i, (ticker, score, breakdown) in enumerate(scored):
            info = get_stock_info(ticker)
            name = info.name if info else ticker
            ind = info.industry if info else "??"
            rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"  #{i+1}"
            signal_emoji = "🟢" if score > 15 else ("🔴" if score < -15 else "🟡")

            lines.append(f"**{rank_emoji} {name} ({ticker})** — {ind}")
            lines.append(f"")
            lines.append(f"- **综合得分**: {score:+.1f} {signal_emoji}")
            lines.append(f"")

            # Agent信号详情
            if breakdown:
                bullish = []
                bearish = []
                neutral = []
                for agent_key, bd in breakdown.items():
                    s = bd.get("signal", "neutral")
                    c = bd.get("confidence", 0)
                    dn = bd.get("display_name", agent_key)
                    if s == "bullish":
                        bullish.append(f"{dn}({c}%)")
                    elif s == "bearish":
                        bearish.append(f"{dn}({c}%)")
                    else:
                        neutral.append(f"{dn}({c}%)")

                if bullish:
                    lines.append(f"  🟢 看多 ({len(bullish)}位): {' · '.join(bullish)}")
                if bearish:
                    lines.append(f"  🔴 看空 ({len(bearish)}位): {' · '.join(bearish)}")
                if neutral:
                    lines.append(f"  🟡 中性 ({len(neutral)}位): {' · '.join(neutral)}")

                # 推理详情
                lines.append(f"")
                lines.append(f"  > **推理过程**")
                for agent_key, bd in breakdown.items():
                    reasoning = bd.get("reasoning", "")
                    if reasoning:
                        dn = bd.get("display_name", agent_key)
                        s = bd.get("signal", "neutral")
                        c = bd.get("confidence", 0)
                        # Extract plain text reasoning; skip if only metadata dict
                        re = reasoning.get("reasoning", reasoning) if isinstance(reasoning, dict) else reasoning
                        if isinstance(re, dict):
                            # No plain text field — show signal summary instead
                            sig_summary = f"[{s} {c}%]"
                            short_re = sig_summary
                        else:
                            short_re = str(re).strip()[:120]
                        lines.append(f"  > **{dn}** ({s}, {c}%): {short_re}")

            lines.append(f"")

    # 总结
    lines.append(f"## 三、操作建议")
    lines.append(f"")
    bullish_sectors = [(s, sc) for s, sc, _ in sector_top_scores if sc > 15]
    bearish_sectors = [(s, sc) for s, sc, _ in sector_top_scores if sc < -15]

    if bullish_sectors:
        lines.append(f"### 🟢 重点关注的板块")
        for s, sc in bullish_sectors:
            lines.append(f"- **{s}** (得分 {sc:+.1f})")
        lines.append(f"")

    if bearish_sectors:
        lines.append(f"### 🔴 需要回避的板块")
        for s, sc in bearish_sectors:
            lines.append(f"- **{s}** (得分 {sc:+.1f})")
        lines.append(f"")

    lines.append(f"### ⚠️ 风险提示")
    lines.append(f"本报告由AI大师分析系统自动生成，仅供参考，不构成投资建议。")
    lines.append(f"所有Agent信号的置信度基于其各自分析框架计算，可能存在模型偏差。")
    lines.append(f"")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="A股板块扫描仪 — 19位大师×16板块一览",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--sectors", type=str, nargs="+", default=None,
                        help="指定板块（默认全部16板块）")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None,
                        help="只跑Phase 1（快速计算）或Phase 2（LLM大师）")
    parser.add_argument("--top-n", type=int, default=3,
                        help="每板块候选股数（默认3）")
    parser.add_argument("--llm-top", type=int, default=1,
                        help="每板块送入LLM大师的top数（默认1）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出报告到文件")
    parser.add_argument("--list-sectors", action="store_true",
                        help="列出所有板块及候选股数")
    args = parser.parse_args()

    print("\n╔════════════════════════════════════╗")
    print("║   A股板块扫描仪 v1.0                ║")
    print("║   19位投资大师 × 16板块              ║")
    print("╚════════════════════════════════════╝")

    # 板块选择
    if args.list_sectors:
        print("\n[可选板块]")
        for s in sorted(SECTOR_POOL.keys()):
            raw = get_stocks_by_sector(s)
            filtered = filter_st_stocks(raw, min_market_cap=10.0)
            print(f"  {s:8s}: {len(raw):3d}只 → 过滤后 {len(filtered):3d}只")
        return

    sectors = args.sectors or sorted(SECTOR_POOL.keys())
    print(f"\n📋 板块: {len(sectors)} 个 | 候选: 每板块Top {args.top_n}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Phase 1
    if args.phase is None or args.phase == 1:
        sector_ranked, phase1_signals = phase1_scan(sectors, args.top_n)
        print(f"\n✅ Phase 1 完成! 各板块前3名得分:")
    else:
        # 从缓存加载
        cache_file = f"scan_{timestamp}_phase1.json"
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                cached = json.load(f)
            sector_ranked = cached["sector_ranked"]
            phase1_signals = cached["phase1_signals"]
        else:
            # Fallback: run phase1 anyway
            sector_ranked, phase1_signals = phase1_scan(sectors, args.top_n)

    if args.phase == 1:
        # 生成Phase1报告
        report = generate_report(sector_ranked, phase1_signals)
        print("\n" + report)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\n📝 报告已保存: {args.output}")
        else:
            out_file = f"scan_report_{timestamp}_phase1.md"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\n📝 报告已保存: {out_file}")
        return

    # Phase 2
    if args.phase is None or args.phase == 2:
        full_result, phase2_signals = phase2_deep_dive(sector_ranked, args.llm_top)
        print(f"\n✅ Phase 2 完成!")

        # 生成完整报告
        report = generate_report(full_result)
        print("\n" + report)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\n📝 报告已保存: {args.output}")
        else:
            out_file = f"scan_report_{timestamp}_full.md"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\n📝 报告已保存: {out_file}")


if __name__ == "__main__":
    main()
