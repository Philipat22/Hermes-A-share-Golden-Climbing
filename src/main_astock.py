#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股对冲基金系统主入口
使用 DeepSeek + 申万行业选股 + 18位投资大师 Agent

用法:
    poetry run python src/main_astock.py
    poetry run python src/main_astock.py --sector 白酒
    poetry run python src/main_astock.py --tickers 600519.SH 000858.SZ --analysts fundamentals_analyst_agent
    poetry run python src/main_astock.py --sector 半导体 --analysts fundamentals_analyst_agent
"""

from __future__ import annotations
import sys

import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse
import json

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from src.tools.a_stock_api import (
    get_stocks_by_sector,
    filter_st_stocks,
    get_prices,
    get_stock_info,
    SECTOR_POOL,
)
from src.graph.state import AgentState
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.utils.analysts import (
    ANALYST_CONFIG,
    ANALYST_ORDER,
    get_analyst_nodes,
)
from src.agents.fundamentals import fundamentals_analyst_agent
from src.agents.warren_buffett import warren_buffett_agent
from src.agents.charlie_munger import charlie_munger_agent
from src.agents.ben_graham import ben_graham_agent
from src.agents.cathie_wood import cathie_wood_agent
from src.agents.peter_lynch import peter_lynch_agent
from src.agents.phil_fisher import phil_fisher_agent
from src.agents.bill_ackman import bill_ackman_agent
from src.agents.michael_burry import michael_burry_agent
from src.agents.stanley_druckenmiller import stanley_druckenmiller_agent
from src.agents.nassim_taleb import nassim_taleb_agent
from src.agents.growth_agent import growth_analyst_agent
from src.agents.sentiment import sentiment_analyst_agent
from src.agents.technicals import technical_analyst_agent
from src.agents.valuation import valuation_analyst_agent
from src.agents.aswath_damodaran import aswath_damodaran_agent
from src.agents.rakesh_jhunjhunwala import rakesh_jhunjhunwala_agent
from src.agents.mohnish_pabrai import mohnish_pabrai_agent
from src.agents.news_sentiment import news_sentiment_agent
from src.agents.bill_ackman import bill_ackman_agent

load_dotenv()

# ──────────────────────────────────────────────
# 打印工具
# ──────────────────────────────────────────────

def print_banner():
    banner = """
╔══════════════════════════════════════════════════════╗
║           A股智能投研系统 v1.0                        ║
║     DeepSeek + 18位投资大师 + 申万行业选股            ║
╚══════════════════════════════════════════════════════╝
"""
    print(banner)


def print_signal_result(ticker: str, signal: dict):
    """打印单只股票分析结果"""
    info = get_stock_info(ticker)
    name = info.name if info else ticker

    signal_str = signal.get("signal", "unknown")
    conf = signal.get("confidence", 0)
    reasoning = signal.get("reasoning", {})

    emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(signal_str, "⚪")

    print(f"\n{'='*60}")
    print(f"  {emoji} {name}({ticker})")
    print(f"{'='*60}")
    print(f"  信号: {signal_str.upper()}  置信度: {conf}%")

    if reasoning:
        for key, val in reasoning.items():
            sig = val.get("signal", "")
            detail = val.get("details", "")
            e = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(sig, "⚪")
            print(f"  {e} {key}: {sig} — {detail[:60]}")


def print_portfolio_decision(decision: dict):
    """打印组合决策"""
    if not decision:
        print("\n⚠️ 无交易决策")
        return

    print(f"\n📊 组合决策:")
    print(f"  现金: ¥{decision.get('cash', 0):,.0f}")
    print(f"  信号: {decision.get('signal', 'N/A')}")
    print(f"  理由: {decision.get('reasoning', 'N/A')[:100]}")

    actions = decision.get("actions", [])
    if actions:
        print(f"\n  执行动作 ({len(actions)}项):")
        for a in actions:
            print(f"    • {a}")


# ──────────────────────────────────────────────
# 选股引擎
# ──────────────────────────────────────────────

def select_stocks(sector: str = None, top_n: int = 5, min_market_cap: float = 10.0) -> list[str]:
    """
    板块选股 + 质量过滤

    Args:
        sector: 板块名（如 "白酒", "半导体"），None则使用14板块合并
        top_n: 每个板块最多取几只
        min_market_cap: 最低流通市值（亿元）

    Returns:
        股票代码列表
    """
    print(f"\n[选股] 板块={sector or '全部14板块'}, top_n={top_n}, 最小市值={min_market_cap}亿")

    if sector:
        # 单板块选股
        raw = get_stocks_by_sector(sector)
        print(f"  板块原始: {len(raw)} 只")
        filtered = filter_st_stocks(raw, min_market_cap=min_market_cap)
        print(f"  过滤后: {len(filtered)} 只")
        return filtered[:top_n]

    # 多板块合并
    all_selected = {}
    for sec, stocks in SECTOR_POOL.items():
        filtered = filter_st_stocks(stocks, min_market_cap=min_market_cap)
        # 按市值排序（简化版：取前 top_n）
        all_selected[sec] = filtered[:top_n]

    # 合并
    result = []
    for sec, stocks in all_selected.items():
        for s in stocks:
            if s not in result:
                result.append(s)

    print(f"  14板块合计: {len(result)} 只")
    return result


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

def run_astock_analysis(
    tickers: list[str],
    start_date: str = None,
    end_date: str = None,
    selected_analysts: list[str] = None,
    model_name: str = "deepseek-chat",
    model_provider: str = "DeepSeek",
    show_reasoning: bool = True,
    initial_cash: float = 100000.0,
):
    """
    运行 A股投研分析

    Args:
        tickers: 股票列表
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        selected_analysts: 分析师列表（默认全部）
        model_name: 模型名
        model_provider: 模型提供商
        show_reasoning: 显示推理过程
        initial_cash: 初始资金
    """
    # 默认日期：最近6个月
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - relativedelta(months=6)).strftime("%Y-%m-%d")

    print(f"\n[参数] 股票: {len(tickers)} 只 | 日期: {start_date} ~ {end_date}")
    print(f"[参数] 模型: {model_provider}/{model_name}")
    analyst_names = selected_analysts or list(ANALYST_CONFIG.keys())
    print(f"[参数] 大师: {len(analyst_names)} 位")

    # 构造 portfolio
    portfolio = {
        "cash": initial_cash,
        "margin_requirement": 0.0,
        "margin_used": 0.0,
        "positions": {
            ticker: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            }
            for ticker in tickers
        },
        "realized_gains": {
            ticker: {"long": 0.0, "short": 0.0}
            for ticker in tickers
        },
    }

    # 分析师节点映射
    analyst_nodes = get_analyst_nodes()

    # ────────────────────────────────────────
    # 安全包装器：单个Agent出错不终止全流程
    # ────────────────────────────────────────
    def safe_agent_wrapper(node_name: str, agent_func):
        """Wrap an agent node with crash isolation."""
        def wrapped(state: AgentState):
            try:
                return agent_func(state)
            except Exception as e:
                import traceback
                error_msg = f"[{node_name}] 崩溃: {e}"
                trace = traceback.format_exc()
                print(f"\n⚠️  {error_msg}")
                # 只打印前3行traceback避免刷屏
                for line in trace.split('\n')[:4]:
                    print(f"   {line}")
                # 返回fallback信号：该Agent对所有ticker给中性0%   
                tickers = state.get("data", {}).get("tickers", [])
                fallback = {
                    "signal": "neutral",
                    "confidence": 0,
                    "reasoning": f"分析异常: {e}",
                }
                signals = {}
                for t in tickers:
                    signals[t] = {**fallback, "ticker": t}
                return {
                    "messages": state.get("messages", []),
                    "data": {
                        **state.get("data", {}),
                        "analyst_signals": {
                            **state.get("data", {}).get("analyst_signals", {}),
                            node_name: signals,
                        },
                    },
                }
        return wrapped

    # 构建 StateGraph workflow
    workflow = StateGraph(AgentState)

    # 入口节点
    def start_node(state: AgentState):
        return state

    workflow.add_node("start", start_node)

    # 添加选中的分析师（安全包装）
    for key in analyst_names:
        if key in analyst_nodes:
            node_name, node_func = analyst_nodes[key]
            wrapped = safe_agent_wrapper(node_name, node_func)
            workflow.add_node(node_name, wrapped)
            workflow.add_edge("start", node_name)

    # 风控 + 组合管理（也加安全包装）
    workflow.add_node("risk_management_agent", safe_agent_wrapper("risk_management_agent", risk_management_agent))
    workflow.add_node("portfolio_manager", safe_agent_wrapper("portfolio_manager", portfolio_management_agent))

    for key in analyst_names:
        if key in analyst_nodes:
            node_name = analyst_nodes[key][0]
            workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)
    workflow.set_entry_point("start")

    compiled = workflow.compile()

    print(f"\n[启动] 18位大师并行分析...")
    final_state = compiled.invoke(
        {
            "messages": [
                HumanMessage(content="分析A股股票，给出交易信号。")
            ],
            "data": {
                "tickers": tickers,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
                "analyst_signals": {},
            },
            "metadata": {
                "show_reasoning": show_reasoning,
                "model_name": model_name,
                "model_provider": model_provider,
            },
        }
    )

    return {
        "analyst_signals": final_state["data"]["analyst_signals"],
        "messages": final_state["messages"],
    }


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="A股智能投研系统 — DeepSeek + 18位投资大师",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 白酒板块 + 巴菲特 + 芒格
  poetry run python src/main_astock.py --sector 白酒 --analysts warren_buffett charlie_munger

  # 指定股票 + 全部大师
  poetry run python src/main_astock.py --tickers 600519.SH 000858.SZ

  # 半导体板块 + growth 大师们
  poetry run python src/main_astock.py --sector 半导体 --analysts cathie_wood phil_fisher growth_analyst

  # 14板块合并 + 3位大师
  poetry run python src/main_astock.py --analysts warren_buffett ben_graham charlie_munger

可用大师:
  warren_buffett, charlie_munger, ben_graham, cathie_wood,
  peter_lynch, phil_fisher, bill_ackman, michael_burry,
  stanley_druckenmiller, nassim_taleb, growth_analyst,
  fundamentals_analyst, sentiment_analyst, technical_analyst,
  valuation_analyst, aswath_damodaran, rakesh_jhunjhunwala,
  mohnish_pabrai, news_sentiment_agent

可用板块:
  """ + ", ".join(SECTOR_POOL.keys()),
    )

    parser.add_argument(
        "--sector", type=str, default=None,
        help="板块名称（如：白酒、半导体、银行、医药、计算机）"
    )
    parser.add_argument(
        "--tickers", type=str, nargs="+", default=None,
        help="指定股票代码（如：600519.SH 000858.SZ）"
    )
    parser.add_argument(
        "--analysts", type=str, nargs="+",
        default=["fundamentals_analyst_agent"],
        help="使用的分析师（默认: fundamentals_analyst_agent）"
    )
    parser.add_argument(
        "--top-n", type=int, default=5,
        help="每个板块最多选几只（默认5）"
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="开始日期 YYYY-MM-DD（默认6个月前）"
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="结束日期 YYYY-MM-DD（默认今天）"
    )
    parser.add_argument(
        "--initial-cash", type=float, default=100000.0,
        help="初始资金（默认10万元）"
    )
    parser.add_argument(
        "--no-reasoning", action="store_true",
        help="隐藏推理过程"
    )
    parser.add_argument(
        "--list-sectors", action="store_true",
        help="显示所有可用板块"
    )
    parser.add_argument(
        "--list-analysts", action="store_true",
        help="显示所有可用大师"
    )
    parser.add_argument(
        "--model", type=str, default="deepseek-chat",
        help="模型名称（默认: deepseek-chat）"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print_banner()

    if args.list_sectors:
        print("[可选板块]")
        for s in SECTOR_POOL:
            stocks = get_stocks_by_sector(s)
            filtered = filter_st_stocks(stocks)
            print(f"  {s:8s}: {len(filtered)} 只")
        sys.exit(0)

    if args.list_analysts:
        print("[可选投资大师]")
        for key, cfg in ANALYST_CONFIG.items():
            print(f"  {key:30s} — {cfg['display_name']}: {cfg['description']}")
        sys.exit(0)

    # 选股
    if args.tickers:
        tickers = args.tickers
        print(f"\n[输入] 指定股票: {tickers}")
    elif args.sector:
        tickers = select_stocks(sector=args.sector, top_n=args.top_n)
        if not tickers:
            print(f"\n❌ 板块 '{args.sector}' 无可用股票！")
            sys.exit(1)
    else:
        # 14板块合并
        tickers = select_stocks(top_n=args.top_n)

    print(f"\n[选股结果] {len(tickers)} 只股票:")
    for t in tickers:
        info = get_stock_info(t)
        name = info.name if info else "??"
        ind = info.industry if info else "??"
        print(f"  {t} {name} ({ind})")

    # 显示分析师
    print(f"\n[大师] 使用 {len(args.analysts)} 位分析师:")
    for a in args.analysts:
        if a in ANALYST_CONFIG:
            print(f"  • {ANALYST_CONFIG[a]['display_name']}")

    # 运行分析
    try:
        result = run_astock_analysis(
            tickers=tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            selected_analysts=args.analysts,
            model_name=args.model,
            model_provider="DeepSeek",
            show_reasoning=not args.no_reasoning,
            initial_cash=args.initial_cash,
        )

        # 打印结果
        print("\n\n" + "=" * 60)
        print("📊 分析结果汇总")
        print("=" * 60)

        analyst_signals = result["analyst_signals"]
        for ticker in tickers:
            # 收集该股票所有大师信号
            signals_for_ticker = {}
            for analyst_key, signals in analyst_signals.items():
                if ticker in signals:
                    signals_for_ticker[analyst_key] = signals[ticker]

            if signals_for_ticker:
                info = get_stock_info(ticker)
                name = info.name if info else ticker
                print(f"\n{name}({ticker}):")
                bullish = bearish = neutral = 0
                for analyst_key, sig in signals_for_ticker.items():
                    s = sig.get("signal", "unknown")
                    c = sig.get("confidence", 0)
                    cfg = ANALYST_CONFIG.get(analyst_key, {})
                    aname = cfg.get("display_name", analyst_key)
                    if s == "bullish":
                        bullish += c
                    elif s == "bearish":
                        bearish += c
                    else:
                        neutral += c
                    e = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(s, "⚪")
                    print(f"  {e} {aname}: {s} ({c}%)")

                total = bullish + bearish + neutral
                if total > 0:
                    verdict = "🟢 整体看多" if bullish > bearish else ("🔴 整体看空" if bearish > bullish else "🟡 中性")
                    print(f"  → {verdict} (多:{bullish:.0f} 空:{bearish:.0f} 中:{neutral:.0f})")

            # Agent对比信号卡 + 交易指令卡
            try:
                from src.utils.display import print_signal_card, print_trading_card
                print()
                print_signal_card(ticker, analyst_signals)
                print()
                print_trading_card(ticker, analyst_signals)
            except Exception:
                pass
        else:
            print(f"\n{ticker}: ⚠️ 无分析信号")

    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
