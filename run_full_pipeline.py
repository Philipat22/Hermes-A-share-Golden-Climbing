#!/usr/bin/env python3
"""全量21位分析师 + 风控 + 组合管理 端到端测试"""
import sys, os, io, json
sys.stdout.reconfigure(encoding='utf-8')

env_path = r'D:\AIHedgeFund\ai-hedge-fund-main\.env'
with open(env_path, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip()

sys.path.insert(0, r'D:\AIHedgeFund\ai-hedge-fund-main')

from src.main_astock import run_astock_analysis, get_stock_info
from src.utils.analysts import ANALYST_CONFIG

# 全部21位分析师
all_analysts = [
    # LLM 大师 (8位，已验证)
    "warren_buffett", "charlie_munger", "ben_graham", "cathie_wood",
    "phil_fisher", "peter_lynch",
    # LLM 大师 (6位，扩展)
    "aswath_damodaran", "bill_ackman", "michael_burry", "mohnish_pabrai",
    "nassim_taleb", "rakesh_jhunjhunwala", "stanley_druckenmiller",
    # 纯计算 (6位)
    "fundamentals_analyst", "technical_analyst", "valuation_analyst",
    "growth_analyst", "sentiment_analyst", "news_sentiment_analyst",
]

tickers = ["600519.SH", "000858.SZ", "000568.SZ"]

print("=" * 70)
print(f"🚀 AI Hedge Fund — 全量测试 ({len(all_analysts)}位分析师 × {len(tickers)}只股票)")
print(f"分析师: {', '.join(all_analysts[:5])}...")
print("=" * 70)

result = run_astock_analysis(
    tickers=tickers,
    selected_analysts=all_analysts,
    model_name="deepseek-chat",
    model_provider="DeepSeek",
    show_reasoning=False,
)

# ========== 汇总 ==========
ana_signals = result.get("analyst_signals", {})

# 按ticker聚合
by_ticker = {}
for agent, signals in ana_signals.items():
    if not isinstance(signals, dict):
        continue
    for ticker, sig in signals.items():
        if ticker not in tickers:
            continue
        if isinstance(sig, dict):
            by_ticker.setdefault(ticker, {})[agent] = sig

print(f"\n{'='*70}")
print(f"📊 最终汇总")
print(f"{'='*70}\n")

for ticker in by_ticker:
    agents = by_ticker[ticker]
    info = get_stock_info(ticker)
    name = info.name if info else ticker
    print(f"▌{name} ({ticker})")
    print(f"  {'─'*55}")
    
    # 分类统计
    by_type = {"LLM大师": {}, "计算型": {}, "风控": {}, "组合管理": {}}
    for akey, sig in agents.items():
        s = sig.get("signal", "?")
        c = sig.get("confidence", 0) or 0
        emoji = {"bullish":"🟢","bearish":"🔴","neutral":"🟡","hold":"🟡","buy":"🟢","sell":"🔴","?":"⚪"}.get(s,"⚪")
        cfg = ANALYST_CONFIG.get(akey, {})
        display = cfg.get("display_name", akey)
        if akey == "risk_management_agent":
            c = sig.get("remaining_position_limit", 0)
            vol_metrics = sig.get("volatility_metrics", {})
            vol = vol_metrics.get("daily_volatility", "?") if isinstance(vol_metrics, dict) else "?"
            vol_str = f"{float(vol):7.2%}" if isinstance(vol, (int, float)) else str(vol)
            print(f"  ⚪ 风险控制:                最大仓位={c:>8.2f}元 | 波动率={vol_str}")
        elif akey == "portfolio_manager":
            dec = sig.get("decision", sig.get("action", "?"))
            print(f"  📋 组合管理:                 决策={dec}")
        else:
            # LLM大师 vs 纯计算
            display_name = cfg.get("display_name", akey)
            print(f"  {emoji} {display_name:30s}: {s:8s} {c:3.0f}%")
    
    # LLM大师加权统计
    bullish = bearish = neutral = 0
    count_llm = 0
    for akey, sig in agents.items():
        if akey in ("risk_management_agent", "portfolio_manager"):
            continue
        s = sig.get("signal", "?")
        c = sig.get("confidence", 0) or 0
        if s in ("bullish", "buy"):
            bullish += c
            count_llm += 1
        elif s in ("bearish", "sell"):
            bearish += c
            count_llm += 1
        else:
            neutral += c
            count_llm += 1
    
    total = bullish + bearish + neutral
    if total > 0:
        verdict = "🟢 偏多" if bullish > bearish else ("🔴 偏空" if bearish > bullish else "🟡 中性")
        print(f"  {'─'*55}")
        print(f"  {verdict:>45s} (多{bullish:4.0f} 空{bearish:4.0f} 中{neutral:4.0f})")
    print()

# 检查portfolio_manager
pm = ana_signals.get("portfolio_manager", {})
if pm:
    print(f"{'='*70}")
    print("📋 组合决策")
    print(f"{'='*70}")
    # Find the decision - could be per-ticker or a single dict
    if isinstance(pm, dict):
        dec = pm.get("decision", pm.get("action", str(pm)))
        reas = pm.get("reasoning", "")
        print(f"  决策: {dec}")
        print(f"  理由: {reas}")

print(f"\n{'='*70}")
print("✅ 测试完成")
print(f"{'='*70}")
