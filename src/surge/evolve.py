#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
surge/evolve.py — 自我进化闭环调度器

完整的反馈循环:
  1. 评估历史信号的实际表现 (evaluate)
  2. 根据结果调整引擎参数 (adjust)
  3. 打印进化摘要 (report)

用法:
  python -m src.surge.evolve                    # 完整闭环: 评估->调参->报告
  python -m src.surge.evolve evaluate           # 只做评估
  python -m src.surge.evolve adjust             # 只调参（需要有已评估的信号）
  python -m src.surge.evolve report             # 只打报告
  python -m src.surge.evolve full-scan          # 全扫描 + 闭环
"""
from __future__ import annotations
import sys, os, logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_evaluate(days_lookback: int = 5, holding_period: int = 10) -> int:
    """评估历史信号的实际表现"""
    from src.surge.feedback import SignalMemory, evaluate_signals
    memory = SignalMemory()
    
    pending = memory.get_pending_evaluation(min_age_days=days_lookback)
    if not pending:
        print(f"[评估] 没有待评估的信号（最近{days_lookback}天内无信号，或全部已评估）")
        print(f"[评估] 当前 SignalMemory 共 {len(memory.signals)} 条记录")
        return 0
    
    print(f"[评估] 待评估信号: {len(pending)} 个（{days_lookback}天前发出）")
    n = evaluate_signals(memory, days_lookback=days_lookback, holding_period=holding_period)
    if n > 0:
        print(f"\n[评估] 完成 {n} 个信号评估")
        print(memory.summary())
    return n


def run_adjust(min_signals: int = 10):
    """根据历史表现调整参数"""
    from src.surge.feedback import SignalMemory, adjust_params
    memory = SignalMemory()
    
    stats = memory.get_pattern_stats()
    total = sum(s["total"] for s in stats.values())
    
    print(f"[调参] 已评估信号: {total} 次（需≥{min_signals}次才调整）")
    if total < min_signals:
        print("[调参] 样本不足，跳过调整")
        return
    
    new_params = adjust_params(memory, min_signals=min_signals)
    
    print(f"\n[调参] 当前参数摘要:")
    for k in ["weak_signal", "strong_signal", "w_price_pattern", "w_volume", "w_sector", "w_acceleration"]:
        print(f"  {k} = {new_params.get(k, '?')}")

    # ── 情绪权重自动调整 ──
    print(f"\n[调参] 调整情绪融合权重...")
    changed = memory.adjust_emotion_weights()
    if changed:
        print(f"[调参] 情绪权重已更新")
        stats = memory.get_emotion_stats()
        if stats.get("status") == "ok":
            print(f"  评估信号数: {stats['evaluated_count']}")
            for b in stats.get("fusion_buckets", []):
                print(f"  融合分 {b['range']:5s}: {b['count']:3d}次, 胜率{b['win_rate']}%")
            print(f"  推荐权重: {stats.get('recommended_weights', {})}")
    else:
        print(f"[调参] 权重未更新（数据不足）")


def run_report():
    """打印进化报告"""
    from src.surge.feedback import SignalMemory
    memory = SignalMemory()
    
    total = len(memory.signals)
    evaluated = sum(1 for s in memory.signals if s["outcome_success"] is not None)
    
    print(f"=== 自我进化报告 ===")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"信号总数: {total}")
    print(f"已评估: {evaluated}")
    print(f"待评估: {total - evaluated}")
    
    if evaluated > 0:
        print()
        print(memory.summary())
    
    # 最近10个信号预览
    recent = memory.signals[-10:] if len(memory.signals) >= 10 else memory.signals
    print(f"\n最近 {len(recent)} 个信号:")
    for s in reversed(recent):
        outcome = "OK" if s.get("outcome_success") else ("FAIL" if s.get("outcome_success") is False else "...")
        ret = s.get("outcome_return", "")
        if ret is not None:
            ret_str = f"{ret*100:+.1f}%"
        else:
            ret_str = "待评估"
        print(f"  {s['ts_code']:>10} {s.get('pattern_type',''):>6} "
              f"score={s.get('final_score',0):3d} grade={s.get('signal_grade',''):>6} "
              f"-> {outcome} {ret_str}")
    
    print(f"\n信号记忆路径: {memory.filepath}")


def run_full_cycle():
    """完整的进化循环：扫描->评估->调参->报告"""
    print("=" * 50)
    print("  SURGE 自我进化闭环")
    print("=" * 50)
    
    # Step 1: 评估历史信号
    print("\n[1/3] 评估历史信号表现...")
    n = run_evaluate(days_lookback=5, holding_period=10)
    
    # Step 2: 参数调整
    print(f"\n[2/3] 参数调整...")
    run_adjust(min_signals=10)
    
    # Step 3: 报告
    print(f"\n[3/3] 进化报告...")
    run_report()
    
    print("\n" + "=" * 50)
    print("  完成")
    print("=" * 50)


def run_full_scan_cycle():
    """全市场扫描 + 进化闭环"""
    print("=" * 50)
    print("  SURGE 全扫描 + 自我进化")
    print("=" * 50)
    
    from src.surge.scanner import scan_market
    from src.surge.engine import load_params
    
    params = load_params()
    print(f"\n当前参数: weak={params.get('weak_signal')} strong={params.get('strong_signal')}")
    
    # Step 1: 全市场扫描（自动记录信号）
    print(f"\n[1/4] 全市场形态扫描...")
    signals = scan_market(params=params, record_signals=True)
    
    # Step 2: 评估历史信号
    print(f"\n[2/4] 评估历史信号表现...")
    run_evaluate(days_lookback=5, holding_period=10)
    
    # Step 3: 参数调整
    print(f"\n[3/4] 参数调整...")
    run_adjust(min_signals=10)
    
    # Step 4: 报告
    print(f"\n[4/4] 进化报告...")
    run_report()
    
    print("\n" + "=" * 50)
    print("  完成")
    print("=" * 50)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))
    # 确保 src 在路径中
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    
    cmd = sys.argv[1] if len(sys.argv) > 1 else "full-cycle"
    
    if cmd == "evaluate":
        run_evaluate()
    elif cmd == "adjust":
        run_adjust()
    elif cmd == "report":
        run_report()
    elif cmd == "full-scan":
        run_full_scan_cycle()
    else:
        run_full_cycle()
