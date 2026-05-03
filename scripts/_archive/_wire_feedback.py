#!/usr/bin/env python3
"""Wire SignalMemory.record() into scanner.py"""
with open(r'D:\AIHedgeFund\ai-hedge-fund-main\src\surge\scanner.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add record_signals param to function signature
old_sig = """    max_per_sector: int = 30,
) -> list[dict]:
    \"\"\"
    全市场形态扫描

    Args:
        stock_pool: 股票代码列表。None 时使用 16 板块池
        days: 获取多少天的数据（用于计算）
        min_price: 最低股价过滤
        max_price: 最高股价过滤
        save_report: 是否保存报告到 quant_archive
        params: 参数覆盖

    Returns:
        sorted signals list (按 final_score 降序)
    \"\"\""""

new_sig = """    max_per_sector: int = 30,
    record_signals: bool = True,
) -> list[dict]:
    \"\"\"
    全市场形态扫描

    Args:
        stock_pool: 股票代码列表。None 时使用 16 板块池
        days: 获取多少天的数据（用于计算）
        min_price: 最低股价过滤
        max_price: 最高股价过滤
        save_report: 是否保存报告到 quant_archive
        params: 参数覆盖
        record_signals: 是否自动记录信号到 feedback SignalMemory

    Returns:
        sorted signals list (按 final_score 降序)
    \"\"\""""

assert old_sig in text, "signature block not found!"
text = text.replace(old_sig, new_sig)

# 2. Add signal recording before return
old_end = """    # ── 保存报告 ──
    if save_report:
        report_path = _save_report(sorted_signals, params)
    else:
        report_path = None

    return sorted_signals"""

new_end = """    # ── 保存报告 ──
    if save_report:
        report_path = _save_report(sorted_signals, params)
    else:
        report_path = None

    # ── 自动记录到反馈系统（自我进化） ──
    if record_signals and sorted_signals:
        try:
            from src.surge.feedback import SignalMemory
            memory = SignalMemory()
            recorded = 0
            for s in sorted_signals:
                if s.get("signal_grade") in ("STRONG", "WEAK"):
                    memory.record(s)
                    recorded += 1
            print(f"\\n[feedback] recorded {recorded} signals to SignalMemory")
            print(f"[feedback] total in memory: {len(memory.signals)}")
        except Exception as ex:
            logger.debug(f"feedback record failed: {ex}")

    return sorted_signals"""

assert old_end in text, "end block not found!"
text = text.replace(old_end, new_end)

with open(r'D:\AIHedgeFund\ai-hedge-fund-main\src\surge\scanner.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("scanner.py patched OK")

# Quick verify
with open(r'D:\AIHedgeFund\ai-hedge-fund-main\src\surge\scanner.py', 'r', encoding='utf-8') as f:
    c = f.read()
print(f"record_signals: {'record_signals' in c}")
print(f"SignalMemory: {'SignalMemory' in c}")
print(f"Total lines: {len(c.splitlines())}")
