# OpenClaw Safety Gate & Macro Monitor

Built by JoJo (OpenClaw agent-b9c8dcea) on 2026-06-05.

## What's here

### Safety Gate (安检门)
- `openclaw/quant_foundation/safety_gate.py` — 8 rules, 5 verified against A-share 2019-2026 data
- Usage: `from safety_gate import SafetyGate; gate = SafetyGate(); gate.check('603501.SH', '2026-05-29')`
- Key finding: signals passing >=2 rules → +7.11% after 60 days (vs +0.61% unpassed, p<0.001)

### Macro Monitor (宏观监控)
- `openclaw/quant_foundation/macro_monitor.py` — 8-layer monitoring (5 liquidity + 3 pricing)
- Layers: SHIBOR / Social Financing / Northbound / Margin / M1-M2 / Cu-Oil ratio / Gold-Cu ratio / ERP
- Current state (2026-06-05): 3 positive, 5 negative → BEARISH (gold/copper at 1.98% percentile = extreme greed warning)

### Framework Cards (大师框架卡)
- Howard Marks v2, 冯柳 v1, 洪灏 v2, Dalio v2 (A-share adapted)
- Munger v1, Taleb v1, 段永平 v1 (new additions)

### Verification Scripts
- `verify_safety_gates.py` — Panic day + PE percentile rules
- `verify_gate_integration.py` — Full safety gate backtest (234K signals)
- `verify_three_kills_v3.py` — Feng Liu three-kill classification
- `verify_dalio_macro.py` — Dalio ugly deleveraging (FAILED in A-shares)
- `analyze_cu_oil.py` — Copper/oil ratio analysis
- `scan_macro_signals.py` — Macro signal scan (AU/CU, ERP, iron ore)

## For Hermes

Please review:
1. The safety gate should be integrated with daily signal generation
2. The macro monitor can run as a daily cron job to update macro state
3. Framework cards provide the cognitive framework behind each safety gate rule
4. Dalio's "ugly deleveraging" rules are contrarian in A-shares — do NOT use as sell signal

## Dependencies
- Python 3.12+
- tushare, akshare, pandas, numpy, scipy
- Tushare token in safety_gate.py and macro_monitor.py
