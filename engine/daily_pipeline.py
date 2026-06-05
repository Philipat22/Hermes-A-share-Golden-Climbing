"""
Daily Pipeline — Strategy-agnostic daily report.

Usage:
  python engine/daily_pipeline.py --strategy north_margin --date 2026-06-02
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import strategies
strategies.load_all()
from strategies import STRATEGIES
from engine.scanner import Scanner
from engine.reminders import get_reminders, format_reminders
from engine.production import MarketRegime, PositionSizer


def daily_pipeline(strategy_name: str, date_str: str, top_n: int = 15):
    """Run daily scan for a strategy and produce a report."""

    strat = STRATEGIES.get(strategy_name)
    if strat is None:
        return f"Unknown strategy: {strategy_name}\nAvailable: {list(STRATEGIES.keys())}"

    scanner = Scanner()
    scanner.load()

    class EngineAdapter:
        def __init__(self, scanner):
            self._all_stocks = scanner._all_stocks
            self.csi300 = scanner.csi300
            self.prices = scanner.prices
        def _regime_at(self, date_str):
            return scanner.regime_at(date_str)

    engine_adapter = EngineAdapter(scanner)
    signal_filter = strat.create_signal_filter(engine_adapter)
    candidates = scanner.scan(date_str, signal_filter)

    regime = scanner.regime_at(date_str)

    # ── Determine max DD, market signals ──
    max_dd = min((c.dd for c in candidates if c.dd is not None), default=0)
    # Get market signal info: try scanner, fallback to strategy
    ms = getattr(scanner, '_market_signals', {}).get(date_str, {})
    nm_active = ms.get("nm_active", len(candidates) > 0)  # If NM strategy produced candidates, NM is on

    if not candidates:
        # Build context for reminders even when no candidates
        ctx = {
            "regime": regime,
            "dd": None,
            "north_margin_active": False,
            "deepdd_cross": False,
            "positions": 0,
            "margin_growth": 0.0,
            "new_entry": True,
            "ai_sector": False,
        }
        reminders = format_reminders(get_reminders(ctx))
        return (
            f"=== Daily Report - {date_str} ===\n"
            f"  Strategy: {strat.name}\n"
            f"  Regime: {regime.upper()} | NorthMargin: {'ON' if nm_active else 'OFF'}\n"
            f"  Candidates: 0\n"
            f"{reminders}"
        )

    bt = strat.backtest

    lines = [
        "=" * 60,
        f"  DAILY REPORT - {date_str}",
        f"  Strategy: {strat.name} - {strat.description}",
        f"  Regime: {regime.upper()} | NorthMargin: {'ON' if nm_active else 'OFF'}",
        f"  Candidates: {len(candidates)}",
        "=" * 60,
        f"  {'Code':<12s} {'Price':>8s} {'DD':>8s}",
        f"  {'-'*12} {'-'*8} {'-'*8}",
    ]

    for c in candidates[:top_n]:
        dd_str = f"{c.dd:>+7.1%}" if c.dd is not None else "     N/A"
        lines.append(f"  {c.code:<12s} {c.close:>8.2f} {dd_str}")

    # ── Position sizing based on regime ──
    sizer = PositionSizer()
    plan = sizer.plan(regime, is_confirmed=False)
    
    rec = plan.scout_pct
    med = bt.median if bt else 0.0
    wr = bt.win_rate if bt else 0.0
    lines.extend([
        "",
        f"  Position: scout {plan.scout_pct:.0%} → target {plan.target_pct:.0%} (regime ×{plan.regime_multiplier:.0%})",
        f"  Regime advice: {MarketRegime.regime_advice(regime)}",
        f"  Hard stops: -5% daily / -15% trade | Account: 40000",
        f"  Backtest: median={med:+.1%}, WR={wr:.0%}",
        "=" * 60,
    ])

    # ── Reminders ──
    ctx = {
        "regime": regime,
        "dd": max_dd,
        "north_margin_active": nm_active,
        "deepdd_cross": any(c.deepdd_cross if hasattr(c, 'deepdd_cross') else False for c in candidates[:5]),
        "positions": 5,  # placeholder; connect to tracker later
        "margin_growth": 0.0,
        "new_entry": True,
        "ai_sector": False,
    }
    reminders = format_reminders(get_reminders(ctx))
    lines.append(reminders)

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily Pipeline")
    parser.add_argument("--strategy", type=str, default="north_margin")
    parser.add_argument("--date", type=str, default="2026-04-27")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    print(daily_pipeline(args.strategy, args.date, top_n=args.top))
