"""
Decision Reminders — Context-aware wisdom from Marks + 冯柳.

Triggers based on market state, signal characteristics, and portfolio status.
Surfaced in daily reports at decision time.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Reminder:
    trigger: str
    message: str
    source: str
    priority: str  # "high", "medium", "low"


# ═══════════════════════════════════════════
# Reminder database
# ═══════════════════════════════════════════

REMINDERS = [
    # ── Market State ──
    Reminder(
        trigger="regime == 'severe_bear'",
        message="NorthMargin historically +11.6% in severe bear. This is where the signal is strongest.",
        source="Marks #28: 'When everyone believes something is risky, they refuse to buy, "
               "driving the price to where it is NOT risky.'",
        priority="high",
    ),
    Reminder(
        trigger="regime == 'bull'",
        message="Bull market: NorthMargin median drops to +0.4%. Reduce position size, require "
                "dual confirmation.",
        source="Marks #81: 'We may never know where we're going, but we'd better know where we are.'",
        priority="high",
    ),
    Reminder(
        trigger="regime == 'unknown'",
        message="Can't identify market state. NorthMargin historically -3.3% in unknown regime. "
                "Skip or reduce to minimum position.",
        source="Fengliu: 'When you can't see, don't act.'",
        priority="high",
    ),

    # ── Signal Characteristics ──
    Reminder(
        trigger="dd < -0.50",
        message="DD > 50%: Deep drawdown. Fengliu says at lows, default assumption is 'it has "
                "value' unless you can prove otherwise. Don't need to prove it will go up.",
        source="Fengliu: 'At lows, if you can't prove it's worthless, assume it has value.'",
        priority="medium",
    ),
    Reminder(
        trigger="north_margin_active and deepdd_cross",
        message="Dual confirmation: NM + DeepDD. Historically median +4.8%. This is the highest "
                "confidence tier.",
        source="4-factor validation (2026-06-02)",
        priority="high",
    ),

    # ── Portfolio / Position ──
    Reminder(
        trigger="positions >= 4",
        message="At 4+ positions. Marks: Diversification is defense. Fengliu: Hold many, let "
                "probability work. Don't concentrate prematurely.",
        source="Marks #89 + Fengliu 'odds before probability'",
        priority="medium",
    ),
    Reminder(
        trigger="new_entry",
        message="First entry. Fengliu: Small scout position first. Feel the market before committing.",
        source="Fengliu: 'Open small, feel whether you're anxious or calm, then decide.'",
        priority="medium",
    ),

    # ── Risk ──
    Reminder(
        trigger="margin_growth > 0.30",
        message="Margin accelerating >30%/quarter. Marks: Risk accumulates where it's not perceived. "
                "Reduce total exposure by 30%.",
        source="Marks #29: 'When everyone believes something has no risk, price gets bid up to "
               "where it carries enormous risk.'",
        priority="high",
    ),
    Reminder(
        trigger="open_positions_with_loss",
        message="Open positions showing loss. Marks: Being too early and being wrong look identical. "
                "If position size is within limits, hold. Do NOT add to losers.",
        source="Marks Ch.3: 'Overly ahead of your time is indistinguishable from being wrong.'",
        priority="high",
    ),

    # ── AI-specific ──
    Reminder(
        trigger="ai_sector_signal",
        message="AI sector: Fengliu says in bull markets (strengthening phase), trend reinforces "
                "expectation. Ride the trend but watch for deceleration.",
        source="Fengliu: 'In the strengthening phase, trend is the key factor.'",
        priority="medium",
    ),

    # ── General ──
    Reminder(
        trigger="always",
        message="Marks: 'We can't predict the future. We prepare for various futures.' "
                "No leverage. Keep 20% cash. Hard stops at -15%.",
        source="Marks #78, #100",
        priority="low",
    ),
]


def get_reminders(context: dict) -> List[Reminder]:
    """
    Given current context, return relevant reminders sorted by priority.

    Context keys:
      regime: str            — 'severe_bear', 'bear', 'bull', 'recovery', etc.
      dd: float              — current max DD among candidates
      north_margin_active: bool
      deepdd_cross: bool
      positions: int         — current open paper positions
      margin_growth: float   — quarterly margin growth rate
      new_entry: bool
      ai_sector: bool
    """
    triggered = []

    for r in REMINDERS:
        if _evaluate(r.trigger, context):
            triggered.append(r)

    # Sort: high priority first
    priority_order = {"high": 0, "medium": 1, "low": 2}
    triggered.sort(key=lambda r: priority_order.get(r.priority, 2))

    # Remove duplicate sources
    seen_sources = set()
    deduped = []
    for r in triggered:
        if r.source not in seen_sources:
            deduped.append(r)
            seen_sources.add(r.source)

    return deduped


def _evaluate(trigger: str, ctx: dict) -> bool:
    """Simple condition evaluator."""
    try:
        # Evaluate simple expressions
        if trigger == "always":
            return True

        # Replace context variables
        expr = trigger
        for key, val in ctx.items():
            if isinstance(val, str):
                expr = expr.replace(key, f"'{val}'")
            elif isinstance(val, bool):
                expr = expr.replace(key, str(val))
            elif val is None:
                expr = expr.replace(key, "None")
            else:
                expr = expr.replace(key, str(val))

        # Safe eval (only simple comparisons)
        return bool(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return False


def format_reminders(reminders: List[Reminder]) -> str:
    """Format reminders for display in daily report."""
    if not reminders:
        return ""

    lines = [
        "=" * 60,
        "  DECISION REMINDERS",
        "=" * 60,
    ]

    for r in reminders:
        tag = "!!" if r.priority == "high" else "!" if r.priority == "medium" else " "
        lines.append(f"  [{tag}] {r.message}")
        lines.append(f"       — {r.source}")
        lines.append("")

    return "\n".join(lines)
