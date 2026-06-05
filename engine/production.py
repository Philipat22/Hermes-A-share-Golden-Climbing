"""
Production Engine — Regime switching + position sizing + exit dashboard.

Integrates into daily_pipeline. Closes the last system engineering gaps.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class PositionPlan:
    """Position sizing recommendation."""
    scout_pct: float       # initial scout position (% of account)
    target_pct: float      # full position after confirmation
    max_pct: float         # hard cap
    regime_multiplier: float  # adjusted by market state
    kelly_raw: float       # raw Kelly for reference

    def summary(self) -> str:
        return (f"Scout: {self.scout_pct:.0%} → Target: {self.target_pct:.0%} "
                f"(max: {self.max_pct:.0%}, regime: {self.regime_multiplier:.0%})")


class MarketRegime:
    """Market state → position multiplier."""

    # Regime → position size multiplier
    # Conservative: bull=small, severe_bear=full
    MULTIPLIERS = {
        "severe_bear": 1.0,   # full position — best for mean_rev
        "bear":        0.8,   # 80% — good for mean_rev
        "sideways":    0.5,   # 50% — uncertain
        "recovery":    0.3,   # 30% — transitioning
        "bull":        0.2,   # 20% — mean_rev weakens
        "unknown":     0.0,   # don't trade
    }

    @classmethod
    def multiplier(cls, regime: str) -> float:
        return cls.MULTIPLIERS.get(regime, 0.0)

    @classmethod
    def regime_advice(cls, regime: str) -> str:
        advice = {
            "severe_bear": "Full position. Mean_rev strongest here (+11.6% historically).",
            "bear": "Normal position. Mean_rev works well.",
            "sideways": "Half position. Signals mixed.",
            "recovery": "Light position. Mean_rev weakening, momentum not yet confirmed.",
            "bull": "Minimal position. Mean_rev weak (+0.4%). Consider pausing.",
            "unknown": "Do not trade. Regime unclear.",
        }
        return advice.get(regime, "Unknown regime — do not trade.")


class PositionSizer:
    """Compute optimal position sizes."""

    def __init__(self, win_rate: float = 0.75, avg_win: float = 0.054,
                 avg_loss: float = -0.066, account_size: float = 40000):
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = abs(avg_loss)
        self.account = account_size

    def kelly(self) -> float:
        """Raw Kelly criterion."""
        if self.avg_win <= 0:
            return 0.0
        return max(0, self.win_rate - (1 - self.win_rate) / (self.avg_win / self.avg_loss))

    def plan(self, regime: str, is_confirmed: bool = False) -> PositionPlan:
        """Compute position plan. Regime affects scout only. Confirmed = full allocation."""
        raw_kelly = self.kelly()
        regime_mult = MarketRegime.multiplier(regime)

        # Scout: small position, scaled by regime (bull=small, bear=larger)
        scout = raw_kelly * 0.5 * regime_mult
        scout = min(scout, 0.05)  # scout max 5%

        # Confirmed: FULL allocation. Confirmation filter already ensures quality.
        # Regime does NOT reduce confirmed positions.
        if is_confirmed:
            target = raw_kelly * 0.75  # full Kelly × conservative factor
            target = min(target, 0.15)  # single stock max 15%
        else:
            target = scout  # not confirmed yet, stay at scout

        max_cap = 0.15

        return PositionPlan(
            scout_pct=scout,
            target_pct=target,
            max_pct=max_cap,
            regime_multiplier=regime_mult,
            kelly_raw=raw_kelly,
        )


@dataclass
class ExitDashboard:
    """Information to present at decision time."""

    code: str
    name: str = ""
    entry_price: float = 0
    current_price: float = 0
    peak_price: float = 0
    ma20: float = 0
    ma60: float = 0
    holding_days: int = 0
    confirmed: bool = False
    regime: str = ""

    @property
    def return_pct(self) -> float:
        return (self.current_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0

    @property
    def peak_return(self) -> float:
        return (self.peak_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0

    @property
    def giveback(self) -> float:
        return self.peak_return - self.return_pct

    @property
    def ma20_distance(self) -> float:
        return (self.current_price - self.ma20) / self.ma20 if self.ma20 > 0 else 0

    def dashboard(self) -> str:
        """Human-readable exit decision dashboard."""
        status = "OK" if self.current_price > self.ma20 else "WARN"
        lines = [
            f"  {'='*50}",
            f"  {self.code} {'('+self.name+')' if self.name else ''}",
            f"  {'='*50}",
            f"  Entry: {self.entry_price:.2f} → Now: {self.current_price:.2f} | +{self.holding_days}d",
            f"  Return: {self.return_pct:+.1%} | Peak: {self.peak_return:+.1%} | Giveback: {self.giveback:+.1%}",
            f"  MA20: {self.ma20:.2f} ({self.ma20_distance:+.1%}) | MA60: {self.ma60:.2f}",
            f"  Confirmed: {'Yes' if self.confirmed else 'No'} | Regime: {self.regime}",
            f"  Status: [{status}] {'Hold — trend intact' if status == 'OK' else 'Review — below MA20'}",
            f"  {'='*50}",
        ]
        return "\n".join(lines)


# ── Quick test ──
if __name__ == "__main__":
    # Position sizing demo
    sizer = PositionSizer()
    for regime in ["severe_bear", "bear", "sideways", "bull"]:
        plan = sizer.plan(regime, is_confirmed=True)
        print(f"{regime:<15s}: {plan.summary()}")
        print(f"  {MarketRegime.regime_advice(regime)}")
        print()

    # Exit dashboard demo
    dash = ExitDashboard(
        code="002463.SZ", name="沪电股份",
        entry_price=116, current_price=128, peak_price=133,
        ma20=118, ma60=97, holding_days=30,
        confirmed=True, regime="bull",
    )
    print(dash.dashboard())
