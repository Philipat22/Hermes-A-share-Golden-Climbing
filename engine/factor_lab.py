"""
Factor Research Lab — Batch factor validation pipeline.

Usage:
  python engine/factor_lab.py --all           # run all registered factors
  python engine/factor_lab.py --factor F1     # run single factor
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from engine.layer0_actuarial import ActuarialEngine

# Import strategies to get the registry
import strategies
strategies.load_all()
from strategies import (
    StrategyDef, BacktestResult, Status, STRATEGIES, register, compute_dd_at
)


@dataclass
class FactorReport:
    """Standardized factor report card."""
    name: str
    status: str  # PASS / WARN / FAIL
    n_signals: int = 0
    median: float = 0.0
    win_rate: float = 0.0
    median_diff: float = 0.0  # vs null
    wf_stable: bool = False
    wf_range: float = 0.0
    left_tail_5: float = 0.0
    best_regime: str = ""
    best_regime_median: float = 0.0
    attribution: dict = field(default_factory=dict)
    cost_adjusted_median: float = 0.0
    elapsed_seconds: float = 0.0

    def print_card(self):
        """Print a one-page factor report card."""
        print(f"\n{'='*70}")
        print(f"  FACTOR: {self.name}")
        print(f"  Status: {self.status}")
        print(f"{'='*70}")
        print(f"  Signals:   {self.n_signals:>8d}")
        print(f"  Median:    {self.median:>+8.1%}  (vs null: {self.median_diff:>+.1%})")
        print(f"  Win Rate:  {self.win_rate:>8.0%}")
        print(f"  Left 5%:   {self.left_tail_5:>+8.1%}")
        print(f"  WF Stable: {'YES' if self.wf_stable else 'NO':>8s}  (range: {self.wf_range:.1%})")
        print(f"  Best in:   {self.best_regime:<15s} ({self.best_regime_median:+.1%})")
        if self.attribution and 'alpha' in self.attribution:
            a = self.attribution
            print(f"  Alpha:     {a['alpha']:>+8.2%}  Beta: {a['beta']:.2f}  "
                  f"R2: {a['r_squared']:.1%}")
        print(f"  Net (cost):{self.cost_adjusted_median:>+8.1%}")
        print(f"  Time:      {self.elapsed_seconds:>8.1f}s")


class FactorLab:
    """Batch factor research lab."""

    def __init__(self, sample_stocks: int = 3000, max_signals: int = 20000,
                 forward_days: int = 40):
        self.sample_stocks = sample_stocks
        self.max_signals = max_signals
        self.forward_days = forward_days
        self.engine = None

    def _load_engine(self):
        if self.engine is None:
            print(f"Loading engine (forward={self.forward_days}d)...")
            self.engine = ActuarialEngine(forward_days=self.forward_days).load_data()

    def validate_factor(self, name: str, description: str,
                        signal_filter: Callable,
                        tags: List[str] = None) -> FactorReport:
        """Run a single factor through full validation."""
        self._load_engine()
        t0 = time.time()
        report = FactorReport(name=name, status="FAIL")

        # ── Scan ──
        df = self.engine.find_signals(
            signal_filter,
            sample_stocks=self.sample_stocks,
            max_signals=self.max_signals,
        )

        if len(df) < 200:
            report.n_signals = len(df)
            report.elapsed_seconds = time.time() - t0
            return report

        # ── Layer 0 ──
        ar = self.engine.analyze(df, signal_name=name)
        report.n_signals = ar.n_signals
        report.median = ar.median
        report.win_rate = ar.win_rate
        report.median_diff = ar.median_diff
        report.wf_stable = ar.wf_stable
        report.wf_range = ar.wf_range
        report.left_tail_5 = ar.left_tail_5

        # ── Market state ──
        if 'regime' in df.columns:
            cond = self.engine.analyze_conditional(df, 'regime', signal_name=name)
            best = max(cond.values(), key=lambda r: r.median, default=None)
            if best:
                report.best_regime = best.signal_name.split('=')[-1]
                report.best_regime_median = best.median

        # ── Attribution ──
        try:
            attr = self.engine.analyze_attribution(df, signal_name=name)
            report.attribution = attr
        except Exception:
            pass

        # ── Cost ──
        cost = 0.0005 + 0.00025 * 2 + 0.001
        report.cost_adjusted_median = ar.median - cost

        # ── Status ──
        if ar.n_signals < 200:
            report.status = "FAIL"
        elif ar.wf_stable and ar.median_diff > 0.01 and ar.median > 0:
            report.status = "PASS"
        elif ar.median > 0 and ar.median_diff > 0:
            report.status = "WARN"
        else:
            report.status = "FAIL"

        # ── Register ──
        strat = StrategyDef(
            name=name,
            description=description,
            create_signal_filter=lambda e: signal_filter,
            status=Status.WARN if report.status == "WARN" else (
                Status.PASS if report.status == "PASS" else Status.FAIL
            ),
            tags=tags or [],
            backtest=BacktestResult(
                n_signals=ar.n_signals,
                median=ar.median,
                win_rate=ar.win_rate,
                profit_loss_ratio=ar.profit_loss_ratio,
                left_tail_5=ar.left_tail_5,
                left_tail_1=ar.left_tail_1,
                null_median=ar.null_median,
                median_diff=ar.median_diff,
                wf_stable=ar.wf_stable,
                wf_range=ar.wf_range,
                yearly_medians=ar.yearly_medians,
                date_validated=pd.Timestamp.now().strftime("%Y-%m-%d"),
            ),
        )
        register(strat)

        report.elapsed_seconds = time.time() - t0
        return report

    def compare_all(self) -> str:
        """Return a comparison of all factors in registry."""
        validated = {n: s for n, s in STRATEGIES.items()
                     if s.backtest and s.backtest.n_signals >= 200}
        if not validated:
            return "No validated factors."

        lines = [
            f"\n{'='*80}",
            f"  FACTOR COMPARISON",
            f"{'='*80}",
            f"  {'Factor':<25s} {'N':>6s} {'Median':>8s} {'vsNull':>8s} "
            f"{'WR':>6s} {'WF':>4s} {'Status':>8s}",
            f"  {'-'*25} {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*4} {'-'*8}",
        ]

        for name, s in sorted(validated.items(),
                               key=lambda x: -(x[1].backtest.median if x[1].backtest else 0)):
            bt = s.backtest
            lines.append(
                f"  {name:<25s} {bt.n_signals:>6d} {bt.median:>+7.1%} "
                f"{bt.median_diff:>+7.1%} {bt.win_rate:>5.0%} "
                f"{'Y' if bt.wf_stable else 'N':>4s} {s.status.value:>8s}"
            )

        lines.append(f"{'='*80}")
        return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument("--max", type=int, default=20000)
    args = parser.parse_args()

    lab = FactorLab(sample_stocks=args.sample, max_signals=args.max)

    if args.compare:
        print(lab.compare_all())
    elif args.all:
        print("Run factors via: python engine/factor_lab.py --compare")
        print("Or run individual factors from their strategy files.")
