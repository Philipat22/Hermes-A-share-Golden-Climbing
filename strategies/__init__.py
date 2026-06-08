"""
Strategy Protocol + Factor Registry

To add a new strategy:
  1. Create strategies/your_strategy.py
  2. Define create_signal_filter(engine) -> callable
  3. Register with backtest results

Usage:
  python strategies/__init__.py          # compare all strategies
  python engine/runner.py --validate     # validate a strategy
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict
from enum import Enum
import numpy as np
import pandas as pd


class Status(Enum):
    DRAFT = "draft"          # Not yet validated
    VALIDATING = "validating"  # Currently running
    PASS = "pass"            # Passed all checks
    WARN = "warn"            # Passed with warnings
    FAIL = "fail"            # Failed validation
    DEAD = "dead"            # Killed (data says no)


@dataclass
class BacktestResult:
    """Stored validation results from Layer 0."""
    n_signals: int = 0
    median: float = 0.0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    left_tail_5: float = 0.0
    left_tail_1: float = 0.0
    null_median: float = 0.0
    median_diff: float = 0.0
    ks_pvalue: float = 1.0
    wf_stable: bool = False
    wf_range: float = 0.0
    yearly_medians: Dict[int, float] = field(default_factory=dict)
    avg_win: float = 0.15
    avg_loss: float = -0.15
    date_validated: str = ""
    notes: str = ""


@dataclass
class StrategyDef:
    """A strategy definition that can be plugged into the validation pipeline."""
    name: str
    description: str
    create_signal_filter: Callable
    status: Status = Status.DRAFT
    backtest: Optional[BacktestResult] = None
    tags: List[str] = field(default_factory=list)

    def set_backtest(self, result: BacktestResult):
        self.backtest = result
        if result.n_signals < 200:
            self.status = Status.FAIL
        elif not result.wf_stable:
            self.status = Status.WARN
        elif result.median_diff < 0.01:
            self.status = Status.FAIL
        else:
            self.status = Status.PASS

    @property
    def sharpe_approx(self) -> float:
        """Rough annualized Sharpe from backtest stats."""
        if self.backtest is None or self.backtest.n_signals < 200:
            return 0.0
        # Approximation: median / (interquartile range / 1.35) * sqrt(6.5)
        # Using left_tail_5 as a rough volatility proxy
        return 0.0  # needs proper std from Layer 0


# Registry
STRATEGIES: Dict[str, StrategyDef] = {}


def register(strategy: StrategyDef):
    STRATEGIES[strategy.name] = strategy
    return strategy


def list_strategies(verbose: bool = False):
    """List all registered strategies."""
    if verbose:
        print(f"\n{'='*90}")
        print(f"  FACTOR REGISTRY")
        print(f"{'='*90}")
        print(f"  {'Name':<20s} {'Status':<10s} {'N':>6s} {'Median':>8s} {'WR':>6s} "
              f"{'WF':>4s} {'Tags'}")
        print(f"  {'-'*20} {'-'*10} {'-'*6} {'-'*8} {'-'*6} {'-'*4} {'-'*20}")
        for name, s in sorted(STRATEGIES.items()):
            bt = s.backtest
            n = f"{bt.n_signals}" if bt else "-"
            med = f"{bt.median:+.1%}" if bt else "-"
            wr = f"{bt.win_rate:.0%}" if bt else "-"
            wf = "Y" if (bt and bt.wf_stable) else "N" if bt else "-"
            tags = ", ".join(s.tags[:3]) if s.tags else "-"
            print(f"  {name:<20s} {s.status.value:<10s} {n:>6s} {med:>8s} {wr:>6s} "
                  f"{wf:>4s} {tags}")
        print(f"{'='*90}")
    else:
        for name, s in STRATEGIES.items():
            bt_str = ""
            if s.backtest:
                bt_str = f" [{s.backtest.median:+.1%}, n={s.backtest.n_signals}]"
            print(f"  {name:<20s} {s.status.value:<10s}{bt_str} — {s.description}")


def compare() -> str:
    """Return a comparison table of all validated strategies."""
    validated = {n: s for n, s in STRATEGIES.items()
                 if s.backtest and s.backtest.n_signals >= 200}

    if not validated:
        return "No validated strategies to compare."

    lines = [
        f"\n{'='*80}",
        f"  STRATEGY COMPARISON",
        f"{'='*80}",
        f"  {'Name':<20s} {'N':>6s} {'Median':>8s} {'vsNull':>8s} "
        f"{'WR':>6s} {'WF':>4s} {'Tail5%':>8s}",
        f"  {'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*4} {'-'*8}",
    ]

    for name, s in sorted(validated.items(),
                           key=lambda x: -(x[1].backtest.median if x[1].backtest else -99)):
        bt = s.backtest
        lines.append(
            f"  {name:<20s} {bt.n_signals:>6d} {bt.median:>+7.1%} "
            f"{bt.median_diff:>+7.1%} {bt.win_rate:>5.0%} "
            f"{'Y' if bt.wf_stable else 'N':>4s} {bt.left_tail_5:>+7.1%}"
        )

    lines.append(f"{'='*80}")
    return "\n".join(lines)


# ── Auto-load all strategy modules ──
def load_all():
    """Import all strategy modules to register them. Call once at startup."""
    import importlib
    from pathlib import Path

    strat_dir = Path(__file__).parent
    for f in sorted(strat_dir.glob("*.py")):
        if f.name.startswith("_") or f.name == "__init__.py":
            continue
        module_name = f"strategies.{f.stem}"
        try:
            importlib.import_module(module_name)
        except Exception as e:
            pass


# ── Helper ──

def compute_dd_at(df: pd.DataFrame, dt: pd.Timestamp, lookback_days: int = 252) -> Optional[float]:
    idx = df.index.get_loc(dt)
    if isinstance(idx, slice):
        return None
    if isinstance(idx, np.ndarray):
        idx = idx[0]
    if idx < 60:
        return None
    lookback = df.iloc[max(0, idx - lookback_days):idx + 1]
    peak = lookback["close"].max()
    current = df.iloc[idx]["close"]
    if peak <= 0:
        return None
    return (current - peak) / peak


# ── CLI ──
if __name__ == "__main__":
    list_strategies(verbose=True)
    print(compare())
