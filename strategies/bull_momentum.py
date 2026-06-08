"""
Bull Momentum Strategy — Trend following for bull/recovery markets.

Signal: price > MA20 + MA60 + momentum + volume expansion.
Only activates in bull/recovery regime (opposite of NorthMargin).

Design: complements NorthMargin. When NorthMargin hibernates (bull),
this strategy wakes up. Together they cover both market directions.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from strategies import StrategyDef, register, BacktestResult, Status, compute_dd_at


class BullMomentumFilter:
    """Trend-following signal for bull/recovery markets."""

    def __init__(self, engine,
                 min_momentum: float = 0.05,     # 20-day return > 5%
                 vol_expand_ratio: float = 1.2,   # volume > 1.2x 20d avg
                 allowed_regimes: set = None):
        self.engine = engine
        self.min_momentum = min_momentum
        self.vol_expand_ratio = vol_expand_ratio
        self.allowed_regimes = allowed_regimes or {"bull", "recovery"}

    def __call__(self, df, code, date_str):
        dt = pd.Timestamp(date_str)
        if dt not in df.index:
            return False

        idx = df.index.get_loc(dt)
        if isinstance(idx, np.ndarray):
            idx = idx[0]
        if idx < 60:
            return False

        # Regime check: only bull/recovery
        regime = self.engine._regime_at(date_str)
        if regime not in self.allowed_regimes:
            return False

        # Price check: above MA20 and MA60
        close = df.iloc[idx]["close"]
        ma20 = df.iloc[max(0, idx - 19):idx + 1]["close"].mean()
        ma60 = df.iloc[max(0, idx - 59):idx + 1]["close"].mean()
        if close <= ma20 or close <= ma60:
            return False

        # Momentum: 20-day return
        close_20d = df.iloc[max(0, idx - 19)]["close"]
        momentum = (close - close_20d) / close_20d
        if momentum < self.min_momentum:
            return False

        # Volume: expanding (institutional participation)
        vol_5d = df.iloc[max(0, idx - 4):idx + 1]["vol"].mean()
        vol_20d = df.iloc[max(0, idx - 19):idx + 1]["vol"].mean()
        if vol_20d <= 0 or vol_5d / vol_20d < self.vol_expand_ratio:
            return False

        # Filter: not ST, reasonable price
        if close < 3.0:
            return False

        return True


def create_signal_filter(engine):
    return BullMomentumFilter(engine)


# Register with placeholder backtest (will be filled by runner)
BULL_MOMENTUM = StrategyDef(
    name="bull_momentum",
    description="Trend following: price>MA20+MA60 + momentum>5% + volume expanding. Bull/recovery only.",
    create_signal_filter=create_signal_filter,
    status=Status.DRAFT,
    tags=["momentum", "trend_following", "bull_market"],
)

register(BULL_MOMENTUM)
