"""
NorthMargin Strategy — Market-level north flow + margin signal × stock DD.

Core logic:
  Market: north_flow 5d > 0 AND margin balance 5d declining
  Stock:  DD from 1yr peak >= threshold (default -25%)
  When both conditions met → signal fires.

Backtest result (Layer 0, n=50k):
  Median: +2.64%  |  WR: 56.9%  |  Left tail 5%: -19.9%
  On NM days only: median +5.70%, WR 66.5%
"""

from pathlib import Path
from typing import Optional
import pickle
import numpy as np
import pandas as pd

from strategies import StrategyDef, register, BacktestResult, Status, compute_dd_at

CACHE = Path("data/cache")


class NorthMarginStrategy:
    """NorthMargin signal filter for Layer 0 backtesting."""

    def __init__(self, engine,
                 dd_threshold: float = -0.25,
                 north_window: int = 5,
                 margin_window: int = 5):
        self.engine = engine
        self.dd_threshold = dd_threshold
        self.north_window = north_window
        self.margin_window = margin_window

        # Pre-compute market signal dates
        self._signal_dates = self._build_market_signals()

    def _build_market_signals(self) -> set:
        nf = pd.read_pickle(CACHE / "macro_north_flow.pkl")
        mg = pd.read_pickle(CACHE / "macro_margin_daily.pkl")
        nf["trade_date"] = pd.to_datetime(nf["trade_date"])
        mg["trade_date"] = pd.to_datetime(mg["trade_date"])
        nf = nf.sort_values("trade_date")
        mg = mg.sort_values("trade_date")
        nf["north_5d"] = nf["north_money"].rolling(self.north_window).sum()
        mg["margin_5d_chg"] = mg["rzye"].diff(self.margin_window)
        merged = nf.merge(mg[["trade_date", "margin_5d_chg"]], on="trade_date", how="inner")
        mask = (merged["north_5d"] > 0) & (merged["margin_5d_chg"] < 0)
        return set(merged.loc[mask, "trade_date"].dt.strftime("%Y-%m-%d"))

    def __call__(self, df, code, date_str):
        """Layer 0 signal filter interface."""
        if date_str not in self._signal_dates:
            return False
        dt = pd.Timestamp(date_str)
        if dt not in df.index:
            return False
        dd = compute_dd_at(df, dt)
        if dd is None:
            return False
        return dd <= self.dd_threshold


def create_signal_filter(engine):
    """Factory function for StrategyDef protocol."""
    return NorthMarginStrategy(engine)

# ── Register ──

NORTH_MARGIN = StrategyDef(
    name="north_margin",
    description="North flow inflow + margin decline + DD trigger",
    create_signal_filter=create_signal_filter,
    status=Status.WARN,
    tags=["fund_flow", "mean_reversion", "market_timing"],
    backtest=BacktestResult(
        n_signals=50000,
        median=0.0264,
        win_rate=0.569,
        profit_loss_ratio=1.83,
        left_tail_5=-0.199,
        left_tail_1=-0.327,
        null_median=-0.0044,
        median_diff=0.0308,
        ks_pvalue=0.0,
        wf_stable=False,
        wf_range=0.1152,
        yearly_medians={
            2019: 0.050, 2020: 0.011, 2021: 0.026, 2022: 0.042,
            2023: -0.046, 2024: 0.069, 2025: 0.030, 2026: -0.071,
        },
        avg_win=0.18,
        avg_loss=-0.15,
        date_validated="2026-06-02",
        notes="Best performer. WF unstable. Only trade on NM active days (+5.7% vs +0.96%). "
              "Dry spells: market signal only on 34% of days.",
    ),
)

register(NORTH_MARGIN)
