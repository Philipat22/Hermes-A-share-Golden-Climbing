"""
Generic Daily Scanner — Strategy-agnostic.

Loads data once. Accepts any StrategyDef. Scans all stocks on a given date.
"""

import pickle
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd

CACHE = Path("data/cache")


@dataclass
class Candidate:
    code: str
    date: str
    close: float
    dd: Optional[float] = None
    regime: str = "unknown"


class Scanner:
    """Generic daily scanner. Pass a strategy to scan()."""

    def __init__(self, min_price: float = 2.0):
        self.min_price = min_price
        self.prices = {}
        self.csi300 = None
        self._all_stocks = []
        self._loaded = False

    def load(self):
        t0 = time.time()
        print(f"Loading scanner data...")

        with open(CACHE / "prices_full.pkl", "rb") as f:
            self.prices = pickle.load(f)
        for code, df in self.prices.items():
            if "trade_date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df.set_index("trade_date", inplace=True)
        self._all_stocks = sorted(self.prices.keys())

        cs = pd.read_pickle(CACHE / "csi300.pkl")
        reg = pd.read_pickle(CACHE / "csi300_regime.pkl")
        cs["trade_date"] = pd.to_datetime(cs["trade_date"])
        reg["trade_date"] = pd.to_datetime(reg["trade_date"])
        self.csi300 = cs.merge(
            reg[["trade_date", "regime", "ma20", "ma60"]],
            on="trade_date", how="left", suffixes=("", "_r")
        )
        self.csi300.set_index("trade_date", inplace=True)

        self._loaded = True
        print(f"  {len(self._all_stocks)} stocks, {time.time()-t0:.1f}s")

    def regime_at(self, date_str: str) -> str:
        dt = pd.Timestamp(date_str)
        if dt in self.csi300.index:
            row = self.csi300.loc[dt]
            if "ma60" in row.index and pd.notna(row.get("ma60")) and row.get("ma60", 0) > 0:
                return "bull" if row["close"] > row["ma60"] else "bear"
            idx = self.csi300.index.get_loc(dt)
            if idx >= 60:
                ma60 = self.csi300["close"].iloc[idx - 59:idx + 1].mean()
                return "bull" if row["close"] > ma60 else "bear"
        return "unknown"

    def compute_dd(self, df, dt):
        idx = df.index.get_loc(dt)
        if isinstance(idx, slice):
            return None
        if isinstance(idx, np.ndarray):
            idx = idx[0]
        if idx < 60:
            return None
        lookback = df.iloc[max(0, idx - 252):idx + 1]
        peak = lookback["close"].max()
        current = df.iloc[idx]["close"]
        if peak <= 0:
            return None
        return (current - peak) / peak

    def scan(self, date_str: str, signal_filter) -> List[Candidate]:
        """
        Scan all stocks on a single date using the given signal filter.

        signal_filter(df, code, date_str) -> bool
        """
        if not self._loaded:
            raise RuntimeError("Call load() first.")

        dt = pd.Timestamp(date_str)
        if dt not in self.csi300.index:
            return []

        regime = self.regime_at(date_str)
        candidates = []

        for code in self._all_stocks:
            df = self.prices.get(code)
            if df is None or dt not in df.index or len(df) < 120:
                continue

            close = df.loc[dt, "close"]
            if close < self.min_price:
                continue

            try:
                if signal_filter(df, code, date_str):
                    dd = self.compute_dd(df, dt)
                    candidates.append(Candidate(
                        code=code,
                        date=date_str,
                        close=float(close),
                        dd=float(dd) if dd is not None else None,
                        regime=regime,
                    ))
            except Exception:
                continue

        candidates.sort(key=lambda c: c.dd if c.dd is not None else 0)
        return candidates
