"""
Opponent Engine — Identify who's on the other side of the trade.

Uses:
  - macro_margin.pkl: per-stock margin balance changes
  - prices_full.pkl: volume/price patterns
  - (future) 龙虎榜, 大宗交易, 股东人数

Output:
  OpponentProfile: who's selling, are they done, confidence level.
"""

import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

CACHE = Path("data/cache")


@dataclass
class OpponentProfile:
    code: str
    date: str
    opponent_type: str      # "panic_retail", "stop_loss_trader", "margin_call", "institution_exit", "unknown"
    confidence: float       # 0.0 - 1.0
    signals: dict           # individual signal values
    summary: str            # human-readable

    def is_actionable(self) -> bool:
        return self.opponent_type in ("panic_retail", "margin_call") and self.confidence >= 0.6


class OpponentEngine:
    """Analyze who's on the other side of a trade."""

    def __init__(self):
        self.margin_data = None   # per-stock margin
        self.margin_daily = None  # market-level margin
        self.prices = None
        self._loaded = False

    def load(self):
        print("Loading opponent engine data...")
        
        # Per-stock margin
        self.margin_data = pd.read_pickle(CACHE / "macro_margin.pkl")
        self.margin_data["trade_date"] = pd.to_datetime(self.margin_data["trade_date"])
        
        # Market-level margin
        self.margin_daily = pd.read_pickle(CACHE / "macro_margin_daily.pkl")
        self.margin_daily["trade_date"] = pd.to_datetime(self.margin_daily["trade_date"])
        self.margin_daily = self.margin_daily.set_index("trade_date").sort_index()
        
        # Prices
        with open(CACHE / "prices_full.pkl", "rb") as f:
            self.prices = pickle.load(f)
        for code, df in self.prices.items():
            if "trade_date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df.set_index("trade_date", inplace=True)

        self._loaded = True
        print(f"  Margin: {len(self.margin_data)} rows, Prices: {len(self.prices)} stocks")

    def analyze(self, code: str, date_str: str) -> OpponentProfile:
        """Analyze opponent for a single stock on a single date."""
        if not self._loaded:
            raise RuntimeError("Call load() first.")

        dt = pd.Timestamp(date_str)
        signals = {}
        clues = []

        # ── 1. Margin balance change (per-stock) ──
        stock_margin = self.margin_data[
            (self.margin_data["ts_code"] == code) &
            (self.margin_data["trade_date"] <= dt)
        ].sort_values("trade_date")

        margin_declining = False
        if len(stock_margin) >= 20:
            recent = stock_margin.tail(20)
            # 5-day margin change
            if len(recent) >= 6:
                margin_5d = recent.iloc[-1]["rzye"] - recent.iloc[-6]["rzye"]
                margin_5d_pct = margin_5d / recent.iloc[-6]["rzye"] if recent.iloc[-6]["rzye"] > 0 else 0
                signals["margin_5d_chg_pct"] = float(margin_5d_pct)
                if margin_5d_pct < -0.05:
                    margin_declining = True
                    clues.append(f"Margin -{abs(margin_5d_pct):.1%} in 5d (forced selling)")
                elif margin_5d_pct < -0.02:
                    clues.append(f"Margin -{abs(margin_5d_pct):.1%} in 5d (mild decline)")
                else:
                    clues.append(f"Margin stable ({margin_5d_pct:+.1%})")

        # ── 2. Volume analysis ──
        vol_shrunk = False
        if code in self.prices and dt in self.prices[code].index:
            pdf = self.prices[code]
            idx = pdf.index.get_loc(dt)
            if isinstance(idx, np.ndarray):
                idx = idx[0]

            if idx >= 20:
                vol_5d = pdf.iloc[max(0, idx - 4):idx + 1]["vol"].mean()
                vol_20d = pdf.iloc[max(0, idx - 19):idx + 1]["vol"].mean()
                if vol_20d > 0:
                    vr = vol_5d / vol_20d
                    signals["volume_ratio"] = float(vr)
                    if vr < 0.5:
                        vol_shrunk = True
                        clues.append(f"Volume shrunk to {vr:.2f}x (sellers exhausted)")
                    elif vr < 0.8:
                        clues.append(f"Volume {vr:.2f}x (declining)")
                    else:
                        clues.append(f"Volume {vr:.2f}x (normal/elevated)")

            # ── 3. Price action: DD depth ──
            if idx >= 60:
                lookback = pdf.iloc[max(0, idx - 252):idx + 1]
                peak = lookback["close"].max()
                current = pdf.iloc[idx]["close"]
                if peak > 0:
                    dd = (current - peak) / peak
                    signals["dd"] = float(dd)

        # ── 4. Market-level margin direction ──
        market_margin_ok = False
        if dt in self.margin_daily.index:
            mg_idx = self.margin_daily.index.get_loc(dt)
            if mg_idx >= 10:
                mg_recent = self.margin_daily.iloc[mg_idx - 9:mg_idx + 1]["rzye"]
                mg_10d_chg = (mg_recent.iloc[-1] - mg_recent.iloc[0]) / mg_recent.iloc[0]
                signals["market_margin_10d"] = float(mg_10d_chg)
                if mg_10d_chg < -0.02:
                    market_margin_ok = True
                    clues.append("Market margin declining (forced selling environment)")

        # ── Classification ──
        dd = signals.get("dd", 0)
        
        if margin_declining and vol_shrunk and dd < -0.25:
            opponent = "panic_retail"
            confidence = 0.75
            summary = "Retail panic selling + margin forced liquidation. Volume exhausted."
        elif margin_declining and dd < -0.25:
            opponent = "margin_call"
            confidence = 0.60
            summary = "Margin forced selling likely. Volume not fully exhausted yet."
        elif margin_declining:
            opponent = "stop_loss_trader"
            confidence = 0.40
            summary = "Leverage exiting but not deep enough for panic classification."
        elif dd > -0.10:
            opponent = "unknown"
            confidence = 0.20
            summary = "Not in distress. No clear opponent signal."
        else:
            opponent = "institution_exit"
            confidence = 0.30
            summary = "Declining without margin pressure — possible institutional selling."

        return OpponentProfile(
            code=code,
            date=date_str,
            opponent_type=opponent,
            confidence=confidence,
            signals=signals,
            summary=f"{summary} Clues: {'; '.join(clues)}",
        )

    def vote(self, code: str, date_str: str) -> dict:
        """
        Produce a vote compatible with arbitrator.
        Returns {'verdict': 'buy'/'sell'/'hold', 'confidence': 0-1, 'reason': str}
        """
        profile = self.analyze(code, date_str)
        
        from engine.arbitrator import Verdict
        if profile.is_actionable():
            return {
                "verdict": Verdict.BUY,
                "confidence": profile.confidence,
                "reason": profile.summary,
            }
        else:
            return {
                "verdict": Verdict.HOLD,
                "confidence": 0.30,
                "reason": profile.summary,
            }


# Quick test
if __name__ == "__main__":
    engine = OpponentEngine()
    engine.load()
    
    # Test: a known deep DD stock
    profile = engine.analyze("300430.SZ", "2026-04-27")
    print(f"\n{profile.code} on {profile.date}:")
    print(f"  Opponent: {profile.opponent_type} (conf={profile.confidence:.0%})")
    print(f"  Summary: {profile.summary}")
    print(f"  Signals: {profile.signals}")
    print(f"  Actionable: {profile.is_actionable()}")
