"""
Unified Screening Layer — Run all active strategies, cross-check, grade, remove fraud.

Output: tiered daily candidate pool with quality flags.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Set, Dict
import pickle
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import strategies
strategies.load_all()
from strategies import STRATEGIES
from engine.scanner import Scanner

CACHE = Path("data/cache")

# ── Quality Scoring (DeepSeek framework) ──

class QualityScorer:
    """4-dimension fundamental quality score (0-4)."""
    
    def __init__(self):
        fin = pd.read_pickle(CACHE / "financials.pkl")
        fin["end_date"] = pd.to_datetime(fin["end_date"])
        self.fin = fin.sort_values(["ts_code", "end_date"])
        self._cache = {}

    def score(self, code: str, date_str: str) -> int:
        """Return 0-4 quality score."""
        key = (code, date_str)
        if key in self._cache:
            return self._cache[key]

        sf = self.fin[self.fin["ts_code"] == code].sort_values("end_date")
        dt = pd.Timestamp(date_str)
        valid = sf[sf["end_date"] + pd.Timedelta(days=45) <= dt]
        if len(valid) == 0:
            return 0
        r = valid.iloc[-1]

        s = 0
        # 1. Revenue acceleration: last 3Q rev_yoy increasing
        if len(valid) >= 4:
            r0 = valid.iloc[-1].get("or_yoy")
            r1 = valid.iloc[-2].get("or_yoy")
            r2 = valid.iloc[-3].get("or_yoy")
            if pd.notna(r0) and pd.notna(r1) and pd.notna(r2) and r0 > r1 > r2:
                s += 1

        # 2. Margin quality: net margin > 10% and improving
        margin = r.get("netprofit_margin")
        if pd.notna(margin) and margin > 10:
            if len(valid) >= 2:
                m0 = valid.iloc[-1].get("netprofit_margin")
                m1 = valid.iloc[-2].get("netprofit_margin")
                if pd.notna(m0) and pd.notna(m1) and m0 >= m1:
                    s += 1

        # 3. Cash flow quality: OCF/EPS > 0.7
        ocf = r.get("ocfps")
        eps_val = r.get("eps")
        if pd.notna(ocf) and pd.notna(eps_val) and eps_val > 0 and ocf / eps_val > 0.7:
            s += 1

        # 4. ROE > 8%
        roe = r.get("roe")
        if pd.notna(roe) and roe > 8:
            s += 1

        self._cache[key] = s
        return s


# ── Composite ranking ──

def composite_rank(candidates, quality_scorer: QualityScorer, date_str: str):
    """Rank candidates by quality(0.6) + DD-shallow(0.4)."""
    if not candidates:
        return candidates

    # Compute scores
    dd_vals = [abs(c.dd) for c in candidates]
    if max(dd_vals) == min(dd_vals):
        dd_ranks = [0.5] * len(candidates)
    else:
        dd_ranks = [(1 - (abs(c.dd) - min(dd_vals)) / (max(dd_vals) - min(dd_vals))) for c in candidates]
        # shallow DD = high rank

    for i, c in enumerate(candidates):
        q = quality_scorer.score(c.code, date_str)
        c.quality_score = q
        c.composite_score = q / 4 * 0.6 + dd_ranks[i] * 0.4
        # Quality tier
        if q >= 3:
            c.quality_tier = "A"
        elif q >= 2:
            c.quality_tier = "B"
        else:
            c.quality_tier = "C"

    candidates.sort(key=lambda c: -c.composite_score)
    return candidates


@dataclass
class Candidate:
    code: str
    close: float
    dd: float
    regime: str
    strategies_triggered: List[str]
    tier: str
    flags: List[str] = field(default_factory=list)
    quality_score: int = 0
    quality_tier: str = ""
    composite_score: float = 0.0


class Screener:
    """Unified screening: all strategies → cross-check → grade → fraud filter."""

    def __init__(self, active_strategies: List[str] = None):
        # Default: all non-draft strategies
        if active_strategies is None:
            active_strategies = [
                name for name, s in STRATEGIES.items()
                if s.status.value in ("warn", "pass")
            ]
        self.active = active_strategies
        self.scanner = Scanner()
        self.scanner.load()
        self._fraud_cache = None

    def _load_fraud_filter(self):
        """Build CF/NP fraud flag: stocks in bottom quintile of cash flow quality."""
        if self._fraud_cache is not None:
            return self._fraud_cache

        fin = pd.read_pickle(CACHE / "financials.pkl")
        fin["end_date"] = pd.to_datetime(fin["end_date"])
        fin = fin.sort_values(["ts_code", "end_date"])

        # Compute CF/NP ratio per quarter
        cf_ratios = []
        for _, row in fin.iterrows():
            ocf = row.get("ocfps"); eps_val = row.get("eps")
            if pd.notna(ocf) and pd.notna(eps_val) and eps_val > 0:
                cf_ratios.append((row["ts_code"], row["end_date"], ocf / eps_val))
        cf_df = pd.DataFrame(cf_ratios, columns=["ts_code", "end_date", "cf_ratio"])

        # Bottom quintile threshold
        q20 = cf_df["cf_ratio"].quantile(0.20)
        print(f"  Fraud threshold (bottom 20% CF/NP): {q20:.2f}")

        # Build set of stocks that are in bottom quintile in latest quarter
        latest = cf_df.sort_values("end_date").groupby("ts_code").last()
        fraud_stocks = set(latest[latest["cf_ratio"] < q20].index)

        self._fraud_cache = fraud_stocks
        return fraud_stocks

    def screen(self, date_str: str) -> List[Candidate]:
        """Run all active strategies, cross-check, grade, flag."""
        if not self._loaded():
            self.scanner.load()

        fraud = self._load_fraud_filter()

        # ── Run each strategy ──
        strategy_codes: Dict[str, Set[str]] = {}

        class EngineAdapter:
            def __init__(self, s): self.csi300 = s.csi300; self.prices = s.prices
            def _regime_at(self, d): return self.scanner.regime_at(d)

        engine_adapt = EngineAdapter(self.scanner)

        for sname in self.active:
            strat = STRATEGIES.get(sname)
            if strat is None:
                continue
            try:
                sf = strat.create_signal_filter(engine_adapt)
                cands = self.scanner.scan(date_str, sf)
                strategy_codes[sname] = {c.code for c in cands}
            except Exception as e:
                strategy_codes[sname] = set()

        # ── Cross-check: count how many strategies trigger per stock ──
        stock_triggers: Dict[str, List[str]] = {}
        for sname, codes in strategy_codes.items():
            for code in codes:
                if code not in stock_triggers:
                    stock_triggers[code] = []
                stock_triggers[code].append(sname)

        # ── Grade ──
        candidates = []
        regime = self.scanner.regime_at(date_str)

        for code, trigger_list in stock_triggers.items():
            n = len(trigger_list)
            has_mean_rev = any("mean_rev_accel" in s for s in trigger_list)
            has_nm = any("north_margin" in s for s in trigger_list)

            # Tier classification
            if n >= 2 and has_mean_rev:
                tier = "A"   # double confirm with primary strategy
            elif has_mean_rev:
                tier = "B"   # primary strategy only
            elif n >= 2:
                tier = "B"   # two aux strategies
            else:
                tier = "C"   # single aux strategy
            # A: mean_rev_accel + another
            # B: mean_rev_accel alone OR two aux strategies
            # C: single aux strategy

            # Get price/DD
            df = self.scanner.prices.get(code)
            close = 0; dd = 0
            dt = pd.Timestamp(date_str)
            if df is not None and dt in df.index:
                close = float(df.loc[dt, "close"])
                dd = self.scanner.compute_dd(df, dt) or 0

            # Fraud flag
            flags = []
            if code in fraud:
                flags.append("FRAUD_RISK: CF/NP bottom 20%")

            candidates.append(Candidate(
                code=code,
                close=close,
                dd=float(dd) if dd else 0,
                regime=regime,
                strategies_triggered=trigger_list,
                tier=tier,
                flags=flags,
            ))

        # Sort by composite: quality(0.6) + DD-shallow(0.4)
        qs = QualityScorer()
        candidates = composite_rank(candidates, qs, date_str)
        
        return candidates

    def _loaded(self):
        return self.scanner._loaded

    def report(self, date_str: str) -> str:
        cands = self.screen(date_str)

        if not cands:
            return f"=== Daily Screen - {date_str} ===\n  No candidates."

        tiers = {"A": [], "B": [], "C": []}
        frauds = []
        for c in cands:
            tiers[c.tier].append(c)
            if c.flags:
                frauds.append(c)

        qa = sum(1 for c in cands if c.quality_tier == "A")
        qb = sum(1 for c in cands if c.quality_tier == "B")
        qc = sum(1 for c in cands if c.quality_tier == "C")

        lines = [
            f"=== DAILY CANDIDATE POOL - {date_str} ===",
            f"  Regime: {cands[0].regime}",
            f"  Total: {len(cands)} (Q-A:{qa} Q-B:{qb} Q-C:{qc})",
            f"  Fraud warnings: {len(frauds)}",
            "",
        ]

        for tier_label in ["A", "B", "C"]:
            pool = [c for c in cands if c.quality_tier == tier_label]
            if not pool:
                continue
            stars = "***" if tier_label == "A" else "**" if tier_label == "B" else "*"
            lines.append(f"  [{stars}] QUALITY {tier_label} ({len(pool)} stocks)")
            lines.append(f"  {'Code':<12s} {'Price':>8s} {'DD':>8s} {'Q':>3s} {'Score':>6s} {'Triggers'}")
            lines.append(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*3} {'-'*6} {'-'*30}")
            for c in pool[:10]:
                tr = ",".join(c.strategies_triggered)
                warn = " [FRAUD]" if c.flags else ""
                lines.append(f"  {c.code:<12s} {c.close:>8.2f} {c.dd:>+7.1%} "
                           f"{c.quality_score:>2d}/4 {c.composite_score:>5.3f} {tr:<30s}{warn}")
            lines.append("")

        if frauds:
            lines.append(f"  WARNING: {len(frauds)} stocks flagged for CF/NP fraud risk")
            lines.append(f"  Flagged: {', '.join(c.code for c in frauds[:5])}...")

        return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="2026-04-27")
    args = parser.parse_args()

    screener = Screener()
    print(screener.report(args.date))
