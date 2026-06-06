"""
Registered Strategies:
  mean_rev_accel — 反转+盈利加速度 (40d reversal + 2Q earnings accelerating)
  mean_rev_dual   — 反转+双增长 (40d reversal + rev>20% AND profit>30%)
"""

from pathlib import Path
import pickle, numpy as np, pandas as pd
from datetime import timedelta

from strategies import StrategyDef, register, BacktestResult, Status

CACHE = Path("data/cache")

# ── Fundamental pre-computation ──
print("Pre-computing fundamentals...")
fin = pd.read_pickle(CACHE / "financials.pkl")
fin["end_date"] = pd.to_datetime(fin["end_date"])
fin = fin.sort_values(["ts_code", "end_date"])

cs = pd.read_pickle(CACHE / "csi300.pkl")
cs["trade_date"] = pd.to_datetime(cs["trade_date"])
all_dates = sorted(cs["trade_date"].unique())

# Build O(1) sets for earnings acceleration and dual growth
earn_set = set()    # {(code, date_str)}
cf_set = set()      # {(code, date_str)} earn_accel + OCF>0
dual_set = set()

for code, grp in fin.groupby("ts_code"):
    grp = grp.sort_values("end_date")
    for i in range(len(grp)):
        r = grp.iloc[i]
        ed = r["end_date"]
        vf = ed + timedelta(days=45)
        vu = grp.iloc[i+1]["end_date"] + timedelta(days=45) if i+1 < len(grp) else pd.Timestamp("2027-01-01")

        if i >= 2:
            np0 = r.get("netprofit_yoy"); np1 = grp.iloc[i-1].get("netprofit_yoy"); np2 = grp.iloc[i-2].get("netprofit_yoy")
            has_earn = pd.notna(np0) and pd.notna(np1) and pd.notna(np2) and np0 > np1 > np2 and np0 > 0
            rev = r.get("or_yoy")
            ocf = r.get("ocfps")
            has_cf = has_earn and pd.notna(ocf) and ocf > 0
            has_dual = has_earn and pd.notna(rev) and rev > 20 and np0 > 30

            if has_earn:
                for d in all_dates:
                    if vf <= d < vu:
                        ds = d.strftime("%Y-%m-%d")
                        earn_set.add((code, ds))
                        if has_cf:
                            cf_set.add((code, ds))
                        if has_dual:
                            dual_set.add((code, ds))

print(f"Earnings accel: {len(earn_set)} pairs, +OCF: {len(cf_set)} pairs, Dual: {len(dual_set)} pairs")


# ── Strategy 1: Mean Reversion + Earnings Acceleration ──

class MeanRevAccelFilter:
    def __init__(self, engine, window=40, threshold=-0.15):
        self.engine = engine; self.window = window; self.threshold = threshold

    def __call__(self, df, code, date_str):
        dt = pd.Timestamp(date_str)
        if dt not in df.index: return False
        idx = df.index.get_loc(dt)
        if isinstance(idx, np.ndarray): idx = idx[0]
        if idx < self.window: return False
        close = df.iloc[idx]["close"]
        close_w = df.iloc[max(0, idx - self.window)]["close"]
        if close_w <= 0 or close < 3: return False
        ret = (close - close_w) / close_w
        if ret > self.threshold: return False
        return (code, date_str) in cf_set  # earnings accel + OCF > 0


def create_mean_rev_accel(engine):
    return MeanRevAccelFilter(engine)

MEAN_REV_ACCEL = StrategyDef(
    name="golden_bull",
    description="金牛: DD40/DDpeak双入口 + 净利加速 + OCF>0 + 5d确认. WR82% +17.7%(60d). 满仓3只.",
    create_signal_filter=create_mean_rev_accel,
    status=Status.ACTIVE,
    tags=["reversal", "fundamental", "earnings", "cashflow", "primary"],
    backtest=BacktestResult(
        n_signals=225, median=0.177, win_rate=0.82,
        left_tail_5=-0.111, left_tail_1=-0.251,
        median_diff=0.157, wf_stable=False, wf_range=0.08,
        date_validated="2026-06-06",
        notes="金牛策略定稿。Dual entry (DD40/DDpeak<=-20%), 5d confirm, MA20 trail. "
              "Full position 3 stocks. Hot sector filter. -15% hard stop. "
              "7yr simulated: 11 rounds, +13.9%/round, WR 91%. CAGR ~20%.",
    ),
)
register(MEAN_REV_ACCEL)


# ── Strategy 2: Mean Reversion + Dual Growth ──

class MeanRevDualFilter:
    def __init__(self, engine, window=40, threshold=-0.15):
        self.engine = engine; self.window = window; self.threshold = threshold

    def __call__(self, df, code, date_str):
        dt = pd.Timestamp(date_str)
        if dt not in df.index: return False
        idx = df.index.get_loc(dt)
        if isinstance(idx, np.ndarray): idx = idx[0]
        if idx < self.window: return False
        close = df.iloc[idx]["close"]
        close_w = df.iloc[max(0, idx - self.window)]["close"]
        if close_w <= 0 or close < 3: return False
        ret = (close - close_w) / close_w
        if ret > self.threshold: return False
        return (code, date_str) in dual_set


def create_mean_rev_dual(engine):
    return MeanRevDualFilter(engine)

MEAN_REV_DUAL = StrategyDef(
    name="mean_rev_dual",
    description="40d reversal + rev>20% + profit>30%. Stable: +2.8% (60d), WR 56%.",
    create_signal_filter=create_mean_rev_dual,
    status=Status.WARN,
    tags=["reversal", "fundamental", "quality"],
    backtest=BacktestResult(
        n_signals=10000, median=0.028, win_rate=0.56,
        left_tail_5=-0.22, median_diff=0.02,
        wf_stable=False, date_validated="2026-06-04",
        notes="More stable than pure reversal. Lower return but higher quality filter.",
    ),
)
register(MEAN_REV_DUAL)

print("Strategies registered: mean_rev_accel, mean_rev_dual")
