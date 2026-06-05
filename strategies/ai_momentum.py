"""
AI Momentum Strategy — Chase strongest AI hardware stocks.

Entry: in AI chain + top momentum within AI universe + above MA20 + volume OK.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from strategies import StrategyDef, register, BacktestResult, Status

CACHE = Path("data/cache")


class AIMomentumFilter:
    def __init__(self, engine, min_momentum_pct: float = 0.03):
        self.engine = engine
        self.min_momentum_pct = min_momentum_pct

        # Load AI stock universe
        cm = pickle.load(open(CACHE / "concept_map.pkl", "rb"))
        c2s = cm["concept_to_stocks"]
        ai_hw_kw = ["芯片", "半导体", "算力", "光模块", "CPO", "HBM", "先进封装",
                    "PCB", "服务器", "GPU", "NPU", "存储", "AI", "AIGC", "元器件"]
        self.ai_stocks = set()
        for cname, stocks in c2s.items():
            if len(stocks) < 10 or len(stocks) > 500:
                continue
            for kw in ai_hw_kw:
                if kw in cname:
                    self.ai_stocks.update(stocks)
                    break

    def __call__(self, df, code, date_str):
        if code not in self.ai_stocks:
            return False

        dt = pd.Timestamp(date_str)
        if dt not in df.index:
            return False
        idx = df.index.get_loc(dt)
        if isinstance(idx, np.ndarray):
            idx = idx[0]
        if idx < 20:
            return False

        close = df.iloc[idx]["close"]
        if close < 5:  # no penny stocks
            return False

        # Trend: above MA20
        ma20 = df.iloc[max(0, idx - 19):idx + 1]["close"].mean()
        if close <= ma20:
            return False

        # Momentum: at least 3% in 20 days
        close_20d = df.iloc[max(0, idx - 19)]["close"]
        mom = (close - close_20d) / close_20d
        if mom < self.min_momentum_pct:
            return False

        # Volume: not shrinking
        vol_5d = df.iloc[max(0, idx - 4):idx + 1]["vol"].mean()
        vol_20d = df.iloc[max(0, idx - 19):idx + 1]["vol"].mean()
        if vol_20d <= 0 or vol_5d / vol_20d < 0.8:
            return False

        return True


def create_signal_filter(engine):
    return AIMomentumFilter(engine)


AI_MOMENTUM = StrategyDef(
    name="ai_momentum",
    description="AI hardware chain: top momentum + above MA20 + volume confirm",
    create_signal_filter=create_signal_filter,
    status=Status.DRAFT,
    tags=["AI", "momentum", "sector", "bull_market"],
)

register(AI_MOMENTUM)
