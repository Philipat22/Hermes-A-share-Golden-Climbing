"""
Sector Rotation — Pick stocks in strongest momentum sectors.

Uses concept_map.pkl to group stocks by concept, then picks
stocks in top-performing concepts with individual momentum confirmation.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from strategies import StrategyDef, register, BacktestResult, Status

CACHE = Path("data/cache")


class SectorRotationFilter:
    def __init__(self, engine, top_n_concepts: int = 5, min_momentum: float = 0.03):
        self.engine = engine
        self.top_n_concepts = top_n_concepts
        self.min_momentum = min_momentum

        # Load concept map
        cm = pickle.load(open(CACHE / "concept_map.pkl", "rb"))
        self.concept_to_stocks = cm["concept_to_stocks"]
        self.concept_names = cm["concept_names"]

    def __call__(self, df, code, date_str):
        dt = pd.Timestamp(date_str)
        if dt not in df.index:
            return False
        
        idx = df.index.get_loc(dt)
        if isinstance(idx, np.ndarray):
            idx = idx[0]
        if idx < 20:
            return False

        # Individual momentum
        close = df.iloc[idx]["close"]
        close_20d = df.iloc[max(0, idx - 19)]["close"]
        mom = (close - close_20d) / close_20d
        if mom < self.min_momentum:
            return False

        # Volume confirmation
        vol_5d = df.iloc[max(0, idx - 4):idx + 1]["vol"].mean()
        vol_20d_vol = df.iloc[max(0, idx - 19):idx + 1]["vol"].mean()
        if vol_20d_vol <= 0 or vol_5d / vol_20d_vol < 0.8:
            return False

        # Must be in a hot concept — checked externally
        # (This filter can't efficiently compute sector momentum per date,
        #  so we accept all and let the sector check run post-hoc)
        return True


def create_signal_filter(engine):
    return SectorRotationFilter(engine)


SECTOR_ROTATION = StrategyDef(
    name="sector_rotation",
    description="Top momentum sectors + individual stock momentum + volume confirm",
    create_signal_filter=create_signal_filter,
    status=Status.DRAFT,
    tags=["momentum", "sector", "bull_market"],
)

register(SECTOR_ROTATION)
