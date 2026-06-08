"""AI Trend Mid — stocks in AI chain, started moving, volume confirming."""
from pathlib import Path
import pickle, numpy as np, pandas as pd
from strategies import StrategyDef, register, BacktestResult, Status

CACHE = Path("data/cache")

class AITrendMidFilter:
    def __init__(self, engine, min_mom=0.05, max_mom=0.30):
        self.engine = engine; self.min_mom = min_mom; self.max_mom = max_mom
        cm = pickle.load(open(CACHE/"concept_map.pkl","rb"))
        c2s = cm["concept_to_stocks"]
        ai_kw = ["芯片","半导体","算力","光模块","CPO","HBM","先进封装","PCB","服务器","GPU","NPU","存储","AI","AIGC","元器件"]
        self.ai_stocks = set()
        for cn, ss in c2s.items():
            if 10 <= len(ss) <= 500:
                for kw in ai_kw:
                    if kw in cn: self.ai_stocks.update(ss); break

    def __call__(self, df, code, date_str):
        if code not in self.ai_stocks: return False
        dt = pd.Timestamp(date_str)
        if dt not in df.index: return False
        idx = df.index.get_loc(dt)
        if isinstance(idx, np.ndarray): idx = idx[0]
        if idx < 20: return False
        close = df.iloc[idx]["close"]
        if close < 5: return False
        ma20 = df.iloc[max(0,idx-19):idx+1]["close"].mean()
        if close <= ma20: return False
        close_20d = df.iloc[max(0,idx-19)]["close"]
        mom = (close - close_20d) / close_20d
        if mom < self.min_mom or mom > self.max_mom: return False
        vol_5d = df.iloc[max(0,idx-4):idx+1]["vol"].mean()
        vol_20d = df.iloc[max(0,idx-19):idx+1]["vol"].mean()
        if vol_20d <= 0 or vol_5d/vol_20d < 1.0: return False
        return True

def create_signal_filter(engine): return AITrendMidFilter(engine)

AI_TREND = StrategyDef(name="ai_trend_mid",
    description="AI chain mid-trend: momentum 5-30% + volume expanding + above MA20",
    create_signal_filter=create_signal_filter, status=Status.DRAFT,
    tags=["AI","trend","bull_market"])
register(AI_TREND)
