"""
V11: Large-cap stocks near DD20% — do they predict further decline?
"""
import pickle, numpy as np, pandas as pd
from pathlib import Path
from datetime import timedelta

CACHE = Path("data/cache")
prices = pickle.load(open(CACHE/"prices_full.pkl","rb"))
for c in prices:
    df = prices[c]
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df.set_index("trade_date", inplace=True)

cs = pd.read_pickle(CACHE/"csi300.pkl")
cs["trade_date"] = pd.to_datetime(cs["trade_date"])
sd = cs[(cs["trade_date"]>="2019-01-01")&(cs["trade_date"]<="2025-12-31")]["trade_date"].dt.strftime("%Y-%m-%d").tolist()[::10]
ss = list(prices.keys())

results = []
for ds in sd:
    dt = pd.Timestamp(ds)
    for code in ss:
        df = prices.get(code)
        if df is None or dt not in df.index: continue
        idx = df.index.get_loc(dt)
        if isinstance(idx, np.ndarray): idx = idx[0]
        if idx < 60: continue
        close = df.iloc[idx]["close"]
        if close < 50: continue  # "large cap" proxy: price > 50
        
        ddpeak = (close - df.iloc[max(0,idx-59):idx+1]["close"].max()) / df.iloc[max(0,idx-59):idx+1]["close"].max()
        if ddpeak < -0.22 or ddpeak > -0.18: continue  # DD in the "near -20%" zone
        
        for fd in [20, 40, 60]:
            e = min(idx+fd, len(df)-1)
            if e > idx:
                fwd_ret = (df.iloc[e]["close"]-close)/close
                results.append({"ddpeak": ddpeak, "window": fd, "ret": fwd_ret, "close": close})

rdf = pd.DataFrame(results)
print(f"Large-cap stocks with DDpeak -18~-22%: {len(rdf)} observations")
print(f"Unique dates: {rdf.groupby('window').size().to_dict() if len(rdf)>0 else 'none'}")

for w in [20, 40, 60]:
    sub = rdf[rdf["window"]==w]["ret"]
    if len(sub) > 5:
        print(f"\n  {w}d after DD-20% zone:")
        for p in [10,25,50,75,90]:
            print(f"    P{p}: {np.percentile(sub,p):>+5.1%}")
        print(f"    WR={np.mean(sub>0):.0%} n={len(sub)}")

# Conclusion
all60 = rdf[rdf["window"]==60]["ret"]
print(f"\n  V11 CONCLUSION:")
print(f"  Large-cap stocks near DD-20%: median 60d = {np.median(all60):>+5.1%}")
print(f"  {'→ NOT a crash signal — they tend to bounce' if np.median(all60) > 0 else '→ Crash signal confirmed'}")
