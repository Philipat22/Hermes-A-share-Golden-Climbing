"""
DDpeak vs DD40 distribution + window sensitivity.
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

fin = pd.read_pickle(CACHE/"financials.pkl")
fin["end_date"] = pd.to_datetime(fin["end_date"])
earn_cf = set()
for code, grp in fin.groupby("ts_code"):
    grp = grp.sort_values("end_date")
    for i in range(2, len(grp)):
        np0 = grp.iloc[i].get("netprofit_yoy"); np1 = grp.iloc[i-1].get("netprofit_yoy"); np2 = grp.iloc[i-2].get("netprofit_yoy")
        ocf = grp.iloc[i].get("ocfps")
        if pd.notna(np0) and pd.notna(np1) and pd.notna(np2) and np0 > np1 > np2 and np0 > 0 and pd.notna(ocf) and ocf > 0:
            ed = grp.iloc[i]["end_date"]
            vu = grp.iloc[i+1]["end_date"]+timedelta(days=45) if i+1<len(grp) else pd.Timestamp("2027-01-01")
            for d in pd.date_range(ed+timedelta(days=45), vu):
                earn_cf.add((code, d.strftime("%Y-%m-%d")))

cs = pd.read_pickle(CACHE/"csi300.pkl")
cs["trade_date"] = pd.to_datetime(cs["trade_date"])
scan_dates = cs[(cs["trade_date"]>="2019-01-01")&(cs["trade_date"]<="2025-12-31")]["trade_date"].dt.strftime("%Y-%m-%d").tolist()[::10]
stock_sample = list(prices.keys())[:5000]

# Collect all signals with both DDpeak and DD windows
all_signals = []
for ds in scan_dates:
    dt = pd.Timestamp(ds)
    for code in stock_sample:
        df = prices.get(code)
        if df is None or dt not in df.index: continue
        idx = df.index.get_loc(dt)
        if isinstance(idx, np.ndarray): idx = idx[0]
        if idx < 60: continue
        close = df.iloc[idx]["close"]
        if close < 3: continue
        if (code, ds) not in earn_cf: continue
        
        ddpeak = (close - df.iloc[max(0,idx-59):idx+1]["close"].max()) / df.iloc[max(0,idx-59):idx+1]["close"].max()
        dds = {}
        for w in [20,30,40,50,60]:
            dds[f"dd{w}"] = (close - df.iloc[max(0,idx-w)]["close"]) / df.iloc[max(0,idx-w)]["close"]
        
        i5 = min(idx+5, len(df)-1)
        ret5 = (df.iloc[i5]["close"]-close)/close
        confirmed = ret5 > 0.05
        if not confirmed: continue
        
        e60 = min(idx+60, len(df)-1)
        ret60 = (df.iloc[e60]["close"]-close)/close if e60>idx else 0
        
        all_signals.append({
            "code": code, "date": ds, "ddpeak": ddpeak,
            "ret60": ret60, **dds,
        })

sdf = pd.DataFrame(all_signals)
print(f"Total confirmed signals: {len(sdf)}")

# ========== Q1: DDpeak distribution ==========
ddpeak_conf = sdf[sdf["ddpeak"] <= -0.20]
dd40_conf = sdf[sdf["dd40"] <= -0.20]

for label, sub in [("DDpeak(≤-20%)", ddpeak_conf), ("DD40(≤-20%)", dd40_conf)]:
    print(f"\n=== {label}: n={len(sub)} ===")
    r = sub["ret60"].dropna()
    for p in [1,5,10,25,50,75,90,95,99]:
        print(f"  P{p:>2d}: {np.percentile(r,p):>+7.1%}")
    print(f"  WR={np.mean(r>0):.0%} Mean={np.mean(r):>+6.1%} Min={np.min(r):>+6.0%} Max={np.max(r):>+6.0%}")

# Top/bottom 5 DDpeak
print(f"\n--- DDpeak: Best 5 signals ---")
best = ddpeak_conf.nlargest(5, "ret60")
for _, r in best.iterrows():
    print(f"  {r['code']} {r['date']} peak={r['ddpeak']:+.1%} 60d={r['ret60']:+.0%}")
print(f"\n--- DDpeak: Worst 5 ---")
worst = ddpeak_conf.nsmallest(5, "ret60")
for _, r in worst.iterrows():
    print(f"  {r['code']} {r['date']} peak={r['ddpeak']:+.1%} 60d={r['ret60']:+.0%}")

# ========== Q2: Window sensitivity ==========
print(f"\n=== Window Sensitivity (DD threshold=-20%) ===")
for w in [20,30,40,50,60]:
    col = f"dd{w}"
    sub = sdf[sdf[col] <= -0.20]
    if len(sub) < 5: continue
    r = sub["ret60"].dropna()
    print(f"  DD{w}(≤-20%): n={len(sub):>5d} med={np.median(r):>+5.1%} WR={np.mean(r>0):.0%}")
