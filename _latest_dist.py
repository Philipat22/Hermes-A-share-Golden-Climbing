"""Latest signal distribution + top/bottom 10."""
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
sd = cs[(cs["trade_date"]>="2019-01-01")&(cs["trade_date"]<="2025-12-31")]["trade_date"].dt.strftime("%Y-%m-%d").tolist()[::5]
ss = list(prices.keys())[:5000]

signals = []
for ds in sd:
    dt = pd.Timestamp(ds)
    for code in ss:
        df = prices.get(code)
        if df is None or dt not in df.index: continue
        idx = df.index.get_loc(dt)
        if isinstance(idx, np.ndarray): idx = idx[0]
        if idx < 60: continue
        close = df.iloc[idx]["close"]
        if close < 3: continue
        if (code, ds) not in earn_cf: continue
        
        dd60 = (close - df.iloc[max(0,idx-59)]["close"]) / df.iloc[max(0,idx-59)]["close"]
        ddpeak = (close - df.iloc[max(0,idx-59):idx+1]["close"].max()) / df.iloc[max(0,idx-59):idx+1]["close"].max()
        if dd60 > -0.20 and ddpeak > -0.20: continue
        
        i5 = min(idx+5, len(df)-1)
        ret5 = (df.iloc[i5]["close"]-close)/close
        if ret5 <= 0.05: continue
        
        entry = "BOTH" if dd60 <= -0.20 and ddpeak <= -0.20 else ("DD60" if dd60 <= -0.20 else "DDpeak")
        
        fwd = {}
        for fd in [40, 60]:
            e = min(idx+fd, len(df)-1)
            if e > idx: fwd[fd] = (df.iloc[e]["close"]-close)/close
        
        signals.append({"code": code, "date": ds, "entry": entry, "close": close,
                       "dd60": dd60, "ddpeak": ddpeak, **fwd})

sdf = pd.DataFrame(signals)
print(f"金牛确认信号: {len(sdf)} (2019-2025, 每5天采样)")

# Distribution
for fd in [40, 60]:
    vals = sdf[fd].dropna()
    print(f"\n{fd}天收益分布:")
    for p in [1,5,10,25,50,75,90,95,99]:
        print(f"  P{p:>2d}: {np.percentile(vals,p):>+7.1%}")
    print(f"  WR={np.mean(vals>0):.0%} Mean={np.mean(vals):>+6.1%}")

# Top 10
print(f"\n=== TOP 10 ===")
for _, r in sdf.nlargest(10, 60).iterrows():
    print(f"  {r['code']} {r['date']} {r['entry']:>7s} buy={r['close']:>7.1f} DD60={r['dd60']:>+5.0%} DDpk={r['ddpeak']:>+5.0%} 60d={r[60]:>+6.0%}")

# Bottom 10
print(f"\n=== BOTTOM 10 ===")
for _, r in sdf.nsmallest(10, 60).iterrows():
    print(f"  {r['code']} {r['date']} {r['entry']:>7s} buy={r['close']:>7.1f} DD60={r['dd60']:>+5.0%} DDpk={r['ddpeak']:>+5.0%} 60d={r[60]:>+6.0%}")
