"""
Full robustness tests:
1. Walk-forward validation (3yr train → 1yr test, rolling)
2. Monte Carlo permutation (1000 shuffles)
3. MA exit window scan (MA15/20/25/30)
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
            vu = grp.iloc[i+1]["end_date"]+timedelta(days=45) if i+1 < len(grp) else pd.Timestamp("2027-01-01")
            for d in pd.date_range(ed+timedelta(days=45), vu):
                earn_cf.add((code, d.strftime("%Y-%m-%d")))

cs = pd.read_pickle(CACHE/"csi300.pkl")
cs["trade_date"] = pd.to_datetime(cs["trade_date"])
all_dates = cs[(cs["trade_date"]>="2019-01-01")&(cs["trade_date"]<="2025-12-31")]["trade_date"].dt.strftime("%Y-%m-%d").tolist()[::5]
ss = list(prices.keys())[:5000]

# Collect all BOTH signals with date, returns, data for MA exit
signals = []
for ds in all_dates:
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
        dd40 = (close - df.iloc[max(0,idx-39)]["close"]) / df.iloc[max(0,idx-39)]["close"]
        ddpeak = (close - df.iloc[max(0,idx-59):idx+1]["close"].max()) / df.iloc[max(0,idx-59):idx+1]["close"].max()
        if dd40 > -0.20 or ddpeak > -0.20: continue
        i5 = min(idx+5, len(df)-1)
        ret5 = (df.iloc[i5]["close"]-close)/close
        if ret5 <= 0.05: continue
        e60 = min(idx+60, len(df)-1)
        ret60 = (df.iloc[e60]["close"]-close)/close if e60 > idx else 0
        # Data for MA exit scan
        price_trail = [float(df.iloc[min(idx+j,len(df)-1)]["close"]) for j in range(1,121)]
        signals.append({"date": ds, "ret60": ret60, "price_trail": price_trail, "close": close})

sdf = pd.DataFrame(signals)
sdf["date_dt"] = pd.to_datetime(sdf["date"])
sdf["year"] = sdf["date_dt"].dt.year
print(f"BOTH signals: {len(sdf)}")
print(f"Yearly: {sdf.groupby('year').size().to_dict()}")

# ====== TEST 1: Walk-Forward ======
print(f"\n{'='*50}")
print(f"  TEST 1: Walk-Forward Validation (3yr train, 1yr test)")
print(f"{'='*50}")

wf_results = []
for test_year in [2022, 2023, 2024, 2025]:
    train = sdf[sdf["year"].between(test_year-3, test_year-1)]
    test = sdf[sdf["year"] == test_year]
    if len(train) < 10 or len(test) < 5: continue
    train_wr = np.mean(train["ret60"] > 0)
    test_wr = np.mean(test["ret60"] > 0)
    test_med = np.median(test["ret60"])
    wf_results.append({"test_year": test_year, "train_n": len(train), "test_n": len(test),
                       "train_wr": train_wr, "test_wr": test_wr, "test_med": test_med})
    delta = test_wr - train_wr
    status = "PASS" if test_wr >= 0.60 else "FAIL"
    print(f"  Train {test_year-3}-{test_year-1}(n={len(train)}) → Test {test_year}(n={len(test)}): "
          f"WR train={train_wr:.0%} test={test_wr:.0%} med={test_med:+.1%} {status} (Δ={delta:+.0%})")

wf_df = pd.DataFrame(wf_results)
if len(wf_df) > 0:
    print(f"\n  WF Summary: test WRs = {[f'{w:.0%}' for w in wf_df['test_wr']]}")
    print(f"  All tests pass (>60% WR): {all(wf_df['test_wr'] >= 0.60)}")

# ====== TEST 2: Monte Carlo ======
print(f"\n{'='*50}")
print(f"  TEST 2: Monte Carlo Permutation (1000 shuffles)")
print(f"{'='*50}")

actual_wr = np.mean(sdf["ret60"] > 0)
actual_med = np.median(sdf["ret60"])
rets_arr = sdf["ret60"].values.copy()  # mutable copy

np.random.seed(42)
shuffled_wrs = []
for _ in range(1000):
    np.random.shuffle(rets_arr)
    chunk_wrs = []
    for i in range(0, len(rets_arr), 5):
        chunk = rets_arr[i:i+5]
        if len(chunk) >= 3:
            chunk_wrs.append(np.mean(chunk > 0))
    shuffled_wrs.append(np.mean(chunk_wrs))

shuffled_wrs = np.array(shuffled_wrs)
shuffled_med = np.median(shuffled_wrs)
p_value = np.mean(shuffled_wrs >= actual_wr)
print(f"  Actual WR: {actual_wr:.1%}")
print(f"  Random WR (median of 1000 shuffles): {shuffled_med:.1%}")
print(f"  P-value (random >= actual): {p_value:.3f}")
print(f"  {'SIGNIFICANT — alpha is real!' if p_value < 0.05 else 'NOT significant — may be noise'}")

# ====== TEST 3: MA window scan ======
print(f"\n{'='*50}")
print(f"  TEST 3: MA Exit Window Scan")
print(f"{'='*50}")

for ma_n in [15, 20, 25, 30]:
    rets = []
    for _, row in sdf.iterrows():
        trail = row["price_trail"]
        if len(trail) < ma_n: continue
        p0 = row["close"]
        exit_ret = None
        for t in range(ma_n, min(len(trail), 120)):
            ma = np.mean(trail[t-ma_n:t])
            if trail[t] < ma:
                exit_ret = (trail[t] - p0) / p0
                break
        if exit_ret is not None:
            rets.append(exit_ret)
        elif len(trail) >= 60:
            rets.append((trail[59] - p0) / p0)
    if rets:
        print(f"  MA{ma_n}: n={len(rets)} med={np.median(rets):>+5.1%} WR={np.mean(np.array(rets)>0):.0%}")
print(f"  Fixed 60d: med={actual_med:>+5.1%} WR={actual_wr:.0%}")
