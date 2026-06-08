
# -*- coding: utf-8 -*-
"""Quick debug: manual trace + random baseline + regime check."""
import sys, pickle, time
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from engine.layer0_actuarial import create_engine

engine = create_engine()

# ── Test 1: Manual trace ──
print("TEST 1: Manual golden pit trace")
for code in ['000001.SZ', '600519.SH', '002415.SZ']:
    df = engine.prices[code]
    for date_str in ['2024-09-18', '2024-02-05', '2022-10-31']:
        dt = pd.Timestamp(date_str)
        if dt not in df.index:
            continue
        idx = df.index.get_loc(dt)
        if isinstance(idx, np.ndarray): idx = idx[0]
        
        close = df.iloc[idx]['close']
        high_60 = df.iloc[max(0, idx-60):idx+1]['high'].max()
        dd = (close - high_60) / high_60
        
        exit_pos = min(idx + 40, len(df) - 1)
        fwd_ret = (df.iloc[exit_pos]['close'] - close) / close
        
        if dd <= -0.15:
            print(f"  {code} {date_str}: DD={dd:.1%}, 40d fwd={fwd_ret:+.1%}")

# ── Test 2: Random baseline (fast) ──
print("\nTEST 2: Random forward returns")
rng = np.random.RandomState(42)
all_codes = list(engine.prices.keys())
random_rets = []

t0 = time.time()
for code in all_codes:
    df = engine.prices[code]
    if len(df) < 120:
        continue
    # Take one random entry per stock
    max_idx = len(df) - 50
    if max_idx <= 60:
        continue
    idx = rng.randint(60, max_idx)
    close = df.iloc[idx]['close']
    exit_c = df.iloc[idx + 40]['close']
    if close > 0:
        random_rets.append((exit_c - close) / close)
    if len(random_rets) >= 3000:
        break

random_rets = np.array(random_rets)
print(f"  Samples: {len(random_rets)} ({time.time()-t0:.1f}s)")
print(f"  Median: {np.median(random_rets):+.2%}")
print(f"  Mean: {np.mean(random_rets):+.2%}")
print(f"  Win rate: {np.mean(random_rets > 0):.1%}")
print(f"  P5: {np.percentile(random_rets, 5):+.2%}  P95: {np.percentile(random_rets, 95):+.2%}")

# ── Test 3: Regime ──
print("\nTEST 3: Regime distribution")
cs = engine.csi300
print(f"  regime value_counts:")
print(f"  {cs['regime'].value_counts().to_dict()}")

# Simple MA60 bull
if 'ma60' in cs.columns:
    n_bull = (cs['close'] > cs['ma60']).sum()
    print(f"  Simple bull (close>ma60): {n_bull}/{len(cs)}")

# Date range
print(f"  CSI300 date range: {cs.index.min()} to {cs.index.max()}")

# ── Test 4: Quick golden pit with NO regime filter ──
print("\nTEST 4: Quick golden pit scan (200 stocks, NO regime filter)")
rng = np.random.RandomState(42)
sample = rng.choice(all_codes, 200, replace=False).tolist()

gp_signals = []
cs_dates_subset = cs.loc['2020-01-01':'2026-05-31'].index[::5]  # every 5th day

for code in sample:
    df = engine.prices.get(code)
    if df is None or len(df) < 120:
        continue
    
    for dt in cs_dates_subset:
        if dt not in df.index:
            continue
        idx = df.index.get_loc(dt)
        if isinstance(idx, np.ndarray): idx = idx[0]
        if idx < 60 or idx + 40 >= len(df):
            continue
        
        high_60 = df.iloc[max(0, idx-60):idx+1]['high'].max()
        close = df.iloc[idx]['close']
        dd = (close - high_60) / high_60
        if dd > -0.15:
            continue
        
        vol_5 = df.iloc[max(0, idx-4):idx+1]['vol'].mean()
        vol_20 = df.iloc[max(0, idx-19):idx+1]['vol'].mean()
        if vol_20 <= 0 or vol_5 / vol_20 >= 0.8:
            continue
        
        if '688' in code:
            continue
        
        exit_c = df.iloc[idx + 40]['close']
        fwd_ret = (exit_c - close) / close
        
        is_bull = False
        if dt in cs.index and 'ma60' in cs.columns:
            ma60 = cs.loc[dt, 'ma60']
            if pd.notna(ma60):
                is_bull = cs.loc[dt, 'close'] > ma60
        
        gp_signals.append({
            'code': code, 'date': dt, 'dd': dd, 
            'fwd_ret': fwd_ret, 'bull': is_bull
        })

gp = pd.DataFrame(gp_signals)
print(f"  Total GP signals: {len(gp)}")
print(f"  In bull: {(gp['bull']).sum()}, In bear: {(~gp['bull']).sum()}")

if len(gp) > 0:
    bull = gp[gp['bull']]
    bear = gp[~gp['bull']]
    print(f"\n  Bull: n={len(bull)}, median={np.median(bull['fwd_ret']):+.2%}, wr={np.mean(bull['fwd_ret']>0):.1%}")
    print(f"  Bear: n={len(bear)}, median={np.median(bear['fwd_ret']):+.2%}, wr={np.mean(bear['fwd_ret']>0):.1%}")
    
    # DD gradient in bull
    print("\n  DD gradient (bull only):")
    for lo, hi, lb in [(-1.0, -0.30, '<=-30%'), (-0.30, -0.20, '-30~-20%'), (-0.20, -0.15, '-20~-15%')]:
        sub = bull[(bull['dd']>=lo) & (bull['dd']<hi)]
        if len(sub) >= 5:
            print(f"    {lb}: n={len(sub)}, med={np.median(sub['fwd_ret']):+.2%}, wr={np.mean(sub['fwd_ret']>0):.1%}")
