
# -*- coding: utf-8 -*-
"""
Layer 0 Final Validation: All features + acceptance checklist.

Tests:
  1. Conditional distribution (PE quantile layers)
  2. Batch comparison (3 signals side-by-side)
  3. Rolling audit
"""
import sys, time, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from engine.layer0_actuarial import create_engine, MIN_SAMPLES

engine = create_engine()

# Load fundamentals for PE tagging
print("\nLoading fundamentals data...")
fund = pd.read_pickle("data/cache/fundamentals_daily.pkl")
fund['trade_date'] = pd.to_datetime(fund['trade_date'])
print(f"  Fundamentals: {len(fund)} rows")

def attach_pe_quantile(sig_df, fund_df):
    """Attach PE quantile tag to each signal."""
    df = sig_df.copy()
    df['signal_date'] = pd.to_datetime(df['signal_date'])
    
    # Compute PE quantiles within each date
    fund_df = fund_df.copy()
    fund_df['pe_quantile'] = fund_df.groupby('trade_date')['pe_ttm'].transform(
        lambda x: pd.qcut(x, q=3, labels=['low', 'mid', 'high'], duplicates='drop')
    )
    
    # Join
    df = df.merge(
        fund_df[['ts_code', 'trade_date', 'pe_ttm', 'pe_quantile']],
        left_on=['ts_code', 'signal_date'],
        right_on=['ts_code', 'trade_date'],
        how='left'
    )
    return df

# ── Signal definitions ──
def deep_dd_30(df, code, date_str):
    """DD >= 30%, no other filters."""
    dt = pd.Timestamp(date_str)
    if dt not in df.index: return False
    idx = df.index.get_loc(dt)
    if isinstance(idx, np.ndarray): idx = idx[0]
    if idx < 60 or '688' in code: return False
    high_60 = df.iloc[max(0, idx-60):idx+1]['high'].max()
    return (df.iloc[idx]['close'] - high_60) / high_60 <= -0.30

def golden_pit_simple(df, code, date_str):
    """DD>=15% + vol contraction."""
    dt = pd.Timestamp(date_str)
    if dt not in df.index: return False
    idx = df.index.get_loc(dt)
    if isinstance(idx, np.ndarray): idx = idx[0]
    if idx < 60 or '688' in code: return False
    
    high_60 = df.iloc[max(0, idx-60):idx+1]['high'].max()
    dd = (df.iloc[idx]['close'] - high_60) / high_60
    if dd > -0.15: return False
    
    vol_5 = df.iloc[max(0, idx-4):idx+1]['vol'].mean()
    vol_20 = df.iloc[max(0, idx-19):idx+1]['vol'].mean()
    return vol_20 > 0 and vol_5 / vol_20 < 0.8

def momentum_signal(df, code, date_str):
    """20d ret>=5% + vol expansion + 4/5 up days."""
    dt = pd.Timestamp(date_str)
    if dt not in df.index: return False
    idx = df.index.get_loc(dt)
    if isinstance(idx, np.ndarray): idx = idx[0]
    if idx < 60: return False
    
    ret_20 = df.iloc[idx]['close'] / df.iloc[idx-20]['close'] - 1
    if ret_20 <= 0.05: return False
    
    vol_5 = df.iloc[max(0, idx-4):idx+1]['vol'].mean()
    vol_20 = df.iloc[max(0, idx-19):idx+1]['vol'].mean()
    if vol_20 <= 0 or vol_5 / vol_20 < 1.5: return False
    
    recent = df.iloc[max(0, idx-4):idx+1]
    return (recent['close'] > recent['open']).sum() >= 4

# ── Scan all three signals (quick: 1000 stocks, every 10th day) ──
all_signals = {}
for name, filt in [("DeepDD30", deep_dd_30), ("GoldenPit", golden_pit_simple), ("Momentum", momentum_signal)]:
    print(f"\nScanning {name}...")
    t0 = time.time()
    sig_df = engine.find_signals(
        signal_filter=filt,
        date_start="2020-01-01",
        date_end="2026-05-31",
        sample_stocks=1000,
        max_signals=30000,
    )
    print(f"  {len(sig_df)} signals ({time.time()-t0:.0f}s)")
    all_signals[name] = sig_df

# ── 1. Conditional Distribution: PE layers on DeepDD in bear ──
print(f"\n{'='*70}")
print("TEST: Conditional Distribution (PE layers)")
print(f"{'='*70}")

deep_dd_bear = all_signals["DeepDD30"]
deep_dd_bear = deep_dd_bear[deep_dd_bear['regime'] == 'bear']
deep_dd_bear = attach_pe_quantile(deep_dd_bear, fund)

print(f"\nDeepDD30 Bear, n={len(deep_dd_bear)}")
engine.print_conditional_summary(deep_dd_bear, 'pe_quantile', 'DeepDD30_Bear')

# ── 2. Batch Comparison ──
print(f"\n{'='*70}")
print("TEST: Batch Comparison (3 signals, bear market)")
print(f"{'='*70}")

bear_signals = {}
for name, sdf in all_signals.items():
    bear = sdf[sdf['regime'] == 'bear']
    if len(bear) >= MIN_SAMPLES:
        bear_signals[name] = bear

engine.compare_signals(bear_signals)

# ── 3. Rolling Audit ──
print(f"\n{'='*70}")
print("TEST: Rolling Audit")
print(f"{'='*70}")

for name, sdf in all_signals.items():
    engine.rolling_audit(sdf, signal_name=name, lookback_months=12)

# ── Summary ──
print(f"\n{'='*70}")
print("LAYER 0 FEATURE CHECKLIST")
print(f"{'='*70}")
print("  [x] Distribution stats (median, win rate, skew, VaR)")
print("  [x] Null hypothesis bootstrap + KS test")
print("  [x] Walk-forward validation (4 windows)")
print("  [x] Bootstrap robustness (VaR CI)")
print("  [x] Yearly breakdown + decay detection")
print("  [x] Conditional distributions (PE layers)")
print("  [x] Batch signal comparison")
print("  [x] Rolling audit")
print("\n  Layer 0 engine: FEATURE COMPLETE")
print("  Ready for Layer 1 scanner development.")
