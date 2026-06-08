
# -*- coding: utf-8 -*-
"""
Layer 0 Cross-Validation v2: Golden Pit + Momentum side-by-side.

Engine fix: regime now uses simple close > MA60 (not regime.pkl labels).

Golden pit was DISPROVEN (data contamination / look-ahead bias in cached signals).
Expected result: near-zero or slightly negative median. This validates the engine
is NOT reproducing the contaminated +21% number.

We also test a "dumb" positive signal: DD>=30% (deep drawdown) which should
show SOME mean reversion even without filters.
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from engine.layer0_actuarial import create_engine, MIN_SAMPLES

engine = create_engine()

# ── Signal 1: Golden Pit (simplified, NO trend/C1/C4 filters) ──
def golden_pit_vs(df, code, date_str):
    """DD>=15% + vol contraction, in bull market (engine handles regime)."""
    dt = pd.Timestamp(date_str)
    if dt not in df.index:
        return False
    idx = df.index.get_loc(dt)
    if isinstance(idx, np.ndarray): idx = idx[0]
    if idx < 60 or '688' in code:
        return False
    
    high_60 = df.iloc[max(0, idx-60):idx+1]['high'].max()
    dd = (df.iloc[idx]['close'] - high_60) / high_60
    if dd > -0.15:
        return False
    
    vol_5 = df.iloc[max(0, idx-4):idx+1]['vol'].mean()
    vol_20 = df.iloc[max(0, idx-19):idx+1]['vol'].mean()
    if vol_20 <= 0 or vol_5 / vol_20 >= 0.8:
        return False
    
    return True

# ── Signal 2: Deep DD (DD>=30%, no vol filter) ──
def deep_dd_signal(df, code, date_str):
    """DD>=30% - extreme drawdown, expect mean reversion."""
    dt = pd.Timestamp(date_str)
    if dt not in df.index:
        return False
    idx = df.index.get_loc(dt)
    if isinstance(idx, np.ndarray): idx = idx[0]
    if idx < 60 or '688' in code:
        return False
    
    high_60 = df.iloc[max(0, idx-60):idx+1]['high'].max()
    dd = (df.iloc[idx]['close'] - high_60) / high_60
    return dd <= -0.30

# ── Scan both signals ──
for name, filt in [("GoldenPit_v2", golden_pit_vs), ("DeepDD>=30%", deep_dd_signal)]:
    print(f"\n{'='*70}")
    print(f"SCANNING: {name}")
    print(f"{'='*70}")
    
    t0 = time.time()
    sig_df = engine.find_signals(
        signal_filter=filt,
        date_start="2019-01-01",
        date_end="2026-05-31",
        sample_stocks=3000,
        max_signals=50000,
    )
    t1 = time.time()
    
    n_bull = (sig_df['regime'] == 'bull').sum()
    n_bear = (sig_df['regime'] == 'bear').sum()
    print(f"Scan: {len(sig_df)} signals ({t1-t0:.0f}s) | bull={n_bull}, bear={n_bear}")
    
    # Analyze bull only
    bull = sig_df[sig_df['regime'] == 'bull']
    if len(bull) >= MIN_SAMPLES:
        report = engine.analyze(bull, signal_name=f"{name}_Bull")
        print(report.summary())
    else:
        print(f"  Insufficient bull signals: {len(bull)}")
    
    # Analyze bear only  
    bear = sig_df[sig_df['regime'] == 'bear']
    if len(bear) >= MIN_SAMPLES:
        report_b = engine.analyze(bear, signal_name=f"{name}_Bear")
        print(report_b.summary())

# ── Summary ──
print(f"\n{'='*70}")
print("CROSS-VALIDATION VERDICT")
print(f"{'='*70}")
print("Golden pit was DISPROVEN due to data contamination.")
print("Expected engine output: near-zero or slightly negative median.")
print("If engine outputs +21%: BUG - reproducing contaminated result.")
print("If engine outputs ~0%: CORRECT - matches recomputed clean data.")
print("The engine passes cross-validation when its output aligns with")
print("what we know from manual recomputation of the disproven strategy.")
