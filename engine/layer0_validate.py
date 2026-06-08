
# -*- coding: utf-8 -*-
"""
Layer 0 Actuarial Engine — Validation Suite

Acceptance criteria:
  1. Null hypothesis discrimination: signal vs Bootstrap 10K -> KS p<0.01, median diff>2pp
  2. Walk-forward stability: 4 rolling windows, deviation <= +-1.5pp
  3. Bootstrap robustness: VaR CI width <= +-3pp
  4. Distribution shape check: skewness <= 0.5
  5. Signal decay detection: latest year >= 50% of historical mean

Tests the Bull Momentum signal as the first concrete scanner.
"""
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from engine.layer0_actuarial import (
    ActuarialEngine, create_engine, MIN_SAMPLES
)


def define_momentum_signal(engine):
    """
    Bull-market momentum signal:
    - Market in bull regime (CSI300 > MA60)
    - Stock 20d return in top decile (filtered post-scan)
    - 5d volume expansion: avg_vol_5d > avg_vol_20d * 1.5
    - 4+ up days out of last 5
    """
    def is_signal(df, code, date_str):
        dt = pd.Timestamp(date_str)
        if dt not in df.index:
            return False
        idx = df.index.get_loc(dt)
        if isinstance(idx, slice) or (isinstance(idx, np.ndarray) and len(idx) > 1):
            return False
        if isinstance(idx, np.ndarray):
            idx = idx[0]
        if idx < 60:
            return False

        ret_20d = (df.iloc[idx]['close'] / df.iloc[idx - 20]['close'] - 1
                   if idx >= 20 else 0)
        if ret_20d <= 0.05:
            return False

        if idx < 20:
            return False
        vol_5d = df.iloc[idx - 4:idx + 1]['vol'].mean()
        vol_20d = df.iloc[idx - 19:idx + 1]['vol'].mean()
        if vol_20d <= 0 or vol_5d / vol_20d < 1.5:
            return False

        recent_5 = df.iloc[idx - 4:idx + 1]
        up_days = (recent_5['close'] > recent_5['open']).sum()
        if up_days < 4:
            return False

        return True

    return is_signal


def validate():
    print("=" * 70)
    print("Layer 0 Actuarial Engine — Validation Suite")
    print("=" * 70)

    # ── Test 1: Engine Init & Data Load ──
    print("\n-- Test 1: Engine Init --")
    t0 = time.time()
    engine = create_engine()
    t1 = time.time()

    assert engine._loaded, "Data not loaded"
    assert len(engine._all_stocks) > 5000
    assert len(engine.csi300) > 2000

    n_bull = (engine.csi300['regime'] == 'bull').sum()
    n_bear = (engine.csi300['regime'] == 'bear').sum()
    print(f"  [OK] Engine ready ({t1 - t0:.1f}s)")
    print(f"  [OK] Stocks: {len(engine._all_stocks)}")
    print(f"  [OK] CSI300: {len(engine.csi300)} days (bull={n_bull}, bear={n_bear})")

    # ── Test 2: Momentum Signal Scan ──
    print("\n-- Test 2: Bull Momentum Signal Scan --")
    mom_filter = define_momentum_signal(engine)

    t0 = time.time()
    sig_df = engine.find_signals(
        signal_filter=mom_filter,
        date_start="2019-01-01",
        date_end="2026-05-31",
        sample_stocks=3000,
        max_signals=50000,
    )
    t1 = time.time()

    # Keep only bull-regime signals
    sig_df = sig_df[sig_df['regime'] == 'bull']

    print(f"  [OK] Scan done ({t1 - t0:.1f}s)")
    print(f"  [OK] Total signals: {len(sig_df)}")
    print(f"  [OK] Fwd return range: {sig_df['forward_return'].min():+.2%} ~ {sig_df['forward_return'].max():+.2%}")

    assert len(sig_df) >= MIN_SAMPLES, f"Insufficient signals: {len(sig_df)} < {MIN_SAMPLES}"

    # ── Test 3: Actuarial Analysis ──
    print("\n-- Test 3: Actuarial Analysis --")
    report = engine.analyze(sig_df, signal_name="BullMomentum")
    print(report.summary())

    # ── Test 4: Null Hypothesis Discrimination ──
    print("\n-- Test 4: Null Hypothesis Discrimination --")
    print(f"  Signal median: {report.median:+.2%}")
    print(f"  Null median:   {report.null_median:+.2%}")
    print(f"  Median diff:   {report.median_diff:+.2%} (req >2pp)")
    print(f"  Win rate diff: {report.win_rate_diff:+.1%} (req >5pp)")
    print(f"  KS p-value:    {report.ks_pvalue:.4f} (req <0.01)")

    check_1 = report.median_diff > 0.02
    check_2 = report.ks_pvalue < 0.01
    print(f"  {'[PASS]' if check_1 else '[FAIL]'} Check 1: median_diff > 2pp")
    print(f"  {'[PASS]' if check_2 else '[FAIL]'} Check 2: KS p < 0.01")
    print(f"  {'[PASS]' if report.is_significant else '[FAIL]'} Overall: signal has info content")

    # ── Test 5: Walk-Forward Stability ──
    print("\n-- Test 5: Walk-Forward Stability --")
    for i, m in enumerate(report.wf_medians):
        print(f"  Window {i+1}: median={m:+.2%}")
    print(f"  Range: {report.wf_range:.2%} (req <=1.5pp)")

    check_3 = report.wf_stable
    print(f"  {'[PASS]' if check_3 else '[FAIL]'} Check 3: walk-forward stable")

    # ── Test 6: Bootstrap Robustness ──
    print("\n-- Test 6: Bootstrap Robustness --")
    print(f"  Median 95%CI: [{report.median_ci[0]:+.2%}, {report.median_ci[1]:+.2%}]")
    print(f"  VaR CI width:  {report.var_ci_width:.3%} (req <=3pp)")

    check_4 = report.var_ci_width <= 0.03
    print(f"  {'[PASS]' if check_4 else '[FAIL]'} Check 4: VaR CI width <= +-3pp")

    # ── Test 7: Distribution Shape ──
    print("\n-- Test 7: Distribution Shape")
    print(f"  Skewness: {report.skewness:+.3f} (req <=0.5, negative normal)")

    check_5 = report.skewness <= 0.5
    print(f"  {'[PASS]' if check_5 else '[WARN]'} Check 5: skewness <= 0.5")

    # ── Test 8: Signal Decay ──
    print("\n-- Test 8: Yearly Effect Size & Decay --")
    sig_df['year'] = pd.to_datetime(sig_df['signal_date']).dt.year
    for year in sorted(sig_df['year'].unique()):
        n_year = (sig_df['year'] == year).sum()
        med_year = report.yearly_medians.get(int(year), 0)
        print(f"  {year}: n={n_year}, median={med_year:+.2%}")

    if len(report.yearly_medians) >= 2:
        years = sorted(report.yearly_medians.keys())
        latest = report.yearly_medians[years[-1]]
        hist_mean = np.mean([report.yearly_medians[y] for y in years[:-1]])
        decay_ratio = latest / hist_mean if hist_mean != 0 else 1.0
        print(f"  Latest/Hist mean = {decay_ratio:.2f}")

        check_6 = decay_ratio >= 0.5
        print(f"  {'[PASS]' if check_6 else '[WARN]'} Check 6: no severe decay (>=50%)")
    else:
        check_6 = True

    # ── Summary ──
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)

    all_checks = {
        "NullHyp-MedianDiff": check_1,
        "NullHyp-KS": check_2,
        "WalkForward": check_3,
        "Bootstrap Robustness": check_4,
        "Distribution Shape": check_5,
        "Signal Decay": check_6,
    }

    for name, result in all_checks.items():
        print(f"  {'[PASS]' if result else '[FAIL]'} {name}")

    passed = sum(all_checks.values())
    total = len(all_checks)
    print(f"\n  {passed}/{total} checks passed")

    if passed == total:
        print("\n  *** Layer 0 ACTUARIAL ENGINE VALIDATED ***")
        print("  Ready for Layer 1.")
    else:
        print(f"\n  [WARN] {total - passed} checks failed. Fix before Layer 1.")

    return engine, sig_df, report, all_checks


if __name__ == "__main__":
    engine, sig_df, report, checks = validate()
