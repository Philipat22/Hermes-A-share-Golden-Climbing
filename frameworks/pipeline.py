"""
Full Pipeline — Run all 11 frameworks + Layer 0 on a signal.

Usage (from project root):
  D:\Python\python.exe frameworks/pipeline.py

Or import:
  from frameworks.pipeline import run_full_pipeline
  result = run_full_pipeline(strategy_logic='mean_reversion', signal_name='DeepDD30', ...)
"""

import sys
from pathlib import Path

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from frameworks.checklist import (
    CheckResult,
    check_physical_layer,
    check_opponent,
    check_reflexivity,
    check_quant_validation,
    check_safety_gate,
    check_incremental_audit,
    check_position_sizing,
    check_portfolio_correlation,
    check_tail_risk,
    check_actuarial_synthesis,
    check_execution_iron_law,
)


def run_full_pipeline(
    # ── Strategy identity ──
    strategy_logic: str,           # "mean_reversion", "momentum", etc.
    signal_name: str,             # "DeepDD30", "GoldenPit", etc.
    
    # ── Opponent inputs ──
    signal_type: str = "deep_dd",  # "deep_dd", "narrow_dd", "limit_down_streak"
    market_regime: str = "bear",   # "bear", "bull", "sideways"
    vr_ratio: float = None,
    dd_3day_change: float = None,
    fall_speed_ratio: float = None,
    
    # ── Reflexivity inputs ──
    market_state: str = "normal",
    signal_state_median: float = 0.03,
    signal_overall_median: float = 0.03,
    rolling_3m_median: float = 0.03,
    rolling_deviation: float = 0.005,
    
    # ── Quant validation (from Layer 0) ──
    n_signals: int = 0,
    median: float = 0.0,
    win_rate: float = 0.0,
    null_median: float = 0.0,
    null_win_rate: float = 0.0,
    ks_pvalue: float = 1.0,
    profit_loss_ratio: float = 0.0,
    skewness: float = 0.0,
    var_ci_width: float = 0.0,
    wf_medians: list = None,
    yearly_medians: dict = None,
    
    # ── Safety gate extras ──
    left_tail_1: float = 0.0,
    bull_median: float = None,
    bear_median: float = None,
    data_clean: bool = True,
    unexecutable_pct: float = 0.0,
    daily_signal_rate: float = 0.0,
    
    # ── Incremental audit (per filter) ──
    filters_to_test: list = None,  # [("PE_filter", baseline_median, filtered_median, baseline_n, filtered_n), ...]
    
    # ── Position sizing ──
    avg_win: float = 0.0,
    avg_loss: float = 0.0,
    left_tail_5: float = 0.0,
    account_total: float = 40000,
    current_total_exposure: float = 0.0,
    same_sector_count: int = 0,
    
    # ── Portfolio correlation ──
    holdings: list = None,  # [{"code":..., "weight":..., "signal_source":..., "sector":...}]
    new_candidate: dict = None,
    
    # ── Tail risk ──
    position_yuan: float = 0,
    daily_volume_avg: float = 0,
    is_st: bool = False,
    consecutive_limit_down_days: int = 0,
    
    # ── Execution ──
    recent_violations: int = 0,
    signal_frequency_spike: bool = False,

) -> dict:
    """
    Run ALL 11 frameworks in order. Returns complete results dict.
    
    Stops early if any framework returns 'reject' in early gates (01, 02).
    """
    
    if wf_medians is None:
        wf_medians = []
    if yearly_medians is None:
        yearly_medians = {}
    if filters_to_test is None:
        filters_to_test = []
    if holdings is None:
        holdings = []

    results = {}
    stopped_early = False

    def _record(name, r):
        results[name] = {"verdict": r.verdict, "reason": r.reason, "details": r.details}
        return r

    # ═══ GATE 1: Physical Layer ═══
    r = check_physical_layer(strategy_logic)
    _record("01_physical", r)
    print(f"\n{'='*60}")
    print(f"[{r.verdict.upper()}] 01 PHYSICAL LAYER")
    print(f"     {r.reason}")
    if r.is_blocked():
        print("  >> PIPELINE STOPPED: Strategy violates physical constraints.")
        stopped_early = True
        return results

    # ═══ GATE 2: Opponent Analysis ═══
    r = check_opponent(signal_type, market_regime, vr_ratio, dd_3day_change, fall_speed_ratio)
    _record("02_opponent", r)
    print(f"\n{'='*60}")
    print(f"[{r.verdict.upper()}] 02 OPPONENT ANALYSIS")
    print(f"     {r.reason}")
    if r.is_blocked():
        print("  >> PIPELINE STOPPED: Opponent not confirmed.")
        stopped_early = True
        return results

    # ═══ STEP 3: Reflexivity ═══
    r = check_reflexivity(market_state, signal_state_median, signal_overall_median,
                          rolling_3m_median, rolling_deviation)
    _record("03_reflexivity", r)
    print(f"\n{'='*60}")
    print(f"[{r.verdict.upper()}] 03 REFLEXIVITY")
    print(f"     {r.reason}")

    # ═══ STEP 4: Quant Validation (requires Layer 0 output) ═══
    if n_signals < 200:
        print(f"\n{'='*60}")
        print(f"[SKIP] 04 QUANT VALIDATION — insufficient signals (n={n_signals})")
        results["04_quant"] = {"verdict": "reject", "reason": f"n={n_signals} < 200"}
        stopped_early = True
        return results

    r = check_quant_validation(
        signal_name=signal_name,
        n_signals=n_signals,
        median=median,
        win_rate=win_rate,
        null_median=null_median,
        null_win_rate=null_win_rate,
        ks_pvalue=ks_pvalue,
        profit_loss_ratio=profit_loss_ratio,
        skewness=skewness,
        var_ci_width=var_ci_width,
        wf_medians=wf_medians,
        yearly_medians=yearly_medians,
    )
    _record("04_quant", r)
    print(f"\n{'='*60}")
    print(f"[{r.verdict.upper()}] 04 QUANT VALIDATION")
    print(f"     {r.reason}")
    if r.is_blocked():
        print("  >> PIPELINE STOPPED: Too many validation failures.")
        stopped_early = True
        return results

    # ═══ STEP 5: Safety Gate ═══
    r = check_safety_gate(
        signal_name=signal_name,
        n_signals=n_signals,
        wf_medians=wf_medians,
        yearly_medians=yearly_medians,
        left_tail_1=left_tail_1,
        bull_median=bull_median,
        bear_median=bear_median,
        data_clean=data_clean,
        unexecutable_pct=unexecutable_pct,
        daily_signal_rate=daily_signal_rate,
    )
    _record("05_safety", r)
    print(f"\n{'='*60}")
    print(f"[{r.verdict.upper()}] 05 SAFETY GATE")
    print(f"     {r.reason}")
    if r.is_blocked():
        print("  >> PIPELINE STOPPED: Safety gate failed.")
        stopped_early = True
        return results

    # ═══ STEP 6: Incremental Audit (for each proposed filter) ═══
    filter_results = {}
    for f_name, base_med, filt_med, base_n, filt_n in filters_to_test:
        r = check_incremental_audit(f_name, base_med, filt_med, base_n, filt_n)
        filter_results[f_name] = {"verdict": r.verdict, "reason": r.reason}
        print(f"\n  [{r.verdict.upper()}] 06 INCREMENTAL AUDIT: {f_name}")
        print(f"       {r.reason}")
    results["06_incremental"] = filter_results

    # ═══ STEP 7-9: Position + Portfolio + Tail (run in logical sequence) ═══
    
    # 07 Position Sizing
    r = check_position_sizing(
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        left_tail_5=left_tail_5,
        account_total=account_total,
        current_total_exposure=current_total_exposure,
        same_sector_count=same_sector_count,
        n_signals=n_signals,
    )
    _record("07_position", r)
    print(f"\n{'='*60}")
    print(f"[{r.verdict.upper()}] 07 POSITION SIZING")
    print(f"     {r.reason}")

    # 08 Portfolio Correlation
    if new_candidate:
        r = check_portfolio_correlation(holdings, new_candidate)
        _record("08_portfolio", r)
        print(f"\n{'='*60}")
        print(f"[{r.verdict.upper()}] 08 PORTFOLIO CORRELATION")
        print(f"     {r.reason}")
    else:
        results["08_portfolio"] = {"verdict": "pass", "reason": "No portfolio context provided."}

    # 09 Tail Risk
    r = check_tail_risk(
        left_tail_5=left_tail_5,
        left_tail_1=left_tail_1,
        account_total=account_total,
        position_yuan=position_yuan,
        daily_volume_avg=daily_volume_avg,
        is_st=is_st,
        consecutive_limit_down_days=consecutive_limit_down_days,
    )
    _record("09_tail", r)
    print(f"\n{'='*60}")
    print(f"[{r.verdict.upper()}] 09 TAIL RISK")
    print(f"     {r.reason}")

    # ═══ STEP 10: Actuarial Synthesis ═══
    framework_verdicts = [
        (name, info["verdict"], info["reason"])
        for name, info in results.items()
        if name not in ("06_incremental",)
    ]
    r = check_actuarial_synthesis(framework_verdicts)
    _record("10_synthesis", r)
    print(f"\n{'='*60}")
    print(f"[{r.verdict.upper()}] 10 ACTUARIAL SYNTHESIS")
    print(f"     {r.reason}")

    # ═══ STEP 11: Execution Iron Law ═══
    r = check_execution_iron_law(
        synthesis_verdict=r.verdict,
        hard_stop_triggered=False,
        recent_violations=recent_violations,
        signal_frequency_spike=signal_frequency_spike,
    )
    _record("11_execution", r)
    print(f"\n{'='*60}")
    print(f"[{r.verdict.upper()}] 11 EXECUTION IRON LAW")
    print(f"     {r.reason}")
    print(f"\n{'='*60}")
    print(f"FINAL: {'GO' if r.verdict == 'pass' else 'REDUCE' if r.verdict == 'warn' else 'NO GO'}")
    print(f"{'='*60}\n")

    return results


# ══════════════════════════════════════════════
# DEMO: Run with DeepDD30 data from Layer 0
# ══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  FRAMEWORK PIPELINE — DEMO RUN")
    print("  Signal: DeepDD30 (Bear Market)")
    print("=" * 60)

    results = run_full_pipeline(
        strategy_logic="mean_reversion",
        signal_name="DeepDD30",

        # Opponent
        signal_type="deep_dd",
        market_regime="bear",
        vr_ratio=0.45,
        dd_3day_change=0.8,
        fall_speed_ratio=0.65,

        # Reflexivity
        market_state="normal",
        signal_state_median=0.03,
        signal_overall_median=0.03,
        rolling_3m_median=0.028,
        rolling_deviation=0.002,

        # Layer 0 output (from actual run)
        n_signals=17250,
        median=0.03,
        win_rate=0.574,
        null_median=-0.004,
        null_win_rate=0.48,
        ks_pvalue=0.001,
        profit_loss_ratio=1.8,
        skewness=0.35,
        var_ci_width=0.0058,
        wf_medians=[0.017, -0.028, 0.014, 0.049],
        yearly_medians={2022: 0.017, 2023: -0.028, 2024: 0.014, 2025: 0.049},

        # Safety gate
        left_tail_1=-0.40,
        bull_median=-0.004,
        bear_median=0.034,
        data_clean=True,
        unexecutable_pct=0.02,
        daily_signal_rate=8.5,

        # Incremental audit — test PE filter
        filters_to_test=[
            ("PE_lt_20", 0.03, 0.0374, 17250, 8640),
            ("volume_shrink", 0.03, 0.013, 17250, 9200),
        ],

        # Position sizing
        avg_win=0.18,
        avg_loss=-0.15,
        left_tail_5=-0.243,
        account_total=40000,
        current_total_exposure=0.0,
        same_sector_count=0,

        # Portfolio correlation
        holdings=[
            {"code": "600001", "weight": 0.15, "signal_source": "deep_dd30", "sector": "新能源"},
        ],
        new_candidate={"code": "600002", "weight": 0.10, "signal_source": "deep_dd30", "sector": "新能源"},

        # Tail risk
        position_yuan=4000,
        daily_volume_avg=50_000_000,
        is_st=False,
        consecutive_limit_down_days=0,

        recent_violations=0,
        signal_frequency_spike=False,
    )

    # Summary
    print("\nPIPELINE VERDICT SUMMARY:")
    for name, info in results.items():
        if name == "06_incremental":
            for fname, finfo in info.items():
                print(f"  [{finfo['verdict']:6s}] {name}/{fname}")
        else:
            print(f"  [{info['verdict']:6s}] {name}")
