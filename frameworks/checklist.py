"""
Framework Checklists — Executable implementations of all 11 frameworks.

Each function:
  - Takes concrete inputs (facts, data, distributions)
  - Returns (verdict: Literal["pass","warn","reject"], reason: str, details: dict)
  - No role-playing. Pure logic.

Usage:
  from frameworks.checklist import check_physical_layer, ...
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, List

Verdict = Literal["pass", "warn", "reject"]

@dataclass
class CheckResult:
    verdict: Verdict
    reason: str
    details: dict = field(default_factory=dict)

    def is_pass(self) -> bool:
        return self.verdict == "pass"

    def is_blocked(self) -> bool:
        return self.verdict == "reject"


# ══════════════════════════════════════════════
# DIRECTION LAYER (3 frameworks)
# ══════════════════════════════════════════════

def check_physical_layer(strategy_logic: str) -> CheckResult:
    """
    01 — Physical Layer (Musk)
    Check if strategy obeys A-share physical constraints.
    
    Input:
      strategy_logic: "mean_reversion", "momentum", "trend_following", "intraday", "long_short"
    
    Allowed: only mean_reversion (buy low, sell high).
    Everything else violates one or more hard constraints.
    """
    # A-share hard constraints
    CONSTRAINTS = {
        "T+1": "Cannot sell same day as buy",
        "price_limit": "±10% (±20% for STAR), extreme moves = no liquidity",
        "retail_dominant": "60%+ retail investors, emotion-driven > value-driven",
        "short_restricted": "No symmetric short selling, can only go long",
        "no_intraday": "Intraday alpha not achievable",
    }

    ALLOWED = {
        "mean_reversion": (
            "pass",
            "Mean reversion obeys all A-share constraints: T+1 irrelevant, "
            "long-only fits short restriction, retail emotions create mispricing."
        ),
        "trend_following": (
            "warn",
            "Trend following: can only follow up (no short). Risk of chasing into retail-"
            "driven rallies with no opposing force to correct overextension."
        ),
        "momentum": (
            "reject",
            "Momentum violates short_restricted: can only go long on momentum, "
            "no symmetric short. Physically asymmetric strategy in an asymmetric market. "
            "Layer 0 confirmed: momentum median -2.06%, WR 36.1%."
        ),
        "intraday": (
            "reject",
            "Intraday trading directly violates T+1 constraint."
        ),
        "long_short": (
            "reject",
            "Long-short requires short selling, which is restricted in A-shares."
        ),
    }

    logic = strategy_logic.strip().lower().replace(" ", "_")
    if logic in ALLOWED:
        v, r = ALLOWED[logic]
        return CheckResult(v, r)
    else:
        return CheckResult(
            "warn",
            f"Unknown strategy logic '{strategy_logic}'. Default to warn — must manually "
            f"verify against constraints: {CONSTRAINTS}"
        )


def check_opponent(
    signal_type: str,          # "deep_dd", "narrow_dd", "limit_down_streak"
    market_regime: str,        # "bear", "bull", "sideways"
    vr_ratio: Optional[float] = None,       # volume ratio vs 20-day MA
    dd_3day_change: Optional[float] = None,  # change in DD over last 3 days
    fall_speed_ratio: Optional[float] = None, # recent 5d avg fall / prior 20d avg fall
) -> CheckResult:
    """
    02 — Opponent Analysis (冯柳)
    Identify who's on the other side, and whether they've finished selling.

    Required 2 of 3 confirmation signals:
      - vr_ratio < 0.5 (volume shrunk to extreme)
      - dd_3day_change < 2pp (DD has stopped expanding)
      - fall_speed_ratio < 1.0 (sell-off decelerating)
    """
    # Step 1: identify opponent
    OPPONENT_MAP = {
        ("deep_dd", "bear"): ("panic_retail", "Panic selling retail — they ARE losing money"),
        ("deep_dd", "bull"): ("institution_exit", "Institutions exiting + retail bag-holding — DO NOT catch"),
        ("narrow_dd", "bear"): ("stop_loss_traders", "Swing traders hitting stops — discipline, not panic"),
        ("narrow_dd", "bull"): ("profit_taking", "Profit-taking, not capitulation — weak conviction"),
        ("limit_down_streak", "bear"): ("margin_call", "Forced liquidation of leveraged positions"),
        ("limit_down_streak", "bull"): ("unknown", "Unusual — likely stock-specific crisis"),
    }

    key = (signal_type, market_regime)
    opponent, opponent_narrative = OPPONENT_MAP.get(
        key, ("unknown", "Unknown opponent — require manual judgment")
    )

    # Step 2: check if opponent has exited
    if signal_type == "deep_dd" and market_regime == "bull":
        return CheckResult(
            "reject",
            f"Opponent: {opponent_narrative}. Deep DD in bull market = the stock is "
            f"genuinely bad, not just oversold. Layer 0: bull DD30 median -0.4%."
        )

    # Step 3: confirmation signals (only check if signal isn't auto-rejected)
    confirmations = 0
    reasons = []

    if vr_ratio is not None:
        if vr_ratio < 0.5:
            confirmations += 1
            reasons.append(f"Volume shrunk to {vr_ratio:.2f}x (extreme)")
        else:
            reasons.append(f"Volume still {vr_ratio:.2f}x — sellers may remain")

    if dd_3day_change is not None:
        if abs(dd_3day_change) < 2.0:
            confirmations += 1
            reasons.append(f"DD stabilized ({dd_3day_change:+.1f}pp over 3 days)")
        else:
            reasons.append(f"DD still expanding ({dd_3day_change:+.1f}pp) — knife still falling")

    if fall_speed_ratio is not None:
        if fall_speed_ratio < 1.0:
            confirmations += 1
            reasons.append(f"Fall speed decelerating ({fall_speed_ratio:.2f}x)")
        else:
            reasons.append(f"Fall speed {fall_speed_ratio:.2f}x — still accelerating")

    if confirmations >= 2:
        return CheckResult(
            "pass",
            f"Opponent '{opponent}' confirmed exiting: {opponent_narrative}. "
            f"Confirmation signals: {'; '.join(reasons)}"
        )
    elif confirmations == 1:
        return CheckResult(
            "warn",
            f"Opponent '{opponent}' likely still present. Only 1/3 confirmations: "
            f"{'; '.join(reasons)}. Wait or reduce size."
        )
    else:
        return CheckResult(
            "reject",
            f"Opponent '{opponent}' NOT finished selling. 0/3 confirmations: "
            f"{'; '.join(reasons)}. Do not enter."
        )


def check_reflexivity(
    market_state: str,              # "panic_zone", "normal", "overheat", "policy_distorted"
    signal_state_median: float,     # Signal historical median in this state
    signal_overall_median: float,   # Signal overall historical median
    rolling_3m_median: float,       # Signal median over last 3 months
    rolling_deviation: float,       # |rolling_3m - overall| in pp
) -> CheckResult:
    """
    03 — Reflexivity (Soros)
    Check if current market state is a decay zone for this signal.
    
    Reflexivity can only DOWNGRADE confidence, never upgrade beyond historical best.
    """
    # Step 1: state classification
    STATE_MULTIPLIER = {
        "panic_zone": ("signal_strength_zone", 1.0, "Signal historically stronger in panic"),
        "normal": ("normal", 1.0, "Signal at baseline"),
        "overheat": ("decay_zone", 0.5, "Signal decays in overheat — reduce weight"),
        "policy_distorted": ("distorted", 0.3, "Policy may override signal — minimum weight"),
    }

    if market_state not in STATE_MULTIPLIER:
        return CheckResult("warn", f"Unknown market state '{market_state}'. Assume normal.")

    zone, weight, narrative = STATE_MULTIPLIER[market_state]

    # Step 2: conditional distribution check
    state_diff = signal_state_median - signal_overall_median

    # Step 3: decay detection
    if rolling_deviation > 2.0:
        decay = "DECAY — signal medians drifting >2pp from historical baseline"
        weight = min(weight, 0.3)
    elif rolling_deviation > 1.0:
        decay = "WATCH — signal medians 1-2pp from baseline"
        weight = min(weight, 0.7)
    else:
        decay = f"Stable (deviation {rolling_deviation:.1f}pp)"

    if zone == "decay_zone" or zone == "distorted":
        return CheckResult(
            "warn",
            f"Market in {market_state}: {narrative}. Conditional median {signal_state_median:+.1%} "
            f"vs overall {signal_overall_median:+.1%} (diff {state_diff:+.1%}). "
            f"Rolling: {decay}. Effective weight: {weight:.0%}."
        )
    else:
        return CheckResult(
            "pass",
            f"Market in {market_state}: {narrative}. Conditional median {signal_state_median:+.1%} "
            f"vs overall {signal_overall_median:+.1%}. Rolling: {decay}."
        )


# ══════════════════════════════════════════════
# VALIDATION LAYER (3 frameworks)
# ══════════════════════════════════════════════

def check_quant_validation(
    signal_name: str,
    n_signals: int,
    median: float,
    win_rate: float,
    null_median: float,
    null_win_rate: float,
    ks_pvalue: float,
    profit_loss_ratio: float,
    skewness: float,
    # Bootstrap
    var_ci_width: float,
    # Walk-forward
    wf_medians: List[float],
    # Yearly
    yearly_medians: Dict[int, float],
) -> CheckResult:
    """
    04 — Quantitative Validation (Simons)
    Statistical validation gates. All thresholds are hard. No "close enough".
    
    Returns pass/warn/reject with all metrics.
    """
    MIN_SAMPLES = 200
    checks = {}

    # 1. Sample size
    checks["sample_size"] = n_signals >= MIN_SAMPLES
    if not checks["sample_size"]:
        return CheckResult("reject", f"Only {n_signals} signals (need ≥{MIN_SAMPLES})")

    # 2. Median difference
    median_diff = median - null_median
    checks["median_diff"] = median_diff > 0.02

    # 3. Win rate difference
    wr_diff = win_rate - null_win_rate
    checks["wr_diff"] = wr_diff > 0.05

    # 4. KS test
    checks["ks_test"] = ks_pvalue < 0.01

    # 5. Profit/loss ratio
    checks["pl_ratio"] = profit_loss_ratio > 1.2

    # 6. Skewness (real-world rarely positive-skewed)
    checks["skewness"] = skewness < 0.5

    # 7. Bootstrap robustness
    checks["bootstrap"] = var_ci_width < 0.03

    # 8. Walk-forward stability
    if len(wf_medians) >= 2:
        wf_range = max(wf_medians) - min(wf_medians)
        checks["wf_stable"] = wf_range < 0.03
    else:
        wf_range = 0
        checks["wf_stable"] = True

    # 9. Yearly decay trend (last 3 years)
    years = sorted(yearly_medians.keys())
    if len(years) >= 3:
        recent = [yearly_medians[y] for y in years[-3:]]
        decay_trend = recent[-1] < recent[-2] < recent[0]  # monotonically declining
        checks["yearly_decay"] = not decay_trend
    else:
        decay_trend = False
        checks["yearly_decay"] = True

    # Aggregate
    failed = [k for k, v in checks.items() if not v]
    total = len(checks)

    lines = [
        f"═══ {signal_name} (n={n_signals}) ═══",
        f"  Median: {median:+.2%} vs null {null_median:+.2%}  diff={median_diff:+.2%}  {'PASS' if checks['median_diff'] else 'FAIL'}",
        f"  WR:     {win_rate:.1%} vs null {null_win_rate:.1%}  diff={wr_diff:+.1%}  {'PASS' if checks['wr_diff'] else 'FAIL'}",
        f"  KS p={ks_pvalue:.4f}  {'PASS' if checks['ks_test'] else 'FAIL'}  |  PL ratio={profit_loss_ratio:.2f}  {'PASS' if checks['pl_ratio'] else 'FAIL'}",
        f"  Skew={skewness:+.3f}  {'PASS' if checks['skewness'] else 'FAIL'}  |  VaR CI width={var_ci_width:.2%}  {'PASS' if checks['bootstrap'] else 'FAIL'}",
        f"  WF range={wf_range:.2%}  {'PASS' if checks['wf_stable'] else 'FAIL'}  |  Yearly decay={'NO' if checks['yearly_decay'] else 'YES — DANGER'}",
        f"  Failed: {len(failed)}/{total} — {failed if failed else 'none'}",
    ]

    if len(failed) == 0:
        return CheckResult("pass", "\n".join(lines))
    elif len(failed) <= 2:
        return CheckResult("warn", "\n".join(lines))
    else:
        return CheckResult("reject", "\n".join(lines))


def check_safety_gate(
    signal_name: str,
    n_signals: int,
    # Walk-forward
    wf_medians: List[float],
    # Yearly trend
    yearly_medians: Dict[int, float],
    # Tail
    left_tail_1: float,
    # Bull/bear consistency
    bull_median: Optional[float],
    bear_median: Optional[float],
    # Data integrity
    data_clean: bool,  # Has data contamination been ruled out?
    # Execution feasibility
    unexecutable_pct: float,  # % of signals that couldn't be executed (limit-down, suspended, ST)
    # Signal frequency
    daily_signal_rate: float,  # Average signals per trading day
) -> CheckResult:
    """
    05 — Safety Gate (Buffett)
    8-item checklist. Any hard-fail = reject. Any warning = noted.
    This framework can ONLY REJECT, never approve.
    """
    gates = {}

    # Gate 1: Walk-forward stability
    if len(wf_medians) >= 2:
        wf_range = max(wf_medians) - min(wf_medians)
        gates["wf_stability"] = ("pass" if wf_range < 0.03 else "fail", f"WF range={wf_range:.2%}")
    else:
        gates["wf_stability"] = ("warn", "Insufficient WF data")

    # Gate 2: Yearly decay trend
    years = sorted(yearly_medians.keys())
    if len(years) >= 3:
        recent = [yearly_medians[y] for y in years[-3:]]
        decaying = recent[-1] < recent[-2] < recent[0]
        gates["yearly_decay"] = (
            "fail" if decaying else "pass",
            f"Recent 3y: {recent} — {'DECAYING' if decaying else 'stable'}"
        )
    else:
        gates["yearly_decay"] = ("warn", "Insufficient yearly data")

    # Gate 3: Left tail extremes
    gates["left_tail"] = (
        "fail" if left_tail_1 < -0.30 else "pass",
        f"Left tail 1%: {left_tail_1:+.1%}"
    )

    # Gate 4: Sample size (200 minimum)
    gates["sample_size"] = (
        "fail" if n_signals < 200 else "pass",
        f"n={n_signals}"
    )

    # Gate 5: Bull/bear consistency
    if bull_median is not None and bear_median is not None:
        same_sign = (bull_median > 0) == (bear_median > 0)
        gates["bull_bear_consistency"] = (
            "pass" if same_sign else "warn",
            f"Bull median={bull_median:+.1%}, Bear median={bear_median:+.1%} — "
            f"{'same direction' if same_sign else 'OPPOSITE DIRECTIONS — needs regime filter'}"
        )
    else:
        gates["bull_bear_consistency"] = ("warn", "Bull/bear data unavailable")

    # Gate 6: Data contamination
    gates["data_clean"] = (
        "fail" if not data_clean else "pass",
        "Data confirmed clean" if data_clean else "DATA CONTAMINATION SUSPECTED"
    )

    # Gate 7: Execution feasibility
    gates["execution"] = (
        "fail" if unexecutable_pct > 0.05 else "pass",
        f"{unexecutable_pct:.1%} signals unexecutable"
    )

    # Gate 8: Signal frequency
    if daily_signal_rate > 20:
        gates["frequency"] = ("warn", f"{daily_signal_rate:.0f} signals/day — decision burden too high")
    elif daily_signal_rate > 50:
        gates["frequency"] = ("fail", f"{daily_signal_rate:.0f} signals/day — impossible to manage")
    else:
        gates["frequency"] = ("pass", f"{daily_signal_rate:.1f} signals/day")

    fails = [k for k, (v, _) in gates.items() if v == "fail"]
    warns = [k for k, (v, _) in gates.items() if v == "warn"]

    details = "\n".join(f"  [{v.upper():4s}] {k}: {r}" for k, (v, r) in gates.items())
    header = f"Safety Gate — {signal_name} (n={n_signals}): {len(fails)} fails, {len(warns)} warnings"

    if fails:
        return CheckResult("reject", f"{header}\n{details}")
    elif warns:
        return CheckResult("warn", f"{header}\n{details}")
    else:
        return CheckResult("pass", f"{header}\n{details}")


def check_incremental_audit(
    filter_name: str,
    baseline_median: float,
    filtered_median: float,
    baseline_n: int,
    filtered_n: int,
) -> CheckResult:
    """
    06 — Incremental Audit (OpenClaw)
    Does adding this filter improve the signal?
    
    Thresholds:
      median gain > 2pp → meaningful
      median gain 1-2pp → marginal
      median gain < 1pp → noise → REJECT
      sample loss > 30% → too aggressive → REJECT
    
    Must test filters ONE AT A TIME — no bundling.
    """
    median_gain = filtered_median - baseline_median
    sample_loss = 1 - (filtered_n / baseline_n) if baseline_n > 0 else 0

    if sample_loss > 0.30:
        return CheckResult(
            "reject",
            f"Filter '{filter_name}' kills {sample_loss:.0%} of samples "
            f"({baseline_n} → {filtered_n}). Too aggressive — max 30% loss allowed."
        )

    if median_gain < 0.01:
        return CheckResult(
            "reject",
            f"Filter '{filter_name}' adds {median_gain:+.2%} median gain — below 1pp noise floor. "
            f"Baseline={baseline_median:+.1%}, Filtered={filtered_median:+.1%}. REJECT as noise."
        )
    elif median_gain < 0.02:
        return CheckResult(
            "warn",
            f"Filter '{filter_name}' marginal gain {median_gain:+.2%}. "
            f"Baseline={baseline_median:+.1%} → Filtered={filtered_median:+.1%}. "
            f"Sample loss: {sample_loss:.0%}. Weigh carefully."
        )
    else:
        return CheckResult(
            "pass",
            f"Filter '{filter_name}' meaningful gain {median_gain:+.2%}. "
            f"Baseline={baseline_median:+.1%} → Filtered={filtered_median:+.1%}. "
            f"Sample loss: {sample_loss:.0%}. Approved."
        )


# ══════════════════════════════════════════════
# PORTFOLIO LAYER (3 frameworks)
# ══════════════════════════════════════════════

def check_position_sizing(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    left_tail_5: float,
    account_total: float,
    current_total_exposure: float,  # current % deployed
    same_sector_count: int,         # holdings in same sector
    n_signals: int,
    MAX_SINGLE: float = 0.25,       # single position cap
    MAX_TOTAL: float = 0.80,        # total exposure cap
    MAX_SECTOR: int = 1,            # max per sector
) -> CheckResult:
    """
    07 — Position Sizing (葛卫东)
    Kelly formula → tail penalty → single/total caps → sector cap.

    Returns recommended position size as % of account.
    """
    if n_signals < 200:
        return CheckResult(
            "warn",
            f"Only {n_signals} signals — Kelly unreliable. Use fixed 5% position.",
            {"kelly_raw": None, "kelly_tail_adj": None, "recommended": 0.05, "yuan": account_total * 0.05}
        )

    lose_rate = 1 - win_rate
    abs_avg_loss = abs(avg_loss)

    # Step 1: Kelly
    if avg_win > 0 and abs_avg_loss > 0:
        kelly = (win_rate * avg_win - lose_rate * abs_avg_loss) / (avg_win * abs_avg_loss)
        kelly = max(0, min(kelly, 1.0))  # clamp to [0, 1]
    else:
        return CheckResult("reject", "Invalid win/loss averages for Kelly calculation.")

    # Step 2: Tail penalty
    abs_left_tail = abs(left_tail_5)
    if abs_avg_loss > 0:
        tail_penalty = abs_left_tail / abs_avg_loss
        kelly_adj = kelly / tail_penalty
    else:
        kelly_adj = kelly

    # Step 3: Caps
    recommended = min(kelly_adj, MAX_SINGLE)

    # Step 4: Total exposure check
    if current_total_exposure + recommended > MAX_TOTAL:
        recommended = MAX_TOTAL - current_total_exposure
        cap_hit = f"Total exposure cap ({MAX_TOTAL:.0%}) limits position to {recommended:.1%}"
    else:
        cap_hit = "Within total exposure limit"

    # Step 5: Sector check
    if same_sector_count >= MAX_SECTOR:
        return CheckResult(
            "reject",
            f"Sector already has {same_sector_count} position(s) — max {MAX_SECTOR}. Cannot add."
        )

    yuan_amount = account_total * recommended

    # Risk check: can you survive 2 consecutive tail events?
    tail_loss = yuan_amount * abs_left_tail
    two_hit = account_total - 2 * tail_loss
    if two_hit < account_total * 0.5:
        survival = (f"WARNING: 2 consecutive tail events (-{abs_left_tail:.1%} each) = "
                     f"¥{yuan_amount:,.0f} × 2 = ¥{2*tail_loss:,.0f} loss, "
                     f"leaving ¥{two_hit:,.0f} ({two_hit/account_total:.0%})")
    else:
        survival = "Survives 2 consecutive tail events"

    details = {
        "kelly_raw": kelly,
        "kelly_tail_adj": kelly_adj,
        "recommended": recommended,
        "yuan_amount": yuan_amount,
        "cap_note": cap_hit,
        "survival": survival,
    }

    reason = (
        f"Kelly raw={kelly:.1%} → tail-adj={kelly_adj:.1%} → capped={recommended:.1%} "
        f"(¥{yuan_amount:,.0f}). {cap_hit}. {survival}."
    )

    return CheckResult("pass", reason, details)


def check_portfolio_correlation(
    holdings: List[dict],  # [{"code": "000001", "weight": 0.15, "signal_source": "deep_dd30", "sector": "新能源"}, ...]
    new_candidate: dict,
) -> CheckResult:
    """
    08 — Portfolio Correlation (Dalio)
    Check signal source overlap, sector concentration, and correlation.
    
    3 checks: signal source diversity, sector concentration, pairwise correlation.
    """
    # Check 1: Signal source overlap
    sources = {}
    for h in holdings:
        src = h.get("signal_source", "unknown")
        sources[src] = sources.get(src, 0) + h.get("weight", 0)

    new_src = new_candidate.get("signal_source", "unknown")
    total_src_weight = sources.get(new_src, 0) + new_candidate.get("weight", 0)

    issues = []

    if total_src_weight > 0.80:
        issues.append(f"Signal source '{new_src}' overlap: {total_src_weight:.0%} of portfolio (limit 80%)")
    elif total_src_weight > 0.50:
        issues.append(f"Signal source '{new_src}' concentration: {total_src_weight:.0%} (warn >50%)")

    # Check 2: Sector concentration
    new_sector = new_candidate.get("sector", "unknown")
    sector_weight = sum(h.get("weight", 0) for h in holdings if h.get("sector") == new_sector)
    total_sector_weight = sector_weight + new_candidate.get("weight", 0)

    if total_sector_weight > 0.30:
        issues.append(f"Sector '{new_sector}' concentration: {total_sector_weight:.0%} of portfolio (limit 30%)")

    # Check 3: Same-sector count
    same_sector_count = sum(1 for h in holdings if h.get("sector") == new_sector)
    if same_sector_count >= 1:
        issues.append(f"Sector '{new_sector}' already has {same_sector_count} position(s)")

    if len(issues) >= 2:
        return CheckResult(
            "warn",
            f"Portfolio correlation issues: {'; '.join(issues)}. Consider reducing or skipping."
        )
    elif len(issues) == 1:
        return CheckResult("warn", f"Portfolio correlation: {issues[0]}")
    else:
        return CheckResult("pass", "No signal source or sector concentration issues.")


def check_tail_risk(
    left_tail_5: float,
    left_tail_1: float,
    account_total: float,
    position_yuan: float,
    daily_volume_avg: float,       # Average daily trading volume in yuan for this stock
    is_st: bool,
    consecutive_limit_down_days: int,  # Recent consecutive limit-down days (past 30d)
) -> CheckResult:
    """
    09 — Tail Risk (Taleb)
    Check if you can survive worst-case scenarios that history didn't cover.
    
    Outputs hard stop-loss rules. These rules CANNOT be overridden.
    """
    issues = []

    # Check 1: Historical left tail impact
    tail5_loss = position_yuan * abs(left_tail_5)
    tail5_pct = abs(left_tail_5)
    tail1_loss = position_yuan * abs(left_tail_1)

    issues.append(f"Left tail 5%: -{tail5_pct:.1%} = ¥{tail5_loss:,.0f} loss")
    issues.append(f"Left tail 1%: -{abs(left_tail_1):.1%} = ¥{tail1_loss:,.0f} loss")

    # Check 2: Consecutive tail events
    two_hit_remaining = account_total - 2 * tail5_loss
    three_hit_remaining = account_total - 3 * tail5_loss

    if three_hit_remaining < account_total * 0.50:
        issues.append(
            f"CRITICAL: 3 consecutive tail-5% events leave ¥{three_hit_remaining:,.0f} "
            f"({three_hit_remaining/account_total:.0%}). Position too large."
        )
    elif two_hit_remaining < account_total * 0.50:
        issues.append(
            f"WARNING: 2 consecutive tail-5% events leave ¥{two_hit_remaining:,.0f} "
            f"({two_hit_remaining/account_total:.0%})."
        )

    # Check 3: Liquidity
    if is_st:
        issues.append("ST stock — FORBIDDEN.")
    if daily_volume_avg < 5_000_000:
        issues.append(f"Daily volume ¥{daily_volume_avg:,.0f} < ¥5M — liquidity insufficient.")

    # Check 4: Limit-down risk
    if consecutive_limit_down_days >= 2:
        issues.append(
            f"{consecutive_limit_down_days} consecutive limit-down days in past 30d — high lock-up risk."
        )

    # Hard stop-loss rules (always included in output)
    hard_rules = (
        "HARD STOP-LOSS RULES (cannot be overridden):\n"
        "  - Single-day drawdown > 5%: reduce to half position, unconditionally\n"
        "  - Single trade loss > 15%: close entire position, unconditionally\n"
        "  - Monthly drawdown > 10%: stop trading for 1 week\n"
        "  - Quarterly drawdown > 20%: stop trading for 1 month + full review"
    )

    fatal = [i for i in issues if "CRITICAL" in i or "FORBIDDEN" in i]

    if fatal:
        return CheckResult(
            "reject",
            f"Tail risk fatal: {'; '.join(fatal)}\n\n{hard_rules}"
        )
    elif len(issues) > 2:
        return CheckResult(
            "warn",
            f"Tail risk concerns:\n  " + "\n  ".join(issues) + f"\n\n{hard_rules}"
        )
    else:
        return CheckResult(
            "pass",
            f"Tail risk acceptable:\n  " + "\n  ".join(issues) + f"\n\n{hard_rules}"
        )


# ══════════════════════════════════════════════
# EXECUTION LAYER (2 frameworks)
# ══════════════════════════════════════════════

def check_actuarial_synthesis(
    framework_results: List[tuple],  # [("physical_layer", "pass", "reason"), ...]
) -> CheckResult:
    """
    10 — Actuarial Synthesis (精算师)
    Aggregate all framework outputs into a final verdict.
    
    Rules:
      - Any reject = REJECT (no voting out)
      - 0 warns = pass, high confidence
      - 1-2 warns = warn, medium confidence
      - 3+ warns = reject, low confidence
    """
    rejects = [(name, reason) for name, verdict, reason in framework_results if verdict == "reject"]
    warns = [(name, reason) for name, verdict, reason in framework_results if verdict == "warn"]
    passes = [(name, reason) for name, verdict, reason in framework_results if verdict == "pass"]

    summary_parts = [f"{len(passes)} pass, {len(warns)} warn, {len(rejects)} reject"]

    if rejects:
        reject_names = [name for name, _ in rejects]
        return CheckResult(
            "reject",
            f"SYNTHESIS: REJECT — {', '.join(reject_names)} rejected. "
            f"A single fatal flaw cannot be overridden by other passes. Summary: {', '.join(summary_parts)}."
        )

    if len(warns) >= 3:
        return CheckResult(
            "reject",
            f"SYNTHESIS: REJECT — {len(warns)} warnings exceeds threshold. "
            f"Summary: {', '.join(summary_parts)}."
        )

    if len(warns) == 0:
        return CheckResult(
            "pass",
            f"SYNTHESIS: PASS — all {len(passes)} frameworks passed. "
            f"High confidence. Summary: {', '.join(summary_parts)}."
        )
    else:
        warn_names = [name for name, _ in warns]
        # Check if warnings are same-direction (all downside = stronger signal)
        return CheckResult(
            "warn",
            f"SYNTHESIS: WARN — {len(warns)} framework(s) flagged: {', '.join(warn_names)}. "
            f"Medium confidence. Reduce position by 30-50%. Summary: {', '.join(summary_parts)}."
        )


def check_execution_iron_law(
    synthesis_verdict: str,       # "pass", "warn", "reject"
    hard_stop_triggered: bool,    # Has a hard stop-loss been triggered?
    recent_violations: int,        # Number of iron law violations this quarter
    signal_frequency_spike: bool,  # Has signal frequency suddenly doubled?
) -> CheckResult:
    """
    11 — Execution Iron Law (交易员)
    Enforce discipline. Translate analysis into action: DO or DON'T.
    
    This is the LAST framework run. It doesn't analyze data — it enforces
    conclusions already reached by the other 10 frameworks.
    """
    # Iron Law 0: Too many violations → system breakdown
    if recent_violations > 2:
        return CheckResult(
            "reject",
            f"SYSTEM BREAKDOWN: {recent_violations} iron law violations this quarter. "
            f"Stop all trading. Full review required before resuming."
        )

    # Iron Law 1: Synthesis reject = no trade
    if synthesis_verdict == "reject":
        return CheckResult(
            "reject",
            "IRON LAW: Synthesis rejected. NO TRADE. Do not 'look again'. Do not 'this time is different'."
        )

    # Iron Law 2: Hard stop triggered = execute immediately
    if hard_stop_triggered:
        return CheckResult(
            "reject",
            "IRON LAW: Hard stop-loss triggered. Execute NOW. Do not wait for rebound. "
            "Modify the rule BEFORE the NEXT trade, not THIS trade."
        )

    # Iron Law 3: Signal frequency spike → pause
    if signal_frequency_spike:
        return CheckResult(
            "warn",
            "IRON LAW: Signal frequency spike detected. Pause 3 trading days. "
            "Verify no data issues or market anomalies."
        )

    # Iron Law 4: Synthesis warn → reduce position
    if synthesis_verdict == "warn":
        return CheckResult(
            "warn",
            "IRON LAW: Synthesis warned — REDUCE position by 30-50%. "
            "Do not 'round up' because the opportunity looks good."
        )

    # All clear
    return CheckResult(
        "pass",
        "IRON LAW: All checks passed. Proceed with trade at recommended size. "
        "Hard stops are armed. Trust the system."
    )
