"""
Conflict Arbitrator — Multi-strategy voting with historical accuracy weighting.

Core principle (from OpenClaw):
  "Don't trust any single strategy. Trust the CONFLICT between strategies."
  
When 4 independent frameworks agree → act with confidence.
When they disagree → the disagreement IS the signal.

Architecture:
  Strategy votes → Weighted by historical accuracy → Conflict resolution → Position size
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime, timedelta
import numpy as np


class Verdict(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Vote:
    """A single strategy's vote on a stock or market condition."""
    strategy: str
    verdict: Verdict
    confidence: float     # 0.0 - 1.0, how sure is this strategy?
    reason: str
    # Historical accuracy in current market state (populated from backtest)
    historical_accuracy: float = 0.50  # default: no better than coin flip


@dataclass 
class StrategyProfile:
    """A strategy's identity: what it does, what kills it, its track record."""
    name: str
    description: str
    input_type: str       # "price", "flow", "fundamental", "macro"
    death_conditions: List[str] = field(default_factory=list)
    
    # Historical accuracy by market regime
    accuracy_by_regime: Dict[str, float] = field(default_factory=dict)
    # Overall stats
    overall_accuracy: float = 0.50
    total_predictions: int = 0
    
    def is_alive(self, context: dict) -> tuple:
        """Check if strategy's death conditions are triggered.
        Returns (alive: bool, reason: str)"""
        for cond in self.death_conditions:
            if _eval_condition(cond, context):
                return False, f"Death condition triggered: {cond}"
        return True, ""


@dataclass
class ArbitrationResult:
    """Final output of the arbitrator."""
    verdict: Verdict
    position_pct: float      # 0.0 - 1.0, recommended position size
    confidence: float        # overall confidence
    votes: List[Vote]
    consensus_level: str     # "unanimous", "strong", "weak", "split", "deadlocked"
    reasoning: str
    
    def summary(self) -> str:
        buy_votes = sum(1 for v in self.votes if v.verdict == Verdict.BUY)
        sell_votes = sum(1 for v in self.votes if v.verdict == Verdict.SELL)
        hold_votes = sum(1 for v in self.votes if v.verdict == Verdict.HOLD)
        
        lines = [
            f"Arbitration: {self.consensus_level.upper()}",
            f"  Verdict: {self.verdict.value.upper()} @ {self.position_pct:.0%} position",
            f"  Confidence: {self.confidence:.0%}",
            f"  Votes: {buy_votes}B / {sell_votes}S / {hold_votes}H",
            f"  Reason: {self.reasoning}",
        ]
        for v in self.votes:
            acc = f"(acc:{v.historical_accuracy:.0%})" if v.historical_accuracy != 0.50 else ""
            lines.append(f"    [{v.verdict.value.upper():4s}] {v.strategy:<20s} "
                        f"conf={v.confidence:.0%} {acc} — {v.reason[:60]}")
        return "\n".join(lines)


class Arbitrator:
    """
    Multi-strategy conflict arbitrator.
    
    Usage:
        arb = Arbitrator()
        arb.register(strategy1_profile, accuracy_data)
        result = arb.arbitrate(votes, market_context)
    """

    def __init__(self, max_single_position: float = 0.25):
        self.profiles: Dict[str, StrategyProfile] = {}
        self.max_single_position = max_single_position

    def register(self, profile: StrategyProfile):
        self.profiles[profile.name] = profile

    def arbitrate(self, votes: List[Vote], context: dict = None) -> ArbitrationResult:
        """
        Take votes from all strategies, weight by accuracy, resolve conflicts.
        """
        if context is None:
            context = {}

        # ── Filter: only alive strategies vote ──
        alive_votes = []
        dead_reasons = []
        for v in votes:
            profile = self.profiles.get(v.strategy)
            if profile:
                alive, reason = profile.is_alive(context)
                if not alive:
                    dead_reasons.append(f"{v.strategy}: {reason}")
                    continue
            # Attach historical accuracy
            if profile and context.get("regime") in profile.accuracy_by_regime:
                v.historical_accuracy = profile.accuracy_by_regime[context["regime"]]
            elif profile:
                v.historical_accuracy = profile.overall_accuracy
            alive_votes.append(v)

        if len(alive_votes) < 2:
            return ArbitrationResult(
                verdict=Verdict.HOLD,
                position_pct=0.0,
                confidence=0.0,
                votes=alive_votes,
                consensus_level="insufficient",
                reasoning=f"Only {len(alive_votes)} alive strategies. Dead: {dead_reasons}",
            )

        # ── Weighted voting ──
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0

        for v in alive_votes:
            weight = v.confidence * v.historical_accuracy
            total_weight += weight
            if v.verdict == Verdict.BUY:
                buy_score += weight
            elif v.verdict == Verdict.SELL:
                sell_score += weight
            # HOLD doesn't add to either side

        if total_weight == 0:
            return ArbitrationResult(
                verdict=Verdict.HOLD, position_pct=0.0, confidence=0.0,
                votes=alive_votes, consensus_level="deadlocked",
                reasoning="All strategies have zero weight.",
            )

        buy_pct = buy_score / total_weight
        sell_pct = sell_score / total_weight

        # ── P0-1 FIX: Pure weighted score, no head-count ──
        hold_pct = 1.0 - buy_pct - sell_pct
        
        buy_count = sum(1 for v in alive_votes if v.verdict == Verdict.BUY)
        sell_count = sum(1 for v in alive_votes if v.verdict == Verdict.SELL)
        total = len(alive_votes)

        # Consensus determined by weighted score dominance
        if buy_pct >= 0.80:
            consensus = "unanimous"
            verdict = Verdict.BUY
            position = self.max_single_position
            confidence = buy_pct
            reason = f"Overwhelming BUY ({buy_pct:.0%} weighted). {buy_count}/{total} strategies."
        elif buy_pct >= 0.60:
            consensus = "strong"
            verdict = Verdict.BUY
            position = self.max_single_position * 0.75
            confidence = buy_pct
            reason = f"Strong BUY ({buy_pct:.0%} weighted). {buy_count}/{total} strategies."
        elif buy_pct > sell_pct and buy_pct >= 0.40:
            consensus = "weak"
            verdict = Verdict.BUY
            position = self.max_single_position * 0.40
            confidence = buy_pct * 0.7
            reason = f"Weak BUY ({buy_pct:.0%} weighted). {buy_count}/{total} strategies."
        elif sell_pct >= 0.80:
            consensus = "unanimous"
            verdict = Verdict.SELL
            position = 0.0
            confidence = sell_pct
            reason = f"Overwhelming SELL ({sell_pct:.0%} weighted). {sell_count}/{total} strategies."
        elif sell_pct >= 0.60:
            consensus = "strong"
            verdict = Verdict.SELL
            position = 0.0
            confidence = sell_pct
            reason = f"Strong SELL ({sell_pct:.0%} weighted). {sell_count}/{total} strategies."
        elif abs(buy_pct - sell_pct) < 0.15:
            consensus = "deadlocked"
            verdict = Verdict.HOLD
            position = 0.0
            confidence = 0.0
            reason = (f"Weighted scores too close ({buy_pct:.0%} vs {sell_pct:.0%}). "
                      f"No edge. {buy_count}B/{sell_count}S/{total-buy_count-sell_count}H.")
        else:
            consensus = "split"
            verdict = Verdict.HOLD
            position = 0.0
            confidence = 0.0
            reason = f"No clear direction ({buy_pct:.0%}B/{sell_pct:.0%}S/{hold_pct:.0%}H weighted)."

        return ArbitrationResult(
            verdict=verdict,
            position_pct=position,
            confidence=confidence,
            votes=alive_votes,
            consensus_level=consensus,
            reasoning=reason + (f" Dead: {dead_reasons}" if dead_reasons else ""),
        )


# ═══════════════════════════════════════════
# Pre-built strategy profiles
# ═══════════════════════════════════════════

def build_default_profiles() -> List[StrategyProfile]:
    """Build the four default strategy profiles with known accuracy data."""
    
    p1 = StrategyProfile(
        name="price_action",
        description="NorthMargin: DD>=25% + north inflow + margin decline",
        input_type="price",
        death_conditions=[
            "regime == 'bull'",                    # Doesn't work in bull markets
            "wf_decay_detected == True",            # Walk-forward decay
        ],
        accuracy_by_regime={
            "severe_bear": 0.79,
            "bear": 0.52,
            "recovery": 0.62,
            "bull": 0.50,
        },
        overall_accuracy=0.57,
    )

    p2 = StrategyProfile(
        name="flow_analysis",
        description="Capital flow: who's buying/selling, opponent identification",
        input_type="flow",
        death_conditions=[
            "policy_intervention == True",          # Government intervention distorts flows
        ],
        accuracy_by_regime={},  # Not yet validated
        overall_accuracy=0.50,
    )

    p3 = StrategyProfile(
        name="fundamental_value",
        description="Extreme fundamental filter: np>0+debt<50 vs np<-30+debt>70",
        input_type="fundamental",
        death_conditions=[
            "earnings_season == False",             # Only reliable during earnings
        ],
        accuracy_by_regime={},
        overall_accuracy=0.53,  # From Test 4: safe vs risky delta +1.3pp
    )

    p4 = StrategyProfile(
        name="macro_cycle",
        description="Marks cycle + margin gradient + PE percentile (2yr)",
        input_type="macro",
        death_conditions=[
            "policy_distortion == True",             # Policy overrides cycle
        ],
        accuracy_by_regime={
            "bull": 0.65,   # PE 2yr prediction during bull
            "bear": 0.60,
        },
        overall_accuracy=0.60,
    )

    return [p1, p2, p3, p4]


def _eval_condition(condition: str, context: dict) -> bool:
    """Evaluate a simple death condition against context."""
    try:
        expr = condition
        for key, val in context.items():
            if isinstance(val, str):
                expr = expr.replace(key, f"'{val}'")
            elif isinstance(val, bool):
                expr = expr.replace(key, str(val))
            else:
                expr = expr.replace(key, str(val))
        return bool(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return False


# ═══════════════════════════════════════════
# P0-2: Portfolio-level total exposure manager
# ═══════════════════════════════════════════

class PortfolioManager:
    """Ensures total exposure across all positions never exceeds limits."""
    
    def __init__(self, max_total_exposure: float = 0.80, max_per_strategy: float = 0.30):
        self.max_total_exposure = max_total_exposure
        self.max_per_strategy = max_per_strategy
        self.positions = []
    
    def can_add(self, stock: str, strategy: str, desired_weight: float) -> tuple:
        strategy_total = sum(w for _, s, w in self.positions if s == strategy)
        if strategy_total + desired_weight > self.max_per_strategy:
            adj = max(0, self.max_per_strategy - strategy_total)
            return False, adj, f"Strategy '{strategy}' cap {self.max_per_strategy:.0%}"
        total = sum(w for _, _, w in self.positions)
        if total + desired_weight > self.max_total_exposure:
            adj = max(0, self.max_total_exposure - total)
            return False, adj, f"Total exposure cap {self.max_total_exposure:.0%}"
        return True, desired_weight, ""
    
    def add(self, stock: str, strategy: str, weight: float):
        ok, adj, reason = self.can_add(stock, strategy, weight)
        if adj > 0:
            self.positions.append((stock, strategy, adj))
        return ok, adj, reason
    
    @property
    def total(self) -> float:
        return sum(w for _, _, w in self.positions)


# ═══════════════════════════════════════════
# P0-3: Independent Circuit Breaker
# ═══════════════════════════════════════════

class CircuitBreaker:
    """Independent safety switch. No strategy vote can override."""
    
    def __init__(self, max_drawdown: float = -0.20, max_monthly: float = -0.10,
                 max_consecutive: int = 5, cooldown_days: int = 5):
        self.max_drawdown = max_drawdown
        self.max_monthly = max_monthly
        self.max_consecutive = max_consecutive
        self.cooldown_days = cooldown_days
        self.returns = []
        self.consecutive = 0
        self.tripped = False
        self.reason = ""
        self.trip_time = None
        self.peak = 1.0
        self.current = 1.0
    
    def update(self, daily_return: float):
        self.returns.append(daily_return)
        self.current *= (1 + daily_return)
        self.peak = max(self.peak, self.current)
        self.consecutive = self.consecutive + 1 if daily_return < 0 else 0
    
    def check(self) -> tuple:
        if self.tripped:
            if self.trip_time and (datetime.now() - self.trip_time).days >= self.cooldown_days:
                self.tripped = False
                self.reason = ""
                self.trip_time = None
                self.consecutive = 0
                return False, "Cooldown elapsed."
            return True, self.reason
        
        dd = (self.current / self.peak) - 1
        if dd <= self.max_drawdown:
            self._trip(f"Drawdown {dd:.1%} > {self.max_drawdown:.0%}")
            return True, self.reason
        if self.consecutive >= self.max_consecutive:
            self._trip(f"{self.max_consecutive} consecutive losses")
            return True, self.reason
        if len(self.returns) >= 21:
            m = np.prod([1+r for r in self.returns[-21:]]) - 1
            if m <= self.max_monthly:
                self._trip(f"Monthly loss {m:.1%} > {self.max_monthly:.0%}")
                return True, self.reason
        return False, "OK"
    
    def _trip(self, reason: str):
        self.tripped = True
        self.reason = reason
        self.trip_time = datetime.now()


# ═══════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════
if __name__ == "__main__":
    arb = Arbitrator()
    for p in build_default_profiles():
        arb.register(p)

    # P0-1 Test: high-accuracy BUY vs low-accuracy SELL
    votes = [
        Vote("price_action", Verdict.BUY, 0.80, "Trend healthy", historical_accuracy=0.79),
        Vote("macro_cycle", Verdict.BUY, 0.70, "Cycle supports", historical_accuracy=0.65),
        Vote("flow_analysis", Verdict.SELL, 0.30, "No data", historical_accuracy=0.50),
        Vote("fundamental_value", Verdict.SELL, 0.20, "Off season", historical_accuracy=0.50),
    ]
    context = {"regime": "severe_bear", "policy_intervention": False,
               "wf_decay_detected": False, "earnings_season": False,
               "policy_distortion": False, "consecutive_losses": 0}

    result = arb.arbitrate(votes, context)
    print(result.summary())
    print(f"\n  P0-1: Two high-accuracy BUY strategies are NOT overruled by two low-accuracy SELL.")
    print(f"  Weighted score decides: {result.verdict.value.upper()}, position={result.position_pct:.0%}")

    # P0-2 Test: portfolio cap
    pm = PortfolioManager()
    ok1, w1, _ = pm.add("stock_A", "price_action", 0.25)
    ok2, w2, _ = pm.add("stock_B", "price_action", 0.25)
    print(f"\n  P0-2: Added A(25%) + B(25%) via same strategy = {pm.total:.0%}")
    print(f"  Stock B was {'allowed' if ok2 else 'capped'} at {w2:.0%} (strategy cap {pm.max_per_strategy:.0%})")

    # P0-3 Test: circuit breaker
    cb = CircuitBreaker()
    for _ in range(5):
        cb.update(-0.02)
    tripped, reason = cb.check()
    print(f"\n  P0-3: After 5 consecutive -2% days: {'TRIPPED' if tripped else 'OK'} — {reason}")
