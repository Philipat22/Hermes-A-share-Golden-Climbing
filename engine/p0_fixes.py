"""
P0 Fixes — Three critical bugs resolved.

P0-1: Pure weighted voting (no head-count contradiction)
P0-2: Portfolio-level total exposure cap
P0-3: Independent circuit breaker (no strategy can override)
"""

# ── P0-1: Replace head-count with pure weighted score ──

P0_1_CONFLICT_RESOLUTION = """
# OLD (buggy — mixed weighted scores with head-count):  
    if buy_count == total:          # head-count
        ...
    elif buy_count == sell_count:   # head-count
        ...

# NEW (fixed — pure weighted score):  
    buy_pct = buy_score / total_weight
    sell_pct = sell_score / total_weight
    
    # Consensus based on weighted score dominance, not head count
    if buy_pct >= 0.80:
        consensus = "unanimous"
        verdict = Verdict.BUY
        position = self.max_single_position
        confidence = buy_pct
    elif buy_pct >= 0.60:
        consensus = "strong"
        verdict = Verdict.BUY
        position = self.max_single_position * 0.75
        confidence = buy_pct
    elif buy_pct > sell_pct and buy_pct >= 0.40:
        consensus = "weak" 
        verdict = Verdict.BUY
        position = self.max_single_position * 0.40
        confidence = buy_pct * 0.7
    elif sell_pct >= 0.80:
        consensus = "unanimous"
        verdict = Verdict.SELL
        position = 0.0
        confidence = sell_pct
    elif sell_pct >= 0.60:
        consensus = "strong"
        verdict = Verdict.SELL
        position = 0.0
        confidence = sell_pct
    elif abs(buy_pct - sell_pct) < 0.15:
        consensus = "deadlocked"
        verdict = Verdict.HOLD
        position = 0.0
        confidence = 0.0
        reason = f"Scores too close ({buy_pct:.0%} vs {sell_pct:.0%}). No edge."
    else:
        consensus = "split"
        verdict = Verdict.HOLD
        position = 0.0
        confidence = 0.0

# Now: two 79%-accurate strategies voting BUY with high confidence
#      will NOT be overruled by two 50%-accurate strategies voting SELL.
# The weighted score correctly reflects the quality gap.
"""


# ── P0-2: Portfolio-level total exposure cap ──

class PortfolioManager:
    """
    Ensures total exposure across all positions never exceeds limits.
    Called AFTER arbitrator for each individual stock.
    """
    
    def __init__(self, max_total_exposure: float = 0.80, max_per_strategy: float = 0.30):
        self.max_total_exposure = max_total_exposure
        self.max_per_strategy = max_per_strategy
        self.positions = []  # [(stock, strategy, weight), ...]
    
    def can_add(self, stock: str, strategy: str, desired_weight: float) -> tuple:
        """
        Check if adding this position respects all constraints.
        Returns (allowed: bool, adjusted_weight: float, reason: str)
        """
        # Check per-strategy cap
        strategy_total = sum(w for _, s, w in self.positions if s == strategy)
        if strategy_total + desired_weight > self.max_per_strategy:
            adjusted = max(0, self.max_per_strategy - strategy_total)
            return False, adjusted, (
                f"Strategy '{strategy}' would exceed {self.max_per_strategy:.0%} cap "
                f"(currently {strategy_total:.0%})"
            )
        
        # Check total exposure cap
        total = sum(w for _, _, w in self.positions)
        if total + desired_weight > self.max_total_exposure:
            adjusted = max(0, self.max_total_exposure - total)
            return False, adjusted, (
                f"Total exposure would exceed {self.max_total_exposure:.0%} cap "
                f"(currently {total:.0%})"
            )
        
        return True, desired_weight, ""
    
    def add(self, stock: str, strategy: str, weight: float):
        allowed, adj, reason = self.can_add(stock, strategy, weight)
        if adj > 0:
            self.positions.append((stock, strategy, adj))
        return allowed, adj, reason
    
    def total_exposure(self) -> float:
        return sum(w for _, _, w in self.positions)
    
    def by_strategy(self) -> dict:
        result = {}
        for _, s, w in self.positions:
            result[s] = result.get(s, 0) + w
        return result


# ── P0-3: Independent Circuit Breaker ──

class CircuitBreaker:
    """
    Independent safety switch. Cannot be overridden by any strategy vote.
    
    Tracks cumulative portfolio returns. If drawdown exceeds threshold,
    ALL strategies are suspended regardless of their individual votes.
    """
    
    def __init__(self, 
                 max_drawdown: float = -0.20,     # 20% portfolio drawdown → halt
                 max_monthly_loss: float = -0.10,  # 10% monthly → pause 1 week
                 max_consecutive_losses: int = 5,  # 5 consecutive losses → pause
                 cooldown_days: int = 5):           # days to pause after trip
        self.max_drawdown = max_drawdown
        self.max_monthly_loss = max_monthly_loss
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_days = cooldown_days
        
        self.portfolio_returns = []    # list of daily returns
        self.consecutive_losses = 0
        self.tripped = False
        self.trip_reason = ""
        self.trip_date = None
        self.peak_value = 1.0
        self.current_value = 1.0
    
    def update(self, daily_return: float):
        """Call after each trading day with portfolio return."""
        self.portfolio_returns.append(daily_return)
        self.current_value *= (1 + daily_return)
        self.peak_value = max(self.peak_value, self.current_value)
        
        if daily_return < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
    
    def check(self) -> tuple:
        """
        Check if breaker should trip.
        Returns (tripped: bool, reason: str)
        """
        if self.tripped:
            # Check if cooldown has elapsed
            if self.trip_date:
                from datetime import datetime, timedelta
                if (datetime.now() - self.trip_date).days >= self.cooldown_days:
                    self.tripped = False
                    self.trip_reason = ""
                    self.trip_date = None
                    self.consecutive_losses = 0
                    return False, "Cooldown elapsed. Resuming."
            return True, self.trip_reason
        
        drawdown = (self.current_value / self.peak_value) - 1
        
        if drawdown <= self.max_drawdown:
            self._trip(f"Portfolio drawdown {drawdown:.1%} exceeds {self.max_drawdown:.0%} limit")
            return True, self.trip_reason
        
        if self.consecutive_losses >= self.max_consecutive_losses:
            self._trip(f"{self.max_consecutive_losses} consecutive losing days")
            return True, self.trip_reason
        
        # Monthly check (simplified: last 21 trading days)
        if len(self.portfolio_returns) >= 21:
            monthly_ret = np.prod([1 + r for r in self.portfolio_returns[-21:]]) - 1
            if monthly_ret <= self.max_monthly_loss:
                self._trip(f"Monthly loss {monthly_ret:.1%} exceeds {self.max_monthly_loss:.0%} limit")
                return True, self.trip_reason
        
        return False, "OK"
    
    def _trip(self, reason: str):
        self.tripped = True
        self.trip_reason = reason
        from datetime import datetime
        self.trip_date = datetime.now()
    
    def status(self) -> str:
        if self.tripped:
            return f"BREAKER TRIPPED: {self.trip_reason}"
        drawdown = (self.current_value / self.peak_value) - 1
        return f"OK (dd={drawdown:.1%}, cons_loss={self.consecutive_losses})"


import numpy as np  # needed for P0-3 monthly return calc
