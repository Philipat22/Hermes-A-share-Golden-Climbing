"""
Ben Graham Agent — refactored with BaseAnalystAgent

Classic value-investing principles:
1. Earnings stability over multiple years
2. Solid financial strength (low debt, adequate liquidity)
3. Discount to intrinsic value (Graham Number or net-net)
4. Adequate margin of safety
"""
import math
from typing import Optional

from src.agents.base_analyst import BaseAnalystAgent


class BenGrahamAgentV2(BaseAnalystAgent):
    agent_id = "ben_graham_agent"
    display_name = "Ben Graham"

    prompt_template = """You are Ben Graham, the father of value investing.
Analyze the following stock and provide a trading signal.

Ticker: {ticker}
Analysis Data: {analysis_data}
Market Context: {market_context}

Return JSON:
{{"signal": "bullish"|"bearish"|"neutral", "confidence": 0-100, "reasoning": "..."}}

Your criteria:
- Earnings stability: consistent profits over 5+ years
- Financial strength: current ratio > 2, debt/equity < 0.5
- Margin of safety: price < 2/3 of net current asset value
- Graham Number: price below sqrt(22.5 * EPS * BVPS)
"""

    def scoring_logic(
        self,
        ticker: str,
        metrics: list,
        financial_line_items: list,
        market_cap: Optional[float],
    ) -> dict:
        if not metrics:
            return {"score": 0, "max_score": 100, "details": "Insufficient data"}

        latest = metrics[0]
        score = 0
        details = []

        # Price-to-Book check
        if latest.price_to_book and latest.price_to_book < 1.5:
            score += 30
            details.append(f"P/B {latest.price_to_book:.1f} < 1.5 (deep value)")
        elif latest.price_to_book:
            details.append(f"P/B {latest.price_to_book:.1f}")

        # Debt-to-Equity check
        if latest.debt_to_equity and latest.debt_to_equity < 0.5:
            score += 25
            details.append("Conservative debt (D/E < 0.5)")
        elif latest.debt_to_equity:
            details.append(f"D/E = {latest.debt_to_equity:.1f}")

        # Current Ratio check
        if latest.current_ratio and latest.current_ratio > 2.0:
            score += 20
            details.append(f"Strong liquidity (CR {latest.current_ratio:.1f} > 2)")
        elif latest.current_ratio:
            details.append(f"CR = {latest.current_ratio:.1f}")

        # Graham Number approximation
        if latest.earnings_per_share and latest.book_value_per_share:
            graham_number = math.sqrt(22.5 * latest.earnings_per_share * latest.book_value_per_share)
            if market_cap and market_cap < graham_number:
                score += 25
                details.append(f"Below Graham Number ({graham_number:.0f} vs mcap {market_cap:.0f})")

        return {
            "score": score,
            "max_score": 100,
            "details": "; ".join(details) if details else "No Graham criteria met",
        }


# Create singleton instance for LangGraph compatibility
ben_graham_agent_v2 = BenGrahamAgentV2()
