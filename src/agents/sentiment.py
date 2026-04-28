from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
import pandas as pd
import numpy as np
import json
from src.utils.api_key import get_api_key_from_state
from src.tools.a_stock_api import get_insider_trades, get_company_news


##### Sentiment Agent #####
def sentiment_analyst_agent(state: AgentState, agent_id: str = "sentiment_analyst_agent"):
    """Analyzes market sentiment and generates trading signals for multiple tickers."""
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    # Initialize sentiment analysis for each ticker
    sentiment_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching insider trades")

        # Get the insider trades
        insider_trades = get_insider_trades(
            ticker=ticker,
            end_date=end_date,
            limit=1000,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Analyzing trading patterns")

        # Get the signals from the insider trades
        # A股融资融券数据 (A-share margin data, not insider trades):
        # - margin_balance (rzye 融资余额): total margin borrowing outstanding
        # - short_balance (rqye 融券余额): total short-selling outstanding
        # - change_pct (rzmre 融资买入额): today's margin buying amount
        # Strategy: compare day-over-day changes in margin vs short balances
        # Margin balance ↑ = more leveraged longs = bullish
        # Short balance ↑ = more short selling = bearish
        # Short balance / margin ratio ↑ = bearish divergence
        if len(insider_trades) >= 2:
            sorted_trades = sorted(insider_trades, key=lambda t: t.trade_date)
            
            # Compute composite score per day-pair
            scores = []
            for i in range(1, len(sorted_trades)):
                prev = sorted_trades[i-1]
                curr = sorted_trades[i]
                
                # Margin balance change (%)
                margin_pct = (curr.margin_balance - prev.margin_balance) / max(abs(prev.margin_balance), 1)
                # Short balance change (%)
                short_pct = (curr.short_balance - prev.short_balance) / max(abs(prev.short_balance), 1)
                
                # Composite: margin up = bullish, short up = bearish
                # Weight: margin changes more important for A-share sentiment
                composite = margin_pct - short_pct * 0.5
                
                if composite > 0.005:  # significant bullish movement
                    scores.append("bullish")
                elif composite < -0.005:  # significant bearish movement
                    scores.append("bearish")
                else:
                    scores.append("neutral")
            
            insider_signals = scores
        else:
            insider_signals = ["neutral"]

        progress.update_status(agent_id, ticker, "Fetching company news")

        # Get the company news
        company_news = get_company_news(ticker, end_date, limit=100, api_key=api_key)

        # Get the sentiment from the company news
        sentiment = pd.Series([n.sentiment for n in company_news]).dropna()
        news_signals = np.where(sentiment == "negative", "bearish", 
                              np.where(sentiment == "positive", "bullish", "neutral")).tolist()
        
        progress.update_status(agent_id, ticker, "Combining signals")
        # Combine signals from both sources with weights
        insider_weight = 0.3
        news_weight = 0.7
        
        # Calculate weighted signal counts
        bullish_signals = (
            insider_signals.count("bullish") * insider_weight +
            news_signals.count("bullish") * news_weight
        )
        bearish_signals = (
            insider_signals.count("bearish") * insider_weight +
            news_signals.count("bearish") * news_weight
        )

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level based on the weighted proportion
        total_weighted_signals = len(insider_signals) * insider_weight + len(news_signals) * news_weight
        confidence = 0  # Default confidence when there are no signals
        if total_weighted_signals > 0:
            confidence = round((max(bullish_signals, bearish_signals) / total_weighted_signals) * 100, 2)
        
        # Create structured reasoning similar to technical analysis
        reasoning = {
            "insider_trading": {
                "signal": "bullish" if insider_signals.count("bullish") > insider_signals.count("bearish") else 
                         "bearish" if insider_signals.count("bearish") > insider_signals.count("bullish") else "neutral",
                "confidence": round((max(insider_signals.count("bullish"), insider_signals.count("bearish")) / max(len(insider_signals), 1)) * 100),
                "metrics": {
                    "total_trades": len(insider_signals),
                    "bullish_trades": insider_signals.count("bullish"),
                    "bearish_trades": insider_signals.count("bearish"),
                    "weight": insider_weight,
                    "weighted_bullish": round(insider_signals.count("bullish") * insider_weight, 1),
                    "weighted_bearish": round(insider_signals.count("bearish") * insider_weight, 1),
                }
            },
            "news_sentiment": {
                "signal": "bullish" if news_signals.count("bullish") > news_signals.count("bearish") else 
                         "bearish" if news_signals.count("bearish") > news_signals.count("bullish") else "neutral",
                "confidence": round((max(news_signals.count("bullish"), news_signals.count("bearish")) / max(len(news_signals), 1)) * 100),
                "metrics": {
                    "total_articles": len(news_signals),
                    "bullish_articles": news_signals.count("bullish"),
                    "bearish_articles": news_signals.count("bearish"),
                    "neutral_articles": news_signals.count("neutral"),
                    "weight": news_weight,
                    "weighted_bullish": round(news_signals.count("bullish") * news_weight, 1),
                    "weighted_bearish": round(news_signals.count("bearish") * news_weight, 1),
                }
            },
            "combined_analysis": {
                "total_weighted_bullish": round(bullish_signals, 1),
                "total_weighted_bearish": round(bearish_signals, 1),
                "signal_determination": f"{'Bullish' if bullish_signals > bearish_signals else 'Bearish' if bearish_signals > bullish_signals else 'Neutral'} based on weighted signal comparison"
            }
        }

        sentiment_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(reasoning, indent=4))

    # Create the sentiment message
    message = HumanMessage(
        content=json.dumps(sentiment_analysis),
        name=agent_id,
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sentiment_analysis, "Sentiment Analysis Agent")

    # Add the signal to the analyst_signals list
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"][agent_id] = sentiment_analysis

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": data,
    }
