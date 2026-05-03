"""
BaseAnalystAgent — 投资大师 Agent 基类

所有 18 位投资大师 Agent 的通用逻辑：
1. 数据获取 (API calls)
2. LLM 推理 (prompt → signal)
3. 结果格式化 (standardized output)

子类只需定义:
- scoring_logic(ticker, metrics, line_items) → dict
- prompt_template — LLM 提示词模板

用法:
    class WarrenBuffettAgent(BaseAnalystAgent):
        agent_id = "warren_buffett_agent"
        display_name = "Warren Buffett"
        prompt_template = '''...'''

        def scoring_logic(self, ticker, metrics, line_items):
            return {"score": 85, "details": "Strong moat..."}
"""
from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, Field
from typing_extensions import Literal
from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items
from src.tools.a_stock_api import get_market_context
from src.utils.progress import progress
from src.utils.llm import call_llm
from src.utils.api_key import get_api_key_from_state


class BaseAnalystSignal(BaseModel):
    """统一的 Agent 输出信号格式"""
    signal: Literal["bullish", "bearish", "neutral"] = Field(default="neutral")
    confidence: float = Field(default=0.0, description="Confidence 0-100")
    reasoning: str = Field(default="", description="Reasoning for the decision")


class BaseAnalystAgent:
    """
    投资大师 Agent 基类

    Attributes:
        agent_id: Agent 唯一标识（如 "warren_buffett_agent"）
        display_name: 显示名称（如 "Warren Buffett"）
        prompt_template: LLM 提示词模板字符串
    """

    agent_id: str = "base_analyst_agent"
    display_name: str = "Base Analyst"
    prompt_template: str = ""

    def scoring_logic(
        self,
        ticker: str,
        metrics: list,
        financial_line_items: list,
        market_cap: Optional[float],
    ) -> dict:
        """
        子类重写此方法实现各大师独特的评分逻辑。

        Returns:
            {"score": int, "max_score": int, "details": str, ...}
        """
        return {"score": 0, "max_score": 100, "details": "Base scoring — override me"}

    def call_llm_for_signal(
        self,
        ticker: str,
        analysis_data: dict,
        market_context: dict,
        state: AgentState,
    ) -> BaseAnalystSignal:
        """调用 LLM 生成交易信号。子类可重写以自定义 prompt 构建。"""
        if not self.prompt_template:
            return BaseAnalystSignal(
                signal="neutral",
                confidence=0,
                reasoning="No prompt template configured",
            )

        prompt = self.prompt_template.format(
            ticker=ticker,
            analysis_data=json.dumps(analysis_data, ensure_ascii=False, indent=2),
            market_context=json.dumps(market_context, ensure_ascii=False, indent=2),
        )

        result = call_llm(
            prompt=prompt,
            model_name=state["metadata"].get("model_name", "deepseek-chat"),
            model_provider=state["metadata"].get("model_provider", "DeepSeek"),
        )

        # Parse LLM response into structured signal
        return self._parse_llm_response(result)

    def _parse_llm_response(self, raw_response: str) -> BaseAnalystSignal:
        """从 LLM 文本响应中提取信号。"""
        try:
            data = json.loads(raw_response)
            return BaseAnalystSignal(
                signal=data.get("signal", "neutral"),
                confidence=float(data.get("confidence", 0)),
                reasoning=str(data.get("reasoning", "")),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return BaseAnalystSignal(
                signal="neutral",
                confidence=0,
                reasoning=f"Failed to parse LLM response: {raw_response[:200]}",
            )

    def analyze_ticker(
        self,
        ticker: str,
        end_date: str,
        state: AgentState,
    ) -> dict:
        """
        对单只股票执行完整分析流程:
        1. 获取数据
        2. 计算评分
        3. 调用 LLM 生成信号

        Returns:
            {"signal": str, "confidence": float, "reasoning": str}
        """
        api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

        progress.update_status(self.agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=10, api_key=api_key)

        progress.update_status(self.agent_id, ticker, "Fetching financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "net_income", "revenue", "gross_profit",
                "total_assets", "total_liabilities", "shareholders_equity",
                "capital_expenditure", "depreciation_and_amortization",
                "free_cash_flow", "outstanding_shares",
                "dividends_and_other_cash_distributions",
                "issuance_or_purchase_of_equity_shares",
            ],
            end_date,
            period="ttm",
            limit=10,
            api_key=api_key,
        )

        progress.update_status(self.agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(self.agent_id, ticker, f"Running {self.display_name} analysis")
        scoring = self.scoring_logic(ticker, metrics, financial_line_items, market_cap)

        progress.update_status(self.agent_id, ticker, "Fetching market context")
        market_context = get_market_context(ticker, end_date)

        # Build analysis data for LLM
        analysis_data = {
            "ticker": ticker,
            "scoring": scoring,
            "market_cap": market_cap,
        }

        progress.update_status(self.agent_id, ticker, f"Generating {self.display_name} signal")
        signal = self.call_llm_for_signal(ticker, analysis_data, market_context, state)

        progress.update_status(self.agent_id, ticker, "Done", analysis=signal.reasoning)

        return {
            "signal": signal.signal,
            "confidence": signal.confidence,
            "reasoning": signal.reasoning,
        }

    def __call__(self, state: AgentState) -> dict:
        """
        作为 LangGraph node 被调用的入口。

        Returns:
            {"messages": [...], "data": {...}}
        """
        data = state["data"]
        end_date = data["end_date"]
        tickers = data["tickers"]

        analysis_results = {}

        for ticker in tickers:
            try:
                analysis_results[ticker] = self.analyze_ticker(ticker, end_date, state)
            except Exception as e:
                import traceback
                print(f"\n  [{self.agent_id}] {ticker} 分析失败: {e}")
                traceback.print_exc()
                analysis_results[ticker] = {
                    "signal": "neutral",
                    "confidence": 0,
                    "reasoning": f"分析异常: {e}",
                }

        message = HumanMessage(
            content=json.dumps(analysis_results, ensure_ascii=False),
            name=self.agent_id,
        )

        if state["metadata"].get("show_reasoning", False):
            show_agent_reasoning(analysis_results, self.agent_id)

        if "analyst_signals" not in state["data"]:
            state["data"]["analyst_signals"] = {}
        state["data"]["analyst_signals"][self.agent_id] = analysis_results

        progress.update_status(self.agent_id, None, "Done")

        return {"messages": [message], "data": state["data"]}
