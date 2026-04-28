"""
A股分析专属端点 — 直接调用 run_astock_analysis，绕过 LangGraph 复杂流程。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import json

from src.main_astock import run_astock_analysis
from src.utils.analysts import ANALYST_CONFIG, get_agents_list

router = APIRouter(prefix="/api/astock", tags=["A-stock"])


class AStockAnalysisRequest(BaseModel):
    tickers: list[str]
    selected_analysts: Optional[list[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    model_name: str = "deepseek-chat"
    model_provider: str = "DeepSeek"


class AStockAnalysisResponse(BaseModel):
    success: bool
    analyst_signals: dict
    available_analysts: list


@router.get("/analysts")
def list_analysts():
    """列出所有可用的 A股分析师"""
    return {
        "analysts": [
            {"id": key, "name": config.get("name", key), "type": "LLM" if "call_llm" in str(config.get("agent_func", "")) else "calculated"}
            for key, config in ANALYST_CONFIG.items()
        ]
    }


@router.post("/analyze", response_model=AStockAnalysisResponse)
def analyze(request: AStockAnalysisRequest):
    """运行 A股分析"""
    try:
        result = run_astock_analysis(
            tickers=request.tickers,
            selected_analysts=request.selected_analysts or list(ANALYST_CONFIG.keys()),
            start_date=request.start_date,
            end_date=request.end_date,
            model_name=request.model_name,
            model_provider=request.model_provider,
            show_reasoning=False,
        )
        return AStockAnalysisResponse(
            success=True,
            analyst_signals=result.get("analyst_signals", {}),
            available_analysts=list(ANALYST_CONFIG.keys()),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
