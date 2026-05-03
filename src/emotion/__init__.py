"""
src/emotion — 情绪融合引擎

P1 模块结构：
  news_sentiment.py  — 个股新闻情绪评分
  market_mood.py     — 市场整体情绪指标
  fusion.py          — 多信号融合引擎 → 综合情绪评分
"""

from .news_sentiment import get_news_score
from .market_mood import get_market_mood
from .fusion import analyze_emotion, batch_emotion_score, EMOTION_WEIGHTS

__all__ = ["get_news_score", "get_market_mood", "analyze_emotion", "batch_emotion_score", "EMOTION_WEIGHTS"]
