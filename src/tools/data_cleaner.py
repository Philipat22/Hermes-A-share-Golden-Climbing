#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗工具函数
"""
from datetime import datetime, timedelta
from typing import Any


def _safe_float(value: Any, default: float = None) -> float:
    """安全地将值转换为 float。处理 None、字符串、pandas NA 等情况。"""
    if value is None:
        return default if default is not None else 0.0
    import math
    if isinstance(value, float):
        return 0.0 if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if value in ('', 'nan', 'None', 'NA', '--', '-'):
            return default if default is not None else 0.0
        try:
            result = float(value)
            return 0.0 if (math.isnan(result) or math.isinf(result)) else result
        except (ValueError, TypeError):
            return default if default is not None else 0.0
    try:
        result = float(value)
        import math
        return 0.0 if (math.isnan(result) or math.isinf(result)) else result
    except (ValueError, TypeError, AttributeError):
        return default if default is not None else 0.0


def _calc_date(days: int = 0, fmt: str = '%Y%m%d') -> str:
    """计算相对日期。days > 0 未来，days < 0 过去。"""
    return (datetime.now() + timedelta(days=days)).strftime(fmt)
