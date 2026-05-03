#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Surge — 主升浪捕捉模块
情绪感知 + 形态识别 + 自我进化
"""
from src.surge.engine import (
    analyze_stock,
    detect_platform_breakout,
    detect_n_shape,
    detect_vcp,
    measure_acceleration,
    score_volume_structure,
    detect_fake_signal,
    DEFAULT_PARAMS,
)
from src.surge.scanner import scan_market, SCORE_COLS
from src.surge.feedback import SignalMemory, evaluate_signals, adjust_params

__all__ = [
    'analyze_stock', 'detect_platform_breakout', 'detect_n_shape', 'detect_vcp',
    'measure_acceleration', 'score_volume_structure', 'detect_fake_signal',
    'scan_market', 'SCORE_COLS',
    'SignalMemory', 'evaluate_signals', 'adjust_params',
    'DEFAULT_PARAMS',
]
