"""
趋势指标计算模块 — ADX / MACD / Bollinger Bands / ATR 增强

直接操作 Polars DataFrame（列: vt_symbol, date, open, high, low, close, volume）
每只股票独立计算，支持全市场批量处理。

用法:
    from src.features.indicators import compute_all_trend_indicators
    df = compute_all_trend_indicators(prices_df)
    # 新增列: adx, plus_di, minus_di, macd, macd_signal, macd_hist,
    #          bb_upper, bb_middle, bb_lower, bb_pct_b, bb_bandwidth,
    #          atr_14, atr_pct
"""
from __future__ import annotations

import polars as pl
import numpy as np


def _wilder_smooth(series: pl.Expr, window: int) -> pl.Expr:
    """Wilder 平滑 (EMA with alpha=1/window)"""
    return series.ewm_mean(alpha=1.0 / window, min_periods=window)


def compute_adx(
    df: pl.DataFrame,
    window: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pl.DataFrame:
    """计算 ADX / +DI / -DI (Wilder平滑)

    返回原始 df + adx, plus_di, minus_di 三列
    """
    # 按 symbol 分组
    df = df.sort(["vt_symbol", "date"])

    # True Range
    prev_close = pl.col(close_col).shift(1)
    tr1 = pl.col(high_col) - pl.col(low_col)
    tr2 = (pl.col(high_col) - prev_close).abs()
    tr3 = (pl.col(low_col) - prev_close).abs()
    tr = pl.max_horizontal(tr1, tr2, tr3)

    # Directional Movement
    up_move = pl.col(high_col) - pl.col(high_col).shift(1)
    down_move = pl.col(low_col).shift(1) - pl.col(low_col)

    plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0.0)
    minus_dm = pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0.0)

    # Wilder 平滑 (14-period)
    atr = tr.rolling_mean(window, min_periods=window)
    smoothed_plus_dm = plus_dm.rolling_mean(window, min_periods=window)
    smoothed_minus_dm = minus_dm.rolling_mean(window, min_periods=window)

    # Wilder EMA (the proper way)
    atr_w = _wilder_smooth(tr, window)
    plus_dm_w = _wilder_smooth(plus_dm, window)
    minus_dm_w = _wilder_smooth(minus_dm, window)

    # +DI / -DI
    plus_di = (plus_dm_w / atr_w) * 100.0
    minus_di = (minus_dm_w / atr_w) * 100.0

    # DX = |+DI - -DI| / (+DI + -DI) * 100
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100.0

    # ADX = Wilder smooth of DX
    adx = _wilder_smooth(dx.fill_nan(0.0), window)

    df = df.with_columns([
        adx.over("vt_symbol").alias("adx"),
        plus_di.over("vt_symbol").alias("plus_di"),
        minus_di.over("vt_symbol").alias("minus_di"),
    ])

    return df


def compute_macd(
    df: pl.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    close_col: str = "close",
) -> pl.DataFrame:
    """计算 MACD (EMA 实现)

    返回原始 df + macd, macd_signal, macd_hist 三列
    """
    df = df.sort(["vt_symbol", "date"])

    ema_fast = pl.col(close_col).ewm_mean(span=fast, min_periods=fast)
    ema_slow = pl.col(close_col).ewm_mean(span=slow, min_periods=slow)

    macd_line = ema_fast - ema_slow
    macd_signal_line = macd_line.ewm_mean(span=signal, min_periods=signal)
    macd_hist = macd_line - macd_signal_line

    df = df.with_columns([
        macd_line.over("vt_symbol").alias("macd"),
        macd_signal_line.over("vt_symbol").alias("macd_signal"),
        macd_hist.over("vt_symbol").alias("macd_hist"),
    ])

    return df


def compute_bollinger(
    df: pl.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    close_col: str = "close",
) -> pl.DataFrame:
    """计算 Bollinger Bands

    返回原始 df + bb_middle, bb_upper, bb_lower, bb_pct_b, bb_bandwidth 五列

    - bb_pct_b (0~1): 价格在带中的位置；<0 突破下轨, >1 突破上轨
    - bb_bandwidth: 带宽百分比 = (upper-lower)/middle；收窄 = 即将突破
    """
    df = df.sort(["vt_symbol", "date"])

    middle = pl.col(close_col).rolling_mean(window, min_periods=window)
    std = pl.col(close_col).rolling_std(window, min_periods=window)

    upper = middle + std * num_std
    lower = middle - std * num_std

    # %B = (close - lower) / (upper - lower)
    pct_b = (pl.col(close_col) - lower) / (upper - lower)

    # Bandwidth = (upper - lower) / middle
    bandwidth = (upper - lower) / middle

    df = df.with_columns([
        middle.over("vt_symbol").alias("bb_middle"),
        upper.over("vt_symbol").alias("bb_upper"),
        lower.over("vt_symbol").alias("bb_lower"),
        pct_b.over("vt_symbol").alias("bb_pct_b"),
        bandwidth.over("vt_symbol").alias("bb_bandwidth"),
    ])

    return df


def compute_atr_enhanced(
    df: pl.DataFrame,
    window: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pl.DataFrame:
    """计算 ATR 及 ATR 百分比

    返回原始 df + atr_14, atr_pct 两列
    atr_pct = atr / close * 100  (ATR 占价格的百分比，用于止损计算)
    """
    df = df.sort(["vt_symbol", "date"])

    prev_close = pl.col(close_col).shift(1)
    tr1 = pl.col(high_col) - pl.col(low_col)
    tr2 = (pl.col(high_col) - prev_close).abs()
    tr3 = (pl.col(low_col) - prev_close).abs()
    tr = pl.max_horizontal(tr1, tr2, tr3)

    atr = tr.rolling_mean(window, min_periods=window)
    atr_pct = (atr / pl.col(close_col)) * 100.0

    df = df.with_columns([
        atr.over("vt_symbol").alias("atr_14"),
        atr_pct.over("vt_symbol").alias("atr_pct"),
    ])

    return df


def compute_all_trend_indicators(
    df: pl.DataFrame,
    adx_window: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_window: int = 20,
    bb_std: float = 2.0,
    atr_window: int = 14,
) -> pl.DataFrame:
    """一次性计算全部趋势指标

    Args:
        df: Polars DataFrame
            必需列: vt_symbol, date, open, high, low, close
            可选列: volume (仅用于透传)

    Returns:
        原始 df + 所有趋势指标列 (共 13 个新列)
    """
    df = compute_adx(df, window=adx_window)
    df = compute_macd(df, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    df = compute_bollinger(df, window=bb_window, num_std=bb_std)
    df = compute_atr_enhanced(df, window=atr_window)
    return df
