"""
Technical Analysis Functions (no talib dependency)
"""
import polars as pl
import numpy as np
from .data_proxy import DataProxy


def ta_rsi(close: DataProxy, window: int = 14) -> DataProxy:
    """RSI calculation without talib"""
    def _rsi(s: pl.Series) -> float:
        arr = s.to_numpy()
        if len(arr) < window + 1:
            return np.nan
        deltas = np.diff(arr)
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = np.abs(losses)
        avg_gain = np.mean(gains[:window])
        avg_loss = np.mean(losses[:window])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df = close.df.select(
        pl.col("datetime"), pl.col("vt_symbol"),
        pl.col("data").rolling_map(_rsi, window + 1).over("vt_symbol")
    )
    return DataProxy(df)


def ta_atr(high: DataProxy, low: DataProxy, close: DataProxy, window: int = 14) -> DataProxy:
    """ATR calculation without talib"""
    df_merged = high.df.join(low.df, on=["datetime", "vt_symbol"], suffix="_low")
    df_merged = df_merged.join(close.df, on=["datetime", "vt_symbol"], suffix="_close")
    df_merged = df_merged.with_columns([
        (pl.col("data_low") - pl.col("data_close").shift(1)).abs().alias("tr1"),
        (pl.col("data") - pl.col("data_close").shift(1)).abs().alias("tr2"),
        (pl.col("data") - pl.col("data_low")).alias("tr3"),
        (pl.max_horizontal(
            (pl.col("data") - pl.col("data_low")),
            (pl.col("data") - pl.col("data_close").shift(1)).abs(),
            (pl.col("data_low") - pl.col("data_close").shift(1)).abs()
        )).alias("tr")
    ])
    df = df_merged.select(
        pl.col("datetime"), pl.col("vt_symbol"),
        pl.col("tr").rolling_mean(window, min_samples=1).over("vt_symbol").alias("data")
    )
    return DataProxy(df)
