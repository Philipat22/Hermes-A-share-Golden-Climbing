"""
Feature Generator: computes 190+ alpha factors from stock price/volume data.
Acts as the bridge between Tushare/AKShare data and the WorldQuant/Alpha158 factor library.
"""
from __future__ import annotations
import time
import logging
from typing import Optional
from multiprocessing import get_context, Pool
from functools import partial

import polars as pl
import numpy as np

from .engine import DataProxy, calculate_by_expression
from .factors.alpha_101 import ALPHA_101
from .factors.alpha_158 import ALPHA_158

logger = logging.getLogger(__name__)


class FeatureGenerator:
    """
    Computes WorldQuant Alpha 101 + Alpha 158 factors.
    
    Input: Polars DataFrame with columns:
        datetime (datetime), vt_symbol (str),
        open, high, low, close, volume (float),
        vwap (float, optional - computed from turnover if needed)
    
    Output: Same DataFrame + factor columns.
    
    Supports:
    - Single-stock factor computation
    - Multi-stock batch computation with multiprocessing
    - Caching for repeated computations
    """

    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self._factor_results: dict[str, pl.DataFrame] = {}

    # ── Data preparation ──────────────────────────────────────────

    def prepare_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Normalize input DataFrame to standard format.
        Renames columns, computes vwap, ensures types.
        """
        result = df

        # Rename Tushare columns if needed
        rename_map = {
            "ts_code": "vt_symbol",
            "trade_date": "datetime",
            "vol": "volume",
            "amount": "amount",
        }
        for old, new in rename_map.items():
            if old in result.columns and new not in result.columns:
                result = result.rename({old: new})

        # Convert trade_date (int 20260101) to datetime
        if result["datetime"].dtype in (pl.Int32, pl.Int64, pl.Float64):
            result = result.with_columns(
                pl.col("datetime").cast(pl.Utf8).str.to_datetime("%Y%m%d")
            )

        # Add .SH/.SZ suffix to stock codes if missing
        if "vt_symbol" in result.columns:
            result = result.with_columns(
                pl.when(
                    ~pl.col("vt_symbol").str.contains(r"\.")
                ).then(
                    pl.when(pl.col("vt_symbol").str.starts_with("6"))
                    .then(pl.col("vt_symbol") + ".SH")
                    .otherwise(pl.col("vt_symbol") + ".SZ")
                ).otherwise(pl.col("vt_symbol")).alias("vt_symbol")
            )

        # Compute VWAP if not present
        # Tushare amount is in 千元, volume is in 手(100 shares)
        # vwap (元/股) = (amount * 1000) / (volume * 100) = amount * 10 / volume
        if "vwap" not in result.columns:
            if "amount" in result.columns and "volume" in result.columns:
                result = result.with_columns(
                    (pl.col("amount") * 10.0 / (pl.col("volume") + 1e-12)).alias("vwap")
                )
            else:
                result = result.with_columns(
                    ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("vwap")
                )

        return result.sort(["vt_symbol", "datetime"])

    # ── Core factor computation ────────────────────────────────────

    def compute_single_factor(
        self, name: str, expression: str, df: pl.DataFrame
    ) -> tuple[str, pl.Series]:
        """
        Compute a single alpha factor by expression string.
        """
        t0 = time.time()
        try:
            result_df = calculate_by_expression(df, expression)
            series = result_df["data"].alias(name)
            elapsed = time.time() - t0
            if elapsed > 5:
                logger.info(f"  {name}: {elapsed:.1f}s")
            return name, series
        except Exception as e:
            logger.warning(f"  {name}: FAILED - {e}")
            return name, pl.Series(name, [None] * df.height)

    def _compute_factors_multi(self, factor_dict: dict, df: pl.DataFrame, label: str) -> pl.DataFrame:
        """Compute factors using multiprocessing (for max_workers > 1)."""
        names = list(factor_dict.keys())
        expressions = [factor_dict[n] for n in names]
        logger.info(f"Computing {len(names)} {label} factors (multi, {self.max_workers} workers)...")
        t0 = time.time()

        results = []
        with get_context("spawn").Pool(processes=self.max_workers) as pool:
            it = pool.imap(
                partial(self._compute_factor, df=df),
                [(name, expr) for name, expr in zip(names, expressions)],
                chunksize=5,
            )
            for name, series in it:
                results.append(series)

        result = df.with_columns(results)
        logger.info(f"{label} complete: {time.time()-t0:.1f}s")
        return result

    def _compute_factors_seq(self, factor_dict: dict, df: pl.DataFrame, label: str) -> pl.DataFrame:
        """Compute factors sequentially in main process (no multiprocessing).
        Much more memory-efficient on Windows."""
        names = list(factor_dict.keys())
        expressions = [factor_dict[n] for n in names]
        logger.info(f"Computing {len(names)} {label} factors (sequential)...")
        t0 = time.time()

        results = []
        for name, expr in zip(names, expressions):
            try:
                result_df = calculate_by_expression(df, expr)
                series = result_df["data"].alias(name)
            except Exception as e:
                series = pl.Series(name, [None] * df.height)
            results.append(series)

        result = df.with_columns(results)
        logger.info(f"{label} complete: {time.time()-t0:.1f}s")
        return result

    def compute_alpha_101(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute all Alpha 101 factors.
        Auto-prepares data if needed.
        """
        if "vwap" not in df.columns:
            df = self.prepare_df(df)
        if self.max_workers > 1:
            return self._compute_factors_multi(ALPHA_101, df, "Alpha 101")
        else:
            return self._compute_factors_seq(ALPHA_101, df, "Alpha 101")

    def compute_alpha_158(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute all Alpha 158 factors.
        Auto-prepares data if needed.
        """
        if "vwap" not in df.columns:
            df = self.prepare_df(df)
        if self.max_workers > 1:
            return self._compute_factors_multi(ALPHA_158, df, "Alpha 158")
        else:
            return self._compute_factors_seq(ALPHA_158, df, "Alpha 158")

    def compute_all(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute ALL factors (Alpha 101 + Alpha 158).
        Returns DataFrame with ~200 factor columns.
        """
        t0 = time.time()
        logger.info(f"Computing all factors for {len(df)} rows...")

        df101 = self.compute_alpha_101(df)
        df_all = self.compute_alpha_158(df101)

        logger.info(f"All factors complete: {time.time()-t0:.1f}s")
        return df_all

    @staticmethod
    def _compute_factor(args: tuple[str, str], df: pl.DataFrame) -> tuple[str, pl.Series]:
        """Static wrapper for multiprocessing."""
        name, expression = args
        t0 = time.time()
        try:
            result_df = calculate_by_expression(df, expression)
            series = result_df["data"].alias(name)
            elapsed = time.time() - t0
            if elapsed > 3:
                logger.info(f"  {name}: {elapsed:.1f}s")
            return name, series
        except Exception as e:
            logger.warning(f"  {name}: FAILED - {e}")
            return name, pl.Series(name, [None] * df.height)

    # ── Factor reduction / selection helpers ───────────────────────

    def compute_factors_for_date(
        self, df: pl.DataFrame, target_date: str
    ) -> pl.DataFrame:
        """
        Compute factors and extract only the target date's values.
        
        This is useful for real-time: compute on a wide window, 
        then take only the latest row per stock.
        """
        # Need enough history for the longest window (250 days)
        result = self.compute_all(df)

        # Filter to target date
        if isinstance(target_date, str):
            target_date = pl.Series([target_date]).str.to_datetime("%Y%m%d")[0]

        return result.filter(pl.col("datetime") == target_date)

    def select_top_factors(
        self, df: pl.DataFrame, n: int = 50
    ) -> list[str]:
        """
        Select the top N most informative factors based on variance.
        Simple heuristic - can be enhanced with IC-based selection later.
        """
        factor_cols = [c for c in df.columns if c.startswith("alpha") or 
                      c in ALPHA_101 or c in ALPHA_158]
        
        variances = []
        for col in factor_cols:
            try:
                v = df[col].drop_nans().std()
                if v is not None and not np.isnan(v):
                    variances.append((col, v))
            except Exception:
                pass

        variances.sort(key=lambda x: x[1], reverse=True)
        return [v[0] for v in variances[:n]]


# ── Convenience function ─────────────────────────────────────────

def compute_features(
    df: pl.DataFrame,
    factors: str = "all",
    max_workers: int = 8
) -> pl.DataFrame:
    """
    One-liner: prepare data and compute features.
    
    Args:
        df: Input DataFrame (price/volume data)
        factors: "all", "alpha101", "alpha158"
        max_workers: Parallel workers
    
    Returns:
        DataFrame with factor columns added
    """
    fg = FeatureGenerator(max_workers=max_workers)
    prepared = fg.prepare_df(df)

    if factors == "alpha101":
        return fg.compute_alpha_101(prepared)
    elif factors == "alpha158":
        return fg.compute_alpha_158(prepared)
    else:
        return fg.compute_all(prepared)


# ── Tushare integration helper ───────────────────────────────────

def fetch_and_compute(
    stock_codes: list[str],
    start_date: str,
    end_date: str,
    max_workers: int = 8,
    extended_days: int = 400,
) -> pl.DataFrame:
    """
    Fetch data from Tushare and compute factors.
    
    This is a convenience wrapper that:
    1. Tries to import our project's data_fetcher
    2. Fetches daily data with extended history
    3. Computes all factors
    
    Returns Polars DataFrame with all factor columns.
    """
    from src.tools.data_fetcher import get_bulk_daily

    df = get_bulk_daily(
        stocks=stock_codes,
        start_date=start_date,
        end_date=end_date,
        extended_days=extended_days,
    )
    pdf = pl.from_pandas(df)
    return compute_features(pdf, max_workers=max_workers)
