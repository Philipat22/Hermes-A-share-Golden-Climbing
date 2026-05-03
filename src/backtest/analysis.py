"""Factor analysis and signal performance evaluation for AHF.

Provides IC (Information Coefficient) analysis, quantile return
analysis, and signal performance evaluation -- replacing alphalens
which is incompatible with Python 3.12+.
"""
from datetime import datetime
from typing import Optional

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from ..tools.data_fetcher import get_prices


class FactorAnalyzer:
    """Factor and signal performance analysis toolkit."""

    @staticmethod
    def _merge_and_add_fwd_ret(
        factor_df: pl.DataFrame,
        price_df: pl.DataFrame,
        forward_period: int
    ) -> pl.DataFrame:
        """Merge factor and price data, compute forward returns per symbol.

        Returns DataFrame with columns: datetime, vt_symbol, factor_value, fwd_ret
        """
        merged = factor_df.join(
            price_df, on=["datetime", "vt_symbol"], how="inner"
        ).sort(["vt_symbol", "datetime"])

        if len(merged) == 0:
            return pl.DataFrame()

        # Compute forward N-period return per symbol: close[t+N] / close[t] - 1
        df = merged.with_columns(
            ((pl.col("close").shift(-forward_period) / pl.col("close")) - 1.0)
            .over("vt_symbol").alias("fwd_ret")
        ).filter(
            pl.col("fwd_ret").is_not_null()
        ).select([
            "datetime", "vt_symbol", "factor_value", "fwd_ret"
        ])

        return df

    @staticmethod
    def calc_ic(
        factor_df: pl.DataFrame,
        price_df: pl.DataFrame,
        forward_periods: list[int] = [1, 5, 10, 20]
    ) -> dict:
        """Compute rank IC (Spearman correlation) for a factor.

        Args:
            factor_df: columns [datetime, vt_symbol, factor_value]
            price_df:  columns [datetime, vt_symbol, close]
            forward_periods: forward return horizons

        Returns:
            {period_str: {"mean_ic": float, "std_ic": float, 
                          "ic_ratio": float, "sample_count": int}}
        """
        results = {}
        dates_all = price_df["datetime"].unique().sort()

        for period in forward_periods:
            df = FactorAnalyzer._merge_and_add_fwd_ret(
                factor_df, price_df, period
            )
            if len(df) == 0:
                continue

            ics = []
            ic_dates = []

            for dt in dates_all:
                day = df.filter(pl.col("datetime") == dt)
                if day.height < 10:
                    continue

                fwd = day["fwd_ret"].to_numpy()
                fact = day["factor_value"].to_numpy()

                # Need variation for correlation
                if np.nanstd(fwd) < 1e-10 or np.nanstd(fact) < 1e-10:
                    continue

                ic, _ = spearmanr(fact, fwd)
                if not np.isnan(ic):
                    ics.append(float(ic))
                    ic_dates.append(dt)

            if ics:
                results[f"{period}d"] = {
                    "mean_ic": float(np.mean(ics)),
                    "std_ic": float(np.std(ics)),
                    "ic_ratio": float(np.mean(ics) / np.std(ics)) if np.std(ics) > 0 else 0.0,
                    "sample_count": len(ics),
                    "ic_series": {
                        str(d.date()): ic
                        for d, ic in zip(ic_dates, ics)
                    },
                }

        return results

    @staticmethod
    def calc_quantile_returns(
        factor_df: pl.DataFrame,
        price_df: pl.DataFrame,
        n_quantiles: int = 5,
        forward_period: int = 5
    ) -> pl.DataFrame:
        """Compute average forward return per factor quantile.

        Returns:
            DataFrame with columns [quantile, mean_return, mean_hit_rate, total_n]
        """
        df = FactorAnalyzer._merge_and_add_fwd_ret(
            factor_df, price_df, forward_period
        )
        if len(df) == 0:
            return pl.DataFrame()

        dates_all = price_df["datetime"].unique().sort()
        records = []

        for dt in dates_all:
            day = df.filter(pl.col("datetime") == dt)
            if day.height < n_quantiles * 2:
                continue

            fwd_vals = day["fwd_ret"].to_numpy()
            factor_vals = day["factor_value"].to_numpy()

            if np.all(factor_vals == factor_vals[0]):
                continue

            # Rank and assign quantile
            ranks = np.argsort(np.argsort(factor_vals))
            qsize = len(factor_vals) / n_quantiles
            qlabels = np.floor(ranks / qsize).clip(0, n_quantiles - 1).astype(int)

            for q in range(n_quantiles):
                mask = qlabels == q
                if mask.sum() == 0:
                    continue
                group = fwd_vals[mask]
                records.append({
                    "date": dt,
                    "quantile": q,
                    "mean_return": float(np.mean(group)),
                    "hit_rate": float(np.mean(group > 0)),
                    "n": int(mask.sum()),
                })

        if not records:
            return pl.DataFrame()

        result = pl.DataFrame(records)
        summary = result.group_by("quantile").agg([
            pl.col("mean_return").mean().alias("mean_return"),
            pl.col("hit_rate").mean().alias("mean_hit_rate"),
            pl.col("n").sum().alias("total_n"),
        ]).sort("quantile")

        return summary

    @staticmethod
    def evaluate_signal(
        signal_df: pl.DataFrame,
        price_df: pl.DataFrame,
        forward_period: int = 5
    ) -> dict:
        """Evaluate prediction signal performance.

        Args:
            signal_df: columns [datetime, vt_symbol, signal]
            price_df:  columns [datetime, vt_symbol, close]

        Returns:
            dict of long/short performance metrics
        """
        # Merge signals with prices, add forward returns
        merged = signal_df.join(
            price_df, on=["datetime", "vt_symbol"], how="inner"
        ).sort(["vt_symbol", "datetime"])

        if len(merged) == 0:
            return {"error": "no data after merge"}

        df = merged.with_columns(
            ((pl.col("close").shift(-forward_period) / pl.col("close")) - 1.0)
            .over("vt_symbol").alias("fwd_ret")
        ).filter(pl.col("fwd_ret").is_not_null())

        dates_all = df["datetime"].unique().sort()
        records = []

        for dt in dates_all:
            day = df.filter(pl.col("datetime") == dt)
            if day.height < 10:
                continue

            sig = day["signal"].to_numpy()
            ret = day["fwd_ret"].to_numpy()

            if np.nanstd(sig) < 1e-10:
                continue

            # Long top 25%, short bottom 25%
            thr_high = np.percentile(sig, 75)
            thr_low = np.percentile(sig, 25)

            long_mask = sig >= thr_high
            short_mask = sig <= thr_low

            records.append({
                "date": dt,
                "n_total": len(day),
                "n_long": int(long_mask.sum()),
                "n_short": int(short_mask.sum()),
                "long_ret": float(np.mean(ret[long_mask])) if long_mask.sum() > 0 else 0.0,
                "short_ret": float(np.mean(ret[short_mask])) if short_mask.sum() > 0 else 0.0,
                "ls_ret": float(np.mean(ret[long_mask]) - np.mean(ret[short_mask]))
                if long_mask.sum() > 0 and short_mask.sum() > 0 else 0.0,
            })

        if not records:
            return {"error": "insufficient data"}

        result = pl.DataFrame(records)
        long_rets = result["long_ret"].to_numpy()
        ls_rets = result["ls_ret"].to_numpy()

        return {
            "period": f"{forward_period}d",
            "n_periods": len(records),
            "long_avg_return": float(np.mean(long_rets)),
            "long_win_rate": float(np.mean(long_rets > 0)),
            "long_short_avg": float(np.mean(ls_rets)),
            "long_short_win_rate": float(np.mean(ls_rets > 0)),
            "long_sharpe": float(np.mean(long_rets) / np.std(long_rets) * np.sqrt(240))
            if np.std(long_rets) > 0 else 0.0,
        }
