"""
Expression calculation engine
"""
import polars as pl
from .data_proxy import DataProxy


def calculate_by_expression(df: pl.DataFrame, expression: str) -> pl.DataFrame:
    """Execute calculation based on expression string"""
    # Import operators locally to avoid polluting global namespace
    from .ts_function import (              # noqa
        ts_delay, ts_min, ts_max,
        ts_argmax, ts_argmin, ts_rank,
        ts_sum, ts_mean, ts_std,
        ts_slope, ts_quantile, ts_rsquare,
        ts_resi, ts_corr, ts_less, ts_greater,
        ts_log, ts_abs, ts_delta, ts_cov,
        ts_decay_linear, ts_product
    )
    from .cs_function import (              # noqa
        cs_rank, cs_mean, cs_std, cs_sum, cs_scale
    )
    from .ta_function import (              # noqa
        ta_rsi, ta_atr
    )
    from .math_function import (              # noqa
        less, greater, log, abs, sign,
        pow1, pow2, quesval, quesval2
    )

    d = dict(locals())
    for column in df.columns:
        if column in {"datetime", "vt_symbol"}:
            continue
        column_df = df[["datetime", "vt_symbol", column]]
        d[column] = DataProxy(column_df)

    other = eval(expression, {}, d)
    return other.df


def calculate_by_polars(df: pl.DataFrame, expression: pl.Expr) -> pl.DataFrame:
    """Execute calculation based on Polars expression"""
    return df.select([
        "datetime", "vt_symbol",
        expression.alias("data")
    ])
