"""
Alpha158 factor expressions - Qlib-style factor set.
Compact dictionary: {name: expression_string}
"""
import polars as pl

# Keep expression building simple - all factors added programmatically
ALPHA_158 = {
    # K-line features
    "kmid": "(close - open) / open",
    "klen": "(high - low) / open",
    "kmid_2": "(close - open) / (high - low + 1e-12)",
    "kup": "(high - ts_greater(open, close)) / open",
    "kup_2": "(high - ts_greater(open, close)) / (high - low + 1e-12)",
    "klow": "(ts_less(open, close) - low) / open",
    "klow_2": "((ts_less(open, close) - low) / (high - low + 1e-12))",
    "ksft": "(close * 2 - high - low) / open",
    "ksft_2": "(close * 2 - high - low) / (high - low + 1e-12)",
}

# Price relative features
for field in ["open", "high", "low", "vwap"]:
    ALPHA_158[f"{field}_0"] = f"{field} / close"

# Time-series features at multiple windows
_windows = [5, 10, 20, 30, 60]

for w in _windows:
    for prefix, expr_template in [
        (f"roc_{w}", f"ts_delay(close, {w}) / close"),
        (f"ma_{w}", f"ts_mean(close, {w}) / close"),
        (f"std_{w}", f"ts_std(close, {w}) / close"),
        (f"beta_{w}", f"ts_slope(close, {w}) / close"),
        (f"rsqr_{w}", f"ts_rsquare(close, {w})"),
        (f"resi_{w}", f"ts_resi(close, {w}) / close"),
        (f"max_{w}", f"ts_max(high, {w}) / close"),
        (f"min_{w}", f"ts_min(low, {w}) / close"),
        (f"qtlu_{w}", f"ts_quantile(close, {w}, 0.8) / close"),
        (f"qtld_{w}", f"ts_quantile(close, {w}, 0.2) / close"),
        (f"rank_{w}", f"ts_rank(close, {w})"),
        (f"rsv_{w}", f"(close - ts_min(low, {w})) / (ts_max(high, {w}) - ts_min(low, {w}) + 1e-12)"),
        (f"imax_{w}", f"ts_argmax(high, {w}) / {w}"),
    ]:
        ALPHA_158[prefix] = expr_template

ALPHA_158_COUNT: int = len(ALPHA_158)
