"""
Surge ML Predictor — live inference for analyze_stock integration.

Loads a trained LightGBM model, computes 163 alpha factors
for a single stock, returns ml_score (0-100 surge probability).

Integration (in engine.py analyze_stock):
    from src.ml.predictor import get_predictor
    predictor = get_predictor()
    ml_score = predictor.score(df)
"""

import sys, os, pickle
from typing import Optional

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Singleton instance
_predictor: Optional["SurgePredictor"] = None


class SurgePredictor:
    """
    Single-stock surge probability predictor.

    Usage:
        p = SurgePredictor()
        p.load("data/models/surge_lgbm.pkl")
        score = p.score(stock_df)  # 0-100, higher = more likely to surge
    """

    def __init__(self, model_path: str = None, max_workers: int = 1):
        self.model = None
        self.factor_names: list[str] = []
        self.model_path = model_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "models", "surge_lgbm.pkl",
        )

    # -----------------------------------------------------------------

    def load(self, path: str = None):
        """Load trained LightGBM model from pickle."""
        p = path or self.model_path
        if not os.path.exists(p):
            return False
        with open(p, "rb") as f:
            self.model = pickle.load(f)
        num = self.model.num_feature()
        print(f"[predictor] Model loaded: {p} (features={num})")
        return True

    # -----------------------------------------------------------------

    def _compute_factors(self, df) -> "pl.DataFrame":
        """
        Convert a stock's price DataFrame (pandas) to Polars,
        then compute all 163 alpha factors.
        """
        from src.features.feature_generator import FeatureGenerator

        if isinstance(df, pl.DataFrame):
            pldf = df.clone()
        else:
            pldf = pl.from_pandas(df.reset_index(drop=True))

        # Normalise columns
        rename = {}
        for col in ["date", "Date"]:
            if col in pldf.columns:
                rename[col] = "datetime"
        pldf = pldf.rename(rename) if rename else pldf

        fg = FeatureGenerator(max_workers=1)  # single-process for inference
        result = fg.compute_all(pldf)

        # Track factor column names (once)
        if not self.factor_names:
            self.factor_names = [c for c in result.columns if c.startswith("alpha")]

        return result

    # -----------------------------------------------------------------

    def predict(self, df) -> float:
        """
        Predict surge probability for a single stock.

        Args:
            df: pandas DataFrame with columns [date, open, high, low, close, volume]
                or Polars DataFrame with same structure.

        Returns:
            Probability (0-1) of >=10% 20d forward return.
        """
        if self.model is None:
            if not self.load():
                return 0.0

        factor_df = self._compute_factors(df)

        # Extract the LAST row's factor values (current point-in-time)
        latest = factor_df.select(self.factor_names).tail(1)
        if latest.is_empty():
            return 0.0

        values = latest.to_numpy()[0].astype(np.float32)

        # Handle NaN
        nan_mask = np.isnan(values)
        if nan_mask.all():
            return 0.0
        values[nan_mask] = 0.0

        prob = float(self.model.predict([values])[0])
        return round(prob, 4)

    # -----------------------------------------------------------------

    def score(self, df) -> int:
        """
        Convenience: predict() -> 0-100 integer score.
        Ready for integration into engine's weighted scoring.
        """
        prob = self.predict(df)
        return min(100, max(0, int(prob * 100)))

    # -----------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if a trained model exists and can be loaded."""
        if self.model is not None:
            return True
        return os.path.exists(self.model_path)


# ---------------------------------------------------------------------
# Singleton factory (for engine integration)
# ---------------------------------------------------------------------

def get_predictor(force_reload: bool = False) -> SurgePredictor:
    """
    Get or create the global predictor instance.
    Call this from analyze_stock to get ml_score.
    """
    global _predictor
    if _predictor is None or force_reload:
        _predictor = SurgePredictor()
        _predictor.load()
    return _predictor


def batch_predict_scores(prices_dict: dict, eval_dates: list[str],
                          pool: list[str] = None,
                          model_path: str = None) -> dict[str, dict[str, int]]:
    """
    Pre-compute ML scores for all stocks at all eval dates in one batch.

    Strategy: compute 160 factors once for all stocks, predict ALL rows,
    then group by date for O(1) lookup during backtest.

    Returns: {eval_date_str: {ts_code: ml_score_0to100}}
    """
    import time
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    from src.features.feature_generator import FeatureGenerator
    from src.ml.pipeline import SurgeMLPipeline

    t0 = time.time()

    if pool is None:
        pool = [c for c in prices_dict if c in prices_dict and prices_dict[c] is not None]

    codes = [c for c in pool if c in prices_dict and prices_dict[c] is not None]
    print(f"[batch_predict] Building dataset for {len(codes)} stocks...")

    # Cache path for factor data (avoids recomputing on re-runs)
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "cache")
    factor_cache = os.path.join(cache_dir, "backtest_factors.pkl")

    if os.path.exists(factor_cache):
        print("[batch_predict] Loading cached factors...")
        with open(factor_cache, "rb") as f:
            X, y, meta, factor_cols = pickle.load(f)
        print(f"[batch_predict] Cached: {len(X)} rows, {len(factor_cols)} factors")
    else:
        # Reuse pipeline's build_dataset which computes all 160 factors
        pln = SurgeMLPipeline()
        X, y, meta = pln.build_dataset(prices_dict, codes)
        factor_cols = pln._factor_names

        if len(X) == 0:
            print("[batch_predict] No data, returning empty")
            return {}

        print(f"[batch_predict] Dataset: {len(X)} rows, {len(factor_cols)} factors. Caching...")
        os.makedirs(cache_dir, exist_ok=True)
        with open(factor_cache, "wb") as f:
            pickle.dump((X, y, meta, factor_cols), f)
        print(f"[batch_predict] Cached to {factor_cache}")

    print(f"[batch_predict] Loading model...")

    # Load model
    mp = model_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "models", "surge_lgbm.pkl")
    with open(mp, "rb") as f:
        model = pickle.load(f)

    # Batch predict
    X_pred = pd.DataFrame(X, columns=factor_cols).fillna(0).astype(np.float32).values
    probs = model.predict(X_pred)

    print(f"[batch_predict] Predicted {len(probs)} rows in {time.time()-t0:.0f}s")

    # Build lookup: {date: {code: score}}
    sorted_dates = sorted(set(meta["datetime"]))
    # Filter to eval dates only
    eval_set = set(eval_dates)
    # meta datetime is like '2025-01-05' format — convert to YYYYMMDD
    def _fmt_date(d):
        dstr = str(d).replace("-", "").replace(" ", "").replace("T", "")[:8]
        return dstr

    result: dict[str, dict[str, int]] = {}
    meta["_date_yyyymmdd"] = meta["datetime"].apply(_fmt_date)

    for ed in sorted_dates:
        ed_yyyymmdd = _fmt_date(ed)
        if ed_yyyymmdd not in eval_set:
            continue

        mask = meta["_date_yyyymmdd"] == ed_yyyymmdd
        if mask.sum() == 0:
            continue

        idxs = np.where(mask)[0]
        scores = {}
        for idx in idxs:
            code = meta.iloc[idx]["vt_symbol"]
            scores[code] = min(100, int(probs[idx] * 100))
        result[ed_yyyymmdd] = scores

    print(f"[batch_predict] Done: {len(result)} eval dates with scores in {time.time()-t0:.0f}s")
    return result
