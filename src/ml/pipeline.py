"""
ML Pipeline for Main Surge Detection
--------------------------------------
LightGBM classifier: 163 Alpha factors -> surge probability (>=10% 20d forward).
Temporal walk-forward backtest matching surge engine evaluation dates.

Usage:
    python -m src.ml.pipeline          # quick test (3 stocks)
    python -m src.ml.pipeline full     # full backtest
"""

import sys, os, time, pickle, json
from datetime import datetime
from typing import Optional
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class SurgeMLPipeline:
    """LightGBM pipeline: compute factors -> create labels -> train -> evaluate."""

    LABEL_HORIZON = 20
    SURGE_THRESHOLD = 0.10  # >=10% return -> surge

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_dir = os.path.join(base, "data", "models")
        self.model_dir = model_dir
        self.model = None
        self._factor_names = []
        self.cache_dir = os.path.join(os.path.dirname(model_dir), "cache")
        self._dataset_cache_path = os.path.join(self.cache_dir, "factor_dataset.pkl")

    # ----------------------------------------------------------------
    # 1. Dataset construction
    # ----------------------------------------------------------------

    def build_dataset(
        self,
        prices_dict: dict,
        stock_codes: list[str],
        use_cache: str = None,
    ) -> tuple:
        """
        Convert cached OHLCV -> Polars -> 163 alpha factors -> labels.

        Returns: (X: np.array, y: np.array, meta: pd.DataFrame)
        """
        import pandas as pd
        from src.features.feature_generator import FeatureGenerator

        t0 = time.time()
        codes = [c for c in stock_codes if c in prices_dict]

        # ---- Phase 1: pandas -> Polars (all stocks) ----
        frames = []
        for code in codes:
            pdf = prices_dict[code]
            if pdf.empty or len(pdf) < 80:
                continue
            pldf = pl.from_pandas(pdf.reset_index(drop=True))
            # Normalise column names
            rename = {}
            for col in ["date", "Date"]:
                if col in pldf.columns:
                    rename[col] = "datetime"
            pldf = pldf.rename(rename) if rename else pldf
            if "vt_symbol" not in pldf.columns:
                pldf = pldf.with_columns(pl.lit(code).alias("vt_symbol"))
            frames.append(pldf)

        polars_df = pl.concat(frames)
        n_stocks = polars_df["vt_symbol"].n_unique()
        n_rows = polars_df.shape[0]
        print(f"Converted: {n_rows} rows, {n_stocks} stocks")

        # ---- Phase 2: compute alpha factors ----
        fg = FeatureGenerator(max_workers=1)
        factor_df = fg.compute_all(polars_df)

        base_cols = {"datetime", "vt_symbol", "open", "high", "low", "close",
                      "volume", "amount", "adj_factor", "vwap"}
        factor_cols = [c for c in factor_df.columns if c not in base_cols]
        self._factor_names = factor_cols
        print(f"Factors: {len(factor_cols)} computed ({time.time()-t0:.1f}s total)")

        # ---- Phase 3: forward-return labels (pandas, easier with groupby) ----
        pdf = factor_df.to_pandas()
        pdf["forward_ret"] = np.nan
        pdf = pdf.sort_values(["vt_symbol", "datetime"])

        horizon = self.LABEL_HORIZON
        for _, group in pdf.groupby("vt_symbol"):
            closes = group["close"].values
            rets = np.full(len(group), np.nan)
            for i in range(len(group)):
                j = i + horizon
                if j < len(group):
                    rets[i] = (closes[j] - closes[i]) / closes[i]
            pdf.loc[group.index, "forward_ret"] = rets

        # Default binary label
        pdf["label"] = self._make_binary_label(pdf["forward_ret"])

        # Drop rows with missing factors or label
        valid_cols = factor_cols + ["label"]
        pdf_valid = pdf.dropna(subset=valid_cols)

        X = pdf_valid[factor_cols].astype(np.float32).values
        y = pdf_valid["label"].astype(np.int32).values
        meta = pdf_valid[["vt_symbol", "datetime", "forward_ret", "label"]].reset_index(drop=True)

        surge_rate = y.mean()
        print(f"Labels: {len(X)} samples, surge_rate={surge_rate:.2%}")

        return X, y, meta

    # ----------------------------------------------------------------
    # 1b. Label definitions (swappable)
    # ----------------------------------------------------------------

    @staticmethod
    def _make_binary_label(forward_ret, threshold: float = 0.10) -> np.ndarray:
        """Binary: forward_return >= threshold"""
        return (forward_ret >= threshold).astype(int)

    @staticmethod
    def _make_rank_label(forward_ret, group_key, top_quantile: float = 0.20) -> np.ndarray:
        """Rank-based: top 20% performers at each date get label=1"""
        import pandas as pd
        df = pd.DataFrame({"ret": forward_ret, "group": group_key})
        labels = df.groupby("group")["ret"].transform(
            lambda x: (x >= x.quantile(1 - top_quantile)).astype(int)
        )
        return labels.values

    @staticmethod
    def _make_regression_y(forward_ret) -> np.ndarray:
        """Regression: predict forward return directly"""
        return forward_ret.values.astype(np.float32)

    def get_labels(self, meta, method: str = "binary", **kwargs):
        """
        Compute labels from meta (which has forward_ret).

        Args:
            meta: DataFrame with columns ['vt_symbol', 'datetime', 'forward_ret']
            method: 'binary' | 'rank' | 'regression'

        Returns:
            y: np.ndarray (1D)
            task: str ('binary' or 'regression') for LightGBM objective
        """
        import pandas as pd
        meta = meta.reset_index(drop=True)
        fr = meta["forward_ret"]

        if method == "binary":
            th = kwargs.get("threshold", self.SURGE_THRESHOLD)
            return self._make_binary_label(fr, th), "binary"
        elif method == "rank":
            q = kwargs.get("top_quantile", 0.20)
            return self._make_rank_label(fr, meta["datetime"], q), "binary"
        elif method == "regression":
            return self._make_regression_y(fr), "regression"
        else:
            raise ValueError(f"Unknown label method: {method}")

    # ----------------------------------------------------------------
    # 1c. Dataset cache
    # ----------------------------------------------------------------

    def build_and_cache(self, prices_dict: dict, stock_codes: list[str]):
        """Build dataset and save to disk cache."""
        X, y, meta = self.build_dataset(prices_dict, stock_codes)
        data = {"X": X, "y": y, "meta": meta, "factor_names": self._factor_names}
        os.makedirs(os.path.dirname(self._dataset_cache_path), exist_ok=True)
        with open(self._dataset_cache_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Dataset cached: {self._dataset_cache_path} ({len(X)} samples)")
        return data

    def load_cached_dataset(self) -> Optional[dict]:
        """Load cached factor dataset."""
        if not os.path.exists(self._dataset_cache_path):
            print(f"No cached dataset at {self._dataset_cache_path}")
            return None
        with open(self._dataset_cache_path, "rb") as f:
            data = pickle.load(f)
        self._factor_names = data.get("factor_names", [])
        print(f"Loaded cached dataset: {len(data['X'])} samples, {len(self._factor_names)} factors")
        return data

    # ----------------------------------------------------------------
    # 2. Training
    # ----------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        lgb_params: dict = None,
        task: str = "binary",
    ):
        """Train LightGBM (binary or regression) with validation."""
        import lightgbm as lgb

        is_binary = task == "binary"
        default = {
            "objective": "binary" if is_binary else "regression",
            "metric": "auc" if is_binary else "l2",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "max_depth": 8,
            "learning_rate": 0.03,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 50,
            "min_split_gain": 0.1,
            "random_state": 42,
            "verbosity": -1,
        }
        if lgb_params:
            default.update(lgb_params)

        dtrain = lgb.Dataset(X, label=y)
        valid_sets, valid_names = [dtrain], ["train"]

        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            valid_sets = [dval]
            valid_names = ["valid"]

        if is_binary:
            surge_rate = y.mean()
            print(f"Training LightGBM ({len(X)} samples, {surge_rate:.2%} surge)...")
        else:
            print(f"Training LightGBM regression ({len(X)} samples)...")

        callbacks = [
            lgb.early_stopping(60),
            lgb.log_evaluation(100),
        ]

        self.model = lgb.train(
            default,
            dtrain,
            num_boost_round=800,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        print(f"Done: best_iter={self.model.best_iteration}, best_score={self.model.best_score}")
        return self.model

    # ----------------------------------------------------------------
    # 3. Walk-forward backtest
    # ----------------------------------------------------------------

    def walk_forward_backtest(
        self, X: np.ndarray, y: np.ndarray, meta, eval_dates: list
    ) -> dict:
        """
        For each eval date: train on prior data, predict, rank top-5.
        """
        import pandas as pd
        import lightgbm as lgb

        print(f"Walk-forward: {len(eval_dates)} dates...")
        meta = meta.copy()
        meta["datetime_pd"] = pd.to_datetime(meta["datetime"])

        results = {}
        wf_params = {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 31,
            "max_depth": 5,              # 限制树深度 — 反过拟合
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "lambda_l1": 0.1,            # L1 正则 — 稀疏特征选择
            "lambda_l2": 0.1,            # L2 正则 — 权重衰减
            "min_child_samples": 100,     # 叶子最小样本 — 防过拟合
            "random_state": 42,
            "verbosity": -1,
        }

        for ed in eval_dates:
            ed_pd = pd.Timestamp(datetime.strptime(ed, "%Y%m%d"))
            mask_train = meta["datetime_pd"] <= ed_pd
            mask_test = meta["datetime_pd"] == ed_pd

            if mask_test.sum() < 5 or mask_train.sum() < 500:
                continue

            X_tr, y_tr = X[mask_train], y[mask_train]
            X_te = X[mask_test]

            m = lgb.train(
                wf_params,
                lgb.Dataset(X_tr, label=y_tr),
                num_boost_round=150,
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
            )
            probs = m.predict(X_te)
            scored = list(zip(probs, meta[mask_test]["vt_symbol"].values))
            scored.sort(key=lambda x: x[0], reverse=True)
            top5 = scored[:5]

            results[ed] = {
                "n_test": len(scored),
                "top5_probs": [round(float(s), 4) for s, _ in top5],
                "top5_codes": [c for _, c in top5],
            }

        return results

    # ----------------------------------------------------------------
    # 4. Feature importance
    # ----------------------------------------------------------------

    def feature_importance(self) -> "pd.DataFrame":
        import pandas as pd
        imp = pd.DataFrame({
            "factor": self._factor_names[:self.model.num_feature()],
            "gain": self.model.feature_importance(importance_type="gain"),
            "split": self.model.feature_importance(importance_type="split"),
        }).sort_values("gain", ascending=False)
        return imp

    # ----------------------------------------------------------------
    # 5. Save / load
    # ----------------------------------------------------------------

    def save(self, path: str = None):
        if self.model is None:
            raise ValueError("No model.")
        p = path or os.path.join(self.model_dir, "surge_lgbm.pkl")
        with open(p, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved: {p}")
        return p

    def load(self, path: str = None):
        p = path or os.path.join(self.model_dir, "surge_lgbm.pkl")
        if not os.path.exists(p):
            raise FileNotFoundError(f"No model at {p}")
        with open(p, "rb") as f:
            self.model = pickle.load(f)
        print(f"Model loaded: {p} (features={self.model.num_feature()})")
        return self.model

    # ----------------------------------------------------------------
    # 6. Run full pipeline
    # ----------------------------------------------------------------

    def run(self, prices_dict: dict, stock_pool: list,
            temporal_split: float = 0.7) -> dict:
        t0 = time.time()

        X, y, meta = self.build_dataset(prices_dict, stock_pool)

        # Temporal split: use quantile to avoid hardcoded date issues
        import pandas as pd
        meta["dt"] = pd.to_datetime(meta["datetime"])
        split_dt = meta["dt"].quantile(temporal_split)
        train_mask = (meta["dt"] < split_dt).values
        test_mask = ~train_mask

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]

        print(f"Split: {train_mask.sum()} train, {test_mask.sum()} test")

        # Train
        self.train(X_tr, y_tr, X_val=X_te, y_val=y_te)

        # Threshold analysis
        probs = self.model.predict(X_te)
        meta_te = meta[test_mask].copy()

        print("\n  Threshold -> picks | win | avg_ret:")
        best = {"th": 0, "excess": -99.0}
        for th in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            picks = meta_te[probs >= th]
            if len(picks) < 5:
                continue
            wr = (picks["forward_ret"] > 0).mean()
            ar = picks["forward_ret"].mean()
            print(f"    {th:.2f}  -> {len(picks):>5} | {wr*100:.1f}% | {ar*100:+.2f}%")
            if ar > best.get("excess", -99.0):
                best = {"th": th, "excess": round(float(ar), 4), "n": len(picks), "win": round(wr, 3)}

        fp = self.feature_importance()
        top10 = fp.head(10)["factor"].tolist()

        self.save()
        elapsed = time.time() - t0

        return {
            "runtime_s": round(elapsed, 1),
            "dataset": {"samples": len(X), "factors": X.shape[1],
                       "surge_rate": round(float(y.mean()), 3)},
            "best_threshold": best,
            "top10_factors": top10,
        }

    # ----------------------------------------------------------------
    # 6b. Run with swappable label methods
    # ----------------------------------------------------------------

    def run_with_labels(
        self,
        prices_dict: dict,
        stock_pool: list,
        label_methods: list[str] = None,
        temporal_split: float = 0.7,
    ) -> dict:
        """
        Compare different label definitions on the same dataset.
        Caches dataset after first build.

        Args:
            label_methods: ['binary', 'rank', 'regression'] or subset

        Returns:
            {label_method: {auc, excess_return, picks, ...}}
        """
        if label_methods is None:
            label_methods = ["binary", "rank", "regression"]

        t0 = time.time()

        # Build or load cached dataset
        cached = self.load_cached_dataset()
        if cached is not None:
            X_factors = cached["X"]
            meta = cached["meta"]
            self._factor_names = cached["factor_names"]
        else:
            X_factors, _, meta = self.build_dataset(prices_dict, stock_pool)
            self.build_and_cache(prices_dict, stock_pool)

        import pandas as pd
        meta["dt"] = pd.to_datetime(meta["datetime"])
        split_dt = meta["dt"].quantile(temporal_split)
        train_mask = (meta["dt"] < split_dt).values
        test_mask = ~train_mask

        results = {}
        for method in label_methods:
            print(f"\n{'='*50}")
            print(f"Label method: {method}")
            print('='*50)

            y, task = self.get_labels(meta, method)
            X_tr, y_tr = X_factors[train_mask], y[train_mask]
            X_te, y_te = X_factors[test_mask], y[test_mask]

            # Train
            self.train(X_tr, y_tr, X_val=X_te, y_val=y_te, task=task)

            # Predict
            scores = self.model.predict(X_te)
            meta_te = meta[test_mask].copy()
            meta_te["score"] = scores
            meta_te["forward_ret"] = meta_te["forward_ret"].astype(float)

            # Evaluate (common metric: forward return of top picks)
            print("\n  Threshold -> picks | win% | avg_ret | excess:")
            best = {"th": 0, "excess": -99.0}
            for th in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
                picks = meta_te[meta_te["score"] >= th]
                if len(picks) < 5:
                    continue
                wr = (picks["forward_ret"] > 0).mean()
                ar = picks["forward_ret"].mean()
                print(f"    {th:.2f}  -> {len(picks):>5} | {wr*100:.1f}% | {ar*100:+.2f}% |")
                if ar > best["excess"]:
                    best = {"th": th, "excess": round(float(ar), 4),
                            "n": len(picks), "win": round(wr, 3)}

            results[method] = best
            print(f"\n  Best: th={best['th']}, excess={best['excess']*100:+.2f}%, "
                  f"win={best['win']*100:.1f}%, picks={best['n']}")

        # Summary
        print(f"\n{'='*50}")
        print("LABEL COMPARISON SUMMARY")
        print('='*50)
        for method, r in sorted(results.items(), key=lambda x: -x[1]["excess"]):
            print(f"  {method:12s}: excess={r['excess']*100:+6.2f}%  win={r['win']*100:5.1f}%  "
                  f"th={r['th']:.2f}  picks={r['n']}")
        print(f"Total: {time.time()-t0:.0f}s")

        return results


# ----------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(msg)s")

    try:
        with open("data/cache/backtest_prices.pkl", "rb") as f:
            prices_cache = pickle.load(f)

        full_mode = len(sys.argv) > 1 and sys.argv[1] == "full"
        from src.tools.data_fetcher import get_16_sector_stocks
        sec = get_16_sector_stocks()
        pool_all = sorted(set(sum((v[:30] for v in sec.values()), [])))
        pool = pool_all if full_mode else pool_all[:10]

        print(f"Pipeline: {len(pool)} stocks (full={full_mode})\n")
        pln = SurgeMLPipeline()
        report = pln.run(prices_cache, pool)

        # Save report JSON
        report_path = os.path.join(pln.model_dir, "surge_ml_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved: {report_path}")

        print("\n" + "=" * 50)
        print(f"  Best: th={report['best_threshold']['th']}, "
              f"excess={report['best_threshold']['excess']*100:+.2f}%, "
              f"picks={report['best_threshold']['n']}")
        print("=" * 50)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
