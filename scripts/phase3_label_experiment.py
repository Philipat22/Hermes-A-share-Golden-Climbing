"""
Phase 3: Label Definition Comparison
=====================================
Loads cached factor dataset, experiments with different label definitions:
  - binary:  forward_ret >= 10%  (baseline)
  - rank:    top 20% performers each date (normalized for market regime)
  - regression: predict forward_return directly

Usage:
    python scripts/phase3_label_experiment.py
    python scripts/phase3_label_experiment.py --methods binary rank
"""
import sys, os, time, pickle, json, warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CACHE_FILE = os.path.join(ROOT, "data", "cache", "factor_dataset.pkl")
MODEL_DIR = os.path.join(ROOT, "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset():
    if not os.path.exists(CACHE_FILE):
        print(f"[ERROR] No cached dataset. Run phase1 first.")
        sys.exit(1)
    with open(CACHE_FILE, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded: {len(data['X'])} samples, {len(data['factor_names'])} factors")
    return data


def run_experiment(data, methods, best_params):
    """Run label experiments and compare."""
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    meta = data["meta"].copy()
    meta["dt"] = pd.to_datetime(meta["datetime"])
    split_dt = meta["dt"].quantile(0.7)
    train_mask = (meta["dt"] < split_dt).values
    test_mask = ~train_mask

    results = {}
    t0 = time.time()

    for method in methods:
        print(f"\n{'='*50}")
        print(f"Label method: {method}")
        print('='*50)

        # Get y and task
        from src.ml.pipeline import SurgeMLPipeline
        pln = SurgeMLPipeline()
        y, task = pln.get_labels(data["meta"], method)
        task_for_train = "binary" if task in ("binary", "rank") else task

        X_tr = data["X"][train_mask]
        y_tr = y[train_mask]
        X_te = data["X"][test_mask]
        y_te = y[test_mask]

        print(f"Split: {len(X_tr)} train, {len(X_te)} test")
        if task == "binary":
            print(f"Positive rate: {y_tr.mean():.2%} train, {y_te.mean():.2%} test")

        # Train params
        base_params = {
            "objective": "binary" if task_for_train == "binary" else "regression",
            "metric": "auc" if task_for_train == "binary" else "l2",
            "boosting_type": "gbdt",
            "num_leaves": best_params.get("nl", 31),
            "max_depth": best_params.get("md", 6),
            "learning_rate": best_params.get("lr", 0.03),
            "feature_fraction": best_params.get("ff", 0.6),
            "bagging_fraction": best_params.get("bf", 0.8),
            "bagging_freq": 5,
            "min_child_samples": 50,
            "min_split_gain": 0.1,
            "random_state": 42,
            "verbosity": -1,
        }

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_te, label=y_te, reference=dtrain)

        model = lgb.train(
            base_params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        scores = model.predict(X_te)

        # Evaluate: forward return of top picks
        mte = meta[test_mask].copy()
        mte["score"] = scores
        mte["forward_ret"] = mte["forward_ret"].astype(float)

        print(f"\n  Threshold -> picks | win% | avg_ret:")
        best = {"th": 0, "excess": -99.0}

        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

        # For regression, use quantile-based thresholds
        if task == "regression":
            thresholds = [float(scores.mean() + i * scores.std() * 0.5)
                         for i in range(-2, 4)]

        for th in thresholds:
            picks = mte[mte["score"] >= th]
            if len(picks) < 5:
                continue
            wr = (picks["forward_ret"] > 0).mean()
            ar = picks["forward_ret"].mean()
            print(f"    {th:.2f}  -> {len(picks):>5} | {wr*100:.1f}% | {ar*100:+.2f}%")
            if ar > best["excess"]:
                best = {"th": th, "excess": round(float(ar), 4),
                        "n": len(picks), "win": round(wr, 3)}

        results[method] = best
        print(f"\n  >> Best: excess={best['excess']*100:+.2f}%  "
              f"win={best['win']*100:.1f}%  th={best['th']:.2f}  picks={best['n']}")

        # Save intermediate
        with open(os.path.join(MODEL_DIR, f"model_{method}.pkl"), "wb") as f:
            pickle.dump(model, f)
        print(f"  Model saved: model_{method}.pkl")

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print("LABEL COMPARISON RESULTS")
    print('='*50)
    for method, r in sorted(results.items(), key=lambda x: -x[1]["excess"]):
        print(f"  {method:12s}: excess={r['excess']*100:+6.2f}%  "
              f"win={r['win']*100:5.1f}%  th={r['th']:.2f}  picks={r['n']}")
    print(f"Total time: {elapsed:.0f}s")

    # Save report
    report = {m: {"excess": r["excess"], "win": r["win"],
                   "threshold": r["th"], "n_picks": r["n"]}
              for m, r in sorted(results.items(), key=lambda x: -x[1]["excess"])}
    with open(os.path.join(MODEL_DIR, "label_experiment_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: label_experiment_report.json")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+",
                        default=["binary", "rank", "regression"],
                        help="Label methods to compare")
    parser.add_argument("--load-params", type=str, default=None,
                        help="JSON file with best params from Phase 2")
    args = parser.parse_args()

    print("=" * 50)
    print("Phase 3: Label Experiment Comparison")
    print("=" * 50)

    # Load best params from Phase 2 if available
    best_params = {}
    if args.load_params and os.path.exists(args.load_params):
        with open(args.load_params) as f:
            report = json.load(f)
        best_params = report.get("best_params", {})
        print(f"Loaded params: {best_params}")
    elif os.path.exists(os.path.join(MODEL_DIR, "tune_report.json")):
        with open(os.path.join(MODEL_DIR, "tune_report.json")) as f:
            report = json.load(f)
        best_params = report.get("best_params", {})
        print(f"Loaded params from tune_report: {best_params}")
    else:
        print("Using default params (no Phase 2 results found)")

    data = load_dataset()
    results = run_experiment(data, args.methods, best_params)

    # Determine best model
    best_method = max(results, key=lambda m: results[m]["excess"])
    print(f"\n>>> Best label method: {best_method} (excess={results[best_method]['excess']*100:+.2f}%)")

    # Copy best model to production path
    best_model_path = os.path.join(MODEL_DIR, f"model_{best_method}.pkl")
    prod_path = os.path.join(MODEL_DIR, "surge_lgbm.pkl")
    import shutil
    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, prod_path)
        print(f">>> Production model updated: {prod_path}")
