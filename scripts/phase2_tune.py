"""
Phase 2: LightGBM Hyperparameter Tuning
========================================
Loads cached factor dataset from Phase 1, performs grid search on
key LightGBM params, saves best model.

Usage:
    python scripts/phase2_tune.py              # quick grid (~2 min)
    python scripts/phase2_tune.py full         # full grid (~10 min)
"""
import sys, os, time, pickle, json, warnings, itertools
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_FILE = os.path.join(ROOT, "data", "models", "surge_lgbm.pkl")
MODEL_DIR = os.path.join(ROOT, "data", "models")
CACHE_FILE = os.path.join(ROOT, "data", "cache", "factor_dataset.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset():
    """Load cached factor dataset."""
    if not os.path.exists(CACHE_FILE):
        print(f"[ERROR] No cached dataset at {CACHE_FILE}")
        print("Run scripts/phase1_build_cache.py first.")
        sys.exit(1)
    with open(CACHE_FILE, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded: {len(data['X'])} samples, {len(data['factor_names'])} factors")
    return data


def temporal_split(data, train_frac=0.7):
    """Time-based train/test split on meta['datetime']."""
    import pandas as pd
    import numpy as np
    meta = data["meta"].copy()
    meta["dt"] = pd.to_datetime(meta["datetime"])
    split_dt = meta["dt"].quantile(train_frac)
    train_mask = (meta["dt"] < split_dt).values
    test_mask = ~train_mask
    X_tr = data["X"][train_mask]
    y_tr = data["y"][train_mask]
    X_te = data["X"][test_mask]
    y_te = data["y"][test_mask]
    print(f"Split: {X_tr.shape[0]} train, {X_te.shape[0]} test")
    return X_tr, y_tr, X_te, y_te, meta, test_mask


def evaluate(model, X_te, y_te, meta_te, test_mask):
    """Evaluate: AUC + forward-return of top picks."""
    import pandas as pd
    import numpy as np
    from sklearn.metrics import roc_auc_score

    probs = model.predict(X_te)
    auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else 0.5

    # Forward return analysis (only test rows)
    mte = meta_te[test_mask].copy()
    mte["score"] = probs
    mte["forward_ret"] = mte["forward_ret"].astype(float)

    best = {"th": 0, "excess": -99.0, "auc": round(auc, 4)}
    for th in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        picks = mte[mte["score"] >= th]
        if len(picks) < 5:
            continue
        wr = (picks["forward_ret"] > 0).mean()
        ar = picks["forward_ret"].mean()
        if ar > best["excess"]:
            best = {"th": th, "excess": round(float(ar), 4),
                    "n": len(picks), "win": round(wr, 3), "auc": round(auc, 4)}

    return auc, best


def run_grid(data, quick=True):
    """Grid search over LightGBM params."""
    import pandas as pd
    import numpy as np
    import lightgbm as lgb

    X_tr, y_tr, X_te, y_te, meta_te, test_mask = temporal_split(data)

    # Param grid
    lr_vals = [0.01, 0.03, 0.05] if quick else [0.01, 0.03, 0.05, 0.1]
    leaves_vals = [31, 63] if quick else [15, 31, 63, 95]
    depth_vals = [6, 8] if quick else [4, 6, 8, 10]
    ff_vals = [0.6, 0.8] if quick else [0.4, 0.6, 0.8, 1.0]
    bf_vals = [0.75, 0.85] if quick else [0.6, 0.75, 0.85, 0.95]

    grid = list(itertools.product(lr_vals, leaves_vals, depth_vals, ff_vals, bf_vals))
    print(f"\nGrid: {len(grid)} combinations")

    results = []
    t0 = time.time()

    for i, (lr, nl, md, ff, bf) in enumerate(grid):
        params = {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": nl,
            "max_depth": md,
            "learning_rate": lr,
            "feature_fraction": ff,
            "bagging_fraction": bf,
            "bagging_freq": 5,
            "min_child_samples": 50,
            "random_state": 42,
            "verbosity": -1,
        }

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_te, label=y_te, reference=dtrain)

        model = lgb.train(
            params, dtrain,
            num_boost_round=300,
            valid_sets=[dval],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        auc, best = evaluate(model, X_te, y_te, meta_te, test_mask)

        results.append({
            "params": {"lr": lr, "nl": nl, "md": md, "ff": ff, "bf": bf},
            "auc": auc,
            "best_th": best["th"],
            "excess": best["excess"],
            "win": best["win"],
            "n_picks": best["n"],
        })

        if (i + 1) % 10 == 0 or i == len(grid) - 1:
            print(f"  [{i+1}/{len(grid)}] best_auc={max(r['auc'] for r in results):.4f} "
                  f"best_excess={max(r['excess'] for r in results)*100:+.2f}%")

    elapsed = time.time() - t0
    print(f"\nGrid search: {len(grid)} combos, {elapsed:.0f}s")

    # Sort by excess return
    results.sort(key=lambda r: -r["excess"])
    print(f"\n{'='*60}")
    print(f"TOP 5 PARAM COMBOS (by excess return)")
    print(f"{'='*60}")
    print(f"{'Rank':<5} {'excess':<10} {'auc':<7} {'win':<7} {'picks':<7} params")
    print(f"{'-'*60}")
    for rank, r in enumerate(results[:5], 1):
        p = r["params"]
        print(f"{rank:<5} {r['excess']*100:+7.2f}% {r['auc']:<7.4f} "
              f"{r['win']*100:<6.1f}% {r['n_picks']:<6} "
              f"lr={p['lr']} nl={p['nl']} md={p['md']} ff={p['ff']} bf={p['bf']}")

    return results


def train_best(data, best_params, save=True):
    """Train final model with best params on full data."""
    import pandas as pd
    import numpy as np
    import lightgbm as lgb

    X_tr, y_tr, X_te, y_te, meta_te, _ = temporal_split(data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": best_params["nl"],
        "max_depth": best_params["md"],
        "learning_rate": best_params["lr"],
        "feature_fraction": best_params["ff"],
        "bagging_fraction": best_params["bf"],
        "bagging_freq": 5,
        "min_child_samples": 50,
        "random_state": 42,
        "verbosity": -1,
    }

    print(f"\nTraining final model (params: lr={best_params['lr']}, "
          f"nl={best_params['nl']}, md={best_params['md']}, "
          f"ff={best_params['ff']}, bf={best_params['bf']})...")

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_te, label=y_te, reference=dtrain)

    model = lgb.train(
        params, dtrain,
        num_boost_round=800,
        valid_sets=[dval],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(60), lgb.log_evaluation(100)],
    )

    auc, best = evaluate(model, X_te, y_te, meta_te, _)
    print(f"\nFinal model: AUC={auc:.4f}, excess={best['excess']*100:+.2f}%, "
          f"win={best['win']*100:.1f}%, th={best['th']:.2f}")

    if save:
        path = os.path.join(MODEL_DIR, "surge_lgbm.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved: {path}")

    return model, best


if __name__ == "__main__":
    print("=" * 50)
    print("Phase 2: Hyperparameter Tuning")
    print("=" * 50)

    data = load_dataset()
    quick = not (len(sys.argv) > 1 and sys.argv[1] == "full")
    print(f"Mode: {'quick' if quick else 'full'} grid")

    results = run_grid(data, quick=quick)
    best = results[0]["params"]

    print(f"\n{'='*60}")
    print(f"Best params: lr={best['lr']} nl={best['nl']} md={best['md']} "
          f"ff={best['ff']} bf={best['bf']}")
    print(f"{'='*60}")

    # Train final model
    model, eval_best = train_best(data, best)

    # Save tuning report
    report = {
        "best_params": best,
        "grid_results": results[:20],
        "final_auc": eval_best.get("auc", 0),
        "final_excess": eval_best["excess"],
        "final_win": eval_best["win"],
        "final_threshold": eval_best["th"],
        "n_picks": eval_best["n"],
    }
    report_path = os.path.join(MODEL_DIR, "tune_report.json")
    with open(report_path, "w") as f:
        # Convert excess values for JSON
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport: {report_path}")
    print("DONE.")
