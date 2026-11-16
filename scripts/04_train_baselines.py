"""Step 4: Train shallow learned baselines.

Trains three regressors on the 8 static features (log(sigma) target):
  - GradientBoosting
  - RandomForest
  - MLP (2-layer, same as src/models.MLPRanker)

Reports Kendall's tau on the test split for each, plus the single-feature
tau values as a reference. These are the baseline numbers reported in
Table 1 of the paper.

Outputs
-------
results/baselines.json : per-model test tau, std across seeds, CI
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from scipy.stats import kendalltau
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from src.models import MLPRanker
from src.training import train_model
from src.metrics import bootstrap_ci, summarize


def load_data():
    processed = Path("data/processed")
    X = np.load(processed / "X_static.npy")
    sigma_full = np.load(processed / "sigma_sir.npy")
    valid_nodes = np.load(processed / "valid_nodes.npy")
    y_sigma = sigma_full[valid_nodes].astype(np.float32)
    y_log = np.log(y_sigma).astype(np.float32)
    split = np.load(processed / "split.npz")
    return X, y_sigma, y_log, split


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    X, y_sigma, y_log, split = load_data()
    train_idx = split["train"]
    val_idx = split["val"]
    test_idx = split["test"]

    print(f"[04_train_baselines]")
    print(f"  features: {X.shape}")
    print(f"  split: train={len(train_idx)}, "
          f"val={len(val_idx)}, test={len(test_idx)}")

    results = {}

    # ---------- 1. Single features (for reference) ----------
    print(f"\n[04_train_baselines] single-feature baselines")
    feature_names = np.load("data/processed/feature_names.npy", allow_pickle=True)
    single = {}
    for i, name in enumerate(feature_names):
        tau, _ = kendalltau(X[test_idx, i], y_sigma[test_idx])
        single[str(name)] = float(tau)
        print(f"  {name:15s}: tau = {tau:+.4f}")
    results["single_features"] = single

    # ---------- 2. GradientBoosting ----------
    print(f"\n[04_train_baselines] GradientBoosting")
    gb_taus = []
    for seed in [42, 123, 456, 789, 1011]:
        gb = GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            random_state=seed,
        )
        gb.fit(X[train_idx], y_log[train_idx])
        tau, _ = kendalltau(gb.predict(X[test_idx]), y_sigma[test_idx])
        gb_taus.append(tau)
    print(f"  {summarize(gb_taus, 'GradientBoosting')}")
    results["gradient_boosting"] = {
        "test_taus": gb_taus,
        "mean": float(np.mean(gb_taus)),
        "std": float(np.std(gb_taus, ddof=1)),
    }

    # ---------- 3. RandomForest ----------
    print(f"\n[04_train_baselines] RandomForest")
    rf_taus = []
    for seed in [42, 123, 456, 789, 1011]:
        rf = RandomForestRegressor(
            n_estimators=300, max_depth=8,
            random_state=seed, n_jobs=-1,
        )
        rf.fit(X[train_idx], y_log[train_idx])
        tau, _ = kendalltau(rf.predict(X[test_idx]), y_sigma[test_idx])
        rf_taus.append(tau)
    print(f"  {summarize(rf_taus, 'RandomForest')}")
    results["random_forest"] = {
        "test_taus": rf_taus,
        "mean": float(np.mean(rf_taus)),
        "std": float(np.std(rf_taus, ddof=1)),
    }

    # ---------- 4. MLP ----------
    print(f"\n[04_train_baselines] MLP")
    X_t = torch.from_numpy(X)
    train_mask_np = np.zeros(len(X), dtype=bool)
    train_mask_np[train_idx] = True
    train_mask_t = torch.from_numpy(train_mask_np)

    mlp_taus = []
    mlp_val_taus = []
    for seed in [42, 123, 456, 789, 1011]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MLPRanker(in_dim=X.shape[1])
        val_tau, test_tau, _ = train_model(
            model=model,
            forward_fn=lambda m: m(X_t),
            y_log=y_log, y_sigma=y_sigma,
            train_mask=train_mask_t,
            val_idx=val_idx, test_idx=test_idx,
            n_valid=len(X),
            epochs=300, lr=5e-3, weight_decay=1e-4,
            patience=50, eval_every=5,
        )
        mlp_val_taus.append(val_tau)
        mlp_taus.append(test_tau)
    print(f"  {summarize(mlp_taus, 'MLP')}")
    results["mlp"] = {
        "test_taus": mlp_taus,
        "val_taus": mlp_val_taus,
        "mean": float(np.mean(mlp_taus)),
        "std": float(np.std(mlp_taus, ddof=1)),
    }

    # ---------- Save ----------
    out_path = results_dir / "baselines.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[04_train_baselines] saved to {out_path}")

    # ---------- Summary table ----------
    print(f"\n[04_train_baselines] SUMMARY")
    print(f"  {'Model':<20s} {'test tau':>12s}")
    print(f"  {'-'*32}")
    best_single = max(single, key=single.get)
    print(f"  {'best single feat':<20s} {single[best_single]:>12.4f}  ({best_single})")
    print(f"  {'GradientBoosting':<20s} {results['gradient_boosting']['mean']:>12.4f}"
          f"  +/- {results['gradient_boosting']['std']:.4f}")
    print(f"  {'RandomForest':<20s} {results['random_forest']['mean']:>12.4f}"
          f"  +/- {results['random_forest']['std']:.4f}")
    print(f"  {'MLP':<20s} {results['mlp']['mean']:>12.4f}"
          f"  +/- {results['mlp']['std']:.4f}")


if __name__ == "__main__":
    main()
