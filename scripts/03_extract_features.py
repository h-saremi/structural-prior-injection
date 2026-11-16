"""Step 3: Extract and standardize the 8 static features.

Computes out_deg, in_deg, betweenness, closeness, eigenvector, k_shell,
event_count, and activity_span for every valid node, then applies
log1p to the heavy-tailed features and standardizes all columns.

Outputs
-------
data/processed/X_raw.npy       : (n_valid, 8) raw features
data/processed/X_static.npy    : (n_valid, 8) standardized features
data/processed/feature_names.npy : (8,) feature names
data/processed/feature_stats.json : per-feature summary
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.features import extract_features, standardize, FEATURE_NAMES


def main():
    processed = Path("data/processed")
    if not (processed / "events.npy").exists():
        raise FileNotFoundError(
            "data/processed/events.npy not found. Run scripts/01_prepare_data.py first."
        )

    events = np.load(processed / "events.npy")
    valid_nodes = np.load(processed / "valid_nodes.npy")
    n_nodes = int((processed / "n_nodes.txt").read_text().strip())

    print(f"[03_extract_features]")
    print(f"  events: {len(events)}, valid nodes: {len(valid_nodes)}")
    print(f"  computing 8 centrality features...")

    X_raw, names = extract_features(events, n_nodes, valid_nodes)
    print(f"  raw feature matrix: {X_raw.shape}")

    X_std, scaler = standardize(X_raw, names)
    print(f"  standardized: shape={X_std.shape}, "
          f"range=[{X_std.min():.2f}, {X_std.max():.2f}]")

    # Save
    np.save(processed / "X_raw.npy", X_raw)
    np.save(processed / "X_static.npy", X_std)
    np.save(processed / "feature_names.npy", np.array(names))

    # Per-feature summary
    stats = {}
    for i, name in enumerate(names):
        stats[name] = {
            "raw_mean": float(X_raw[:, i].mean()),
            "raw_std": float(X_raw[:, i].std()),
            "raw_min": float(X_raw[:, i].min()),
            "raw_max": float(X_raw[:, i].max()),
            "std_mean": float(X_std[:, i].mean()),
            "std_std": float(X_std[:, i].std()),
        }
    (processed / "feature_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    print(f"\n[03_extract_features] saved to {processed}")
    for f in ["X_raw.npy", "X_static.npy", "feature_names.npy", "feature_stats.json"]:
        p = processed / f
        print(f"  {f}: {p.stat().st_size / 1024:.1f} KB")

    # Quick sanity: single-feature tau
    print(f"\n[03_extract_features] single-feature Kendall tau vs sigma")
    sigma_path = processed / "sigma_sir.npy"
    if sigma_path.exists():
        from scipy.stats import kendalltau
        sigma = np.load(sigma_path)[valid_nodes]
        for i, name in enumerate(names):
            tau, _ = kendalltau(X_std[:, i], sigma)
            print(f"  {name:15s}: tau = {tau:+.4f}")
    else:
        print("  (skipped - sigma_sir.npy not yet computed)")


if __name__ == "__main__":
    main()
