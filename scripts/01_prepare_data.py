"""Step 1: Download and prepare a SNAP temporal network.

Loads the dataset, relabels nodes to 0..n-1, sorts edges by timestamp,
filters out nodes with zero out-degree, and saves the processed arrays
to data/processed/.

Outputs
-------
data/processed/events.npy       : (m, 3) int array [u, v, t]
data/processed/valid_nodes.npy  : (n_valid,) int array of node indices
data/processed/n_nodes.txt      : total node count (including filtered)
data/processed/stats.json       : summary statistics
"""
import sys
import json
from pathlib import Path

# Make `src` importable when running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.data import load_events, filter_valid_nodes
from src.features import split_indices


def main(dataset_name: str = "CollegeMsg"):
    data_dir = Path("data")
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    print(f"[01_prepare_data] dataset = {dataset_name}")
    events, n_nodes = load_events(dataset_name, data_dir)
    print(f"  raw nodes: {n_nodes}")
    print(f"  temporal edges: {len(events)}")

    # Aggregated edge count
    agg_edges = set()
    for u, v, _ in events:
        agg_edges.add((int(u), int(v)))
    print(f"  aggregated edges: {len(agg_edges)}")

    # Time span
    t_min = int(events[:, 2].min())
    t_max = int(events[:, 2].max())
    span_days = (t_max - t_min) / 86400
    print(f"  time span: {span_days:.1f} days")

    # Filter
    valid_nodes = filter_valid_nodes(events, n_nodes)
    n_filtered = n_nodes - len(valid_nodes)
    pct = 100 * n_filtered / n_nodes
    print(f"  valid nodes (out_deg > 0): {len(valid_nodes)}")
    print(f"  filtered out: {n_filtered} ({pct:.1f}%)")

    # Save
    np.save(processed / "events.npy", events)
    np.save(processed / "valid_nodes.npy", valid_nodes)
    (processed / "n_nodes.txt").write_text(str(n_nodes))

    # Split indices (fixed seed for reproducibility across all scripts)
    split = split_indices(len(valid_nodes), seed=42)
    np.savez(
        processed / "split.npz",
        train=split["train"],
        val=split["val"],
        test=split["test"],
    )
    print(f"  split: train={len(split['train'])}, "
          f"val={len(split['val'])}, test={len(split['test'])}")

    # Stats
    stats = {
        "dataset": dataset_name,
        "n_nodes": int(n_nodes),
        "n_valid_nodes": int(len(valid_nodes)),
        "n_temporal_edges": int(len(events)),
        "n_aggregated_edges": int(len(agg_edges)),
        "filtered_nodes": int(n_filtered),
        "filtered_pct": float(pct),
        "time_span_days": float(span_days),
        "split_sizes": {
            "train": int(len(split["train"])),
            "val": int(len(split["val"])),
            "test": int(len(split["test"])),
        },
    }
    (processed / "stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    print(f"\n[01_prepare_data] saved to {processed}")
    for f in ["events.npy", "valid_nodes.npy", "n_nodes.txt",
              "split.npz", "stats.json"]:
        p = processed / f
        print(f"  {f}: {p.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CollegeMsg",
                        help="SNAP dataset name (default: CollegeMsg)")
    args = parser.parse_args()
    main(args.dataset)
