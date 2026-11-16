"""Step 2: Compute ground-truth spreading power via Monte Carlo.

Runs the temporal SIR process (event-based recovery) with the parameters
fixed in the paper (beta=0.2, mu=0.1, I=1,000) and saves sigma(v) for
every valid node.

Checkpointing is enabled by default: if the job is interrupted, re-running
the script resumes from the last saved state. This is important on Colab
where sessions are terminated after ~90 min of idle time.

Outputs
-------
data/processed/sigma_sir.npy : (n_nodes,) float32, zeros for invalid nodes
data/processed/gt_stats.json : summary of the distribution
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.ground_truth import compute_ground_truth


def main(
    beta: float = 0.2,
    mu: float = 0.1,
    n_runs: int = 1000,
    base_seed: int = 42,
    checkpoint: bool = True,
):
    processed = Path("data/processed")
    if not (processed / "events.npy").exists():
        raise FileNotFoundError(
            "data/processed/events.npy not found. Run scripts/01_prepare_data.py first."
        )

    events = np.load(processed / "events.npy")
    valid_nodes = np.load(processed / "valid_nodes.npy")
    n_nodes = int((processed / "n_nodes.txt").read_text().strip())

    print(f"[02_compute_ground_truth]")
    print(f"  events: {len(events)}, valid nodes: {len(valid_nodes)}")
    print(f"  beta={beta}, mu={mu}, I={n_runs}, base_seed={base_seed}")
    print(f"  checkpointing: {checkpoint}")

    ckpt_path = processed / "sigma_sir.npy" if checkpoint else None

    sigma = compute_ground_truth(
        events=events,
        valid_nodes=valid_nodes,
        n_nodes=n_nodes,
        beta=beta,
        mu=mu,
        n_runs=n_runs,
        base_seed=base_seed,
        checkpoint_path=ckpt_path,
        checkpoint_every=50,
        verbose=True,
    )

    # Distribution summary over valid nodes
    s_valid = sigma[valid_nodes]
    stats = {
        "beta": beta,
        "mu": mu,
        "n_runs": n_runs,
        "base_seed": base_seed,
        "n_valid_nodes": int(len(valid_nodes)),
        "mean": float(s_valid.mean()),
        "std": float(s_valid.std()),
        "min": float(s_valid.min()),
        "median": float(np.median(s_valid)),
        "max": float(s_valid.max()),
        "p10": float(np.percentile(s_valid, 10)),
        "p90": float(np.percentile(s_valid, 90)),
        "n_sigma_eq_one": int((s_valid == 1).sum()),
    }
    (processed / "gt_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    print(f"\n[02_compute_ground_truth] saved sigma to {processed / 'sigma_sir.npy'}")
    print(f"  distribution: mean={stats['mean']:.2f}, "
          f"std={stats['std']:.2f}, "
          f"range=[{stats['min']:.2f}, {stats['max']:.2f}]")
    print(f"  nodes with sigma=1: {stats['n_sigma_eq_one']}")

    # Correlation with out-degree (for the paper's Section 4.2 sanity check)
    out_deg = np.zeros(n_nodes, dtype=np.int32)
    for u in events[:, 0]:
        out_deg[u] += 1
    from scipy.stats import spearmanr
    rho, p = spearmanr(out_deg[valid_nodes], s_valid)
    print(f"  Spearman rho(sigma, out_degree) = {rho:.4f} (p={p:.2e})")
    print(f"  rho^2 = {rho**2:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--n-runs", type=int, default=1000)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()
    main(
        beta=args.beta,
        mu=args.mu,
        n_runs=args.n_runs,
        base_seed=args.base_seed,
        checkpoint=not args.no_checkpoint,
    )
