"""Step 5: Run the four injection configurations and the TGN reference.

Trains five models on CollegeMsg, each with 5 random seeds:
  - TGN-8: temporal GNN with all 8 static features at the input (reference)
  - none:        4 raw features only, no injection (baseline)
  - input:       4 raw + 4 centrality concatenated at input
  - intermediate: 4 raw at input, centrality added after conv1
  - output:      4 raw at input, centrality as linear bias on score

Computes Kendall's tau and excess-attention metrics for each, saves to
results/injection_results.json.

The critical implementation detail: validation and test predictions are
indexed with `val_idx` / `test_idx` (integer arrays), not boolean masks.
See src/training.py docstring for why.

Expected results (CollegeMsg, from the paper):
  TGN-8:         0.675
  none:          0.656
  input:         0.677
  intermediate:  0.696
  output:        0.672

Excess attention (degree-controlled, tilde_rho):
  none:          -0.253
  input:         -0.146
  intermediate:  +0.064
  output:        +0.169
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.models import InjectionRanker, MLPRanker
from src.training import train_model
from src.interpretability import extract_excess_for_model, alignment_metrics


RAW_IDX = [0, 1, 6, 7]      # out_deg, in_deg, event_count, activity_span
CENT_IDX = [2, 3, 4, 5]     # betweenness, closeness, eigenvector, k_shell
SEEDS = [42, 123, 456, 789, 1011]


def load_data():
    processed = Path("data/processed")
    X = np.load(processed / "X_static.npy")
    X_raw = np.load(processed / "X_raw.npy")
    sigma_full = np.load(processed / "sigma_sir.npy")
    valid_nodes = np.load(processed / "valid_nodes.npy")
    events = np.load(processed / "events.npy")
    n_nodes = int((processed / "n_nodes.txt").read_text().strip())
    split = np.load(processed / "split.npz")

    y_sigma = sigma_full[valid_nodes].astype(np.float32)
    y_log = np.log(y_sigma).astype(np.float32)

    return {
        "X": X, "X_raw": X_raw, "y_sigma": y_sigma, "y_log": y_log,
        "valid_nodes": valid_nodes, "events": events, "n_nodes": n_nodes,
        "train_idx": split["train"],
        "val_idx": split["val"],
        "test_idx": split["test"],
    }


def build_edges(events, valid_nodes, n_valid):
    """Build edge_index restricted to valid-node sources. Invalid targets
    map to a sink node at index n_valid."""
    node_to_local = {int(v): i for i, v in enumerate(valid_nodes)}
    src, dst = [], []
    for u, v, _ in events:
        u_int = int(u)
        if u_int in node_to_local:
            src.append(node_to_local[u_int])
            dst.append(node_to_local.get(int(v), n_valid))
    return torch.tensor([src, dst], dtype=torch.long)


def run_one_config(config_name, data, edge_index, n_valid, seeds, verbose=False):
    """
    Train a single injection configuration across seeds.

    Returns dict with test_taus, val_taus, mean, std, and interpretability
    metrics (rho, rho_tilde) averaged over seeds.
    """
    X = data["X"]
    y_sigma = data["y_sigma"]
    y_log = data["y_log"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    X_raw = X[:, RAW_IDX]
    X_cent = X[:, CENT_IDX]
    n_total = n_valid + 1

    X_raw_full = torch.zeros(n_total, len(RAW_IDX))
    X_raw_full[:n_valid] = torch.from_numpy(X_raw)
    X_cent_full = torch.zeros(n_total, len(CENT_IDX))
    X_cent_full[:n_valid] = torch.from_numpy(X_cent)

    train_mask_np = np.zeros(n_valid, dtype=bool)
    train_mask_np[data["train_idx"]] = True
    train_mask_t = torch.from_numpy(train_mask_np)

    test_taus = []
    val_taus = []
    rho_list = []
    rho_tilde_list = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # TGN-8 reference: 8 features at input, no injection
        if config_name == "tgn8":
            # Temporal GNN with all 8 features in the raw path
            model = InjectionRanker(
                raw_dim=X.shape[1], cent_dim=0,
                hidden=64, injection="none",
            )
            X_all_full = torch.zeros(n_total, X.shape[1])
            X_all_full[:n_valid] = torch.from_numpy(X)
            forward_fn = lambda m: m(edge_index, X_all_full, torch.zeros(n_total, 0))
            val_tau, test_tau, _ = train_model(
                model=model, forward_fn=forward_fn,
                y_log=y_log, y_sigma=y_sigma,
                train_mask=train_mask_t,
                val_idx=val_idx, test_idx=test_idx,
                n_valid=n_valid,
                epochs=400, lr=3e-3, weight_decay=1e-5,
                patience=50, eval_every=5, verbose=False,
            )
            val_taus.append(val_tau)
            test_taus.append(test_tau)
            continue

        # Injection variants
        model = InjectionRanker(
            raw_dim=len(RAW_IDX), cent_dim=len(CENT_IDX),
            hidden=64, injection=config_name,
        )
        forward_fn = lambda m: m(edge_index, X_raw_full, X_cent_full)

        val_tau, test_tau, model = train_model(
            model=model, forward_fn=forward_fn,
            y_log=y_log, y_sigma=y_sigma,
            train_mask=train_mask_t,
            val_idx=val_idx, test_idx=test_idx,
            n_valid=n_valid,
            epochs=400, lr=3e-3, weight_decay=1e-5,
            patience=50, eval_every=5, verbose=False,
        )
        val_taus.append(val_tau)
        test_taus.append(test_tau)

        # Excess attention
        excess = extract_excess_for_model(
            model, edge_index, X_raw_full, X_cent_full, n_valid,
        )
        # degree_control: log1p of RAW (unstandardized) out-degree
        raw_features = data["X_raw"]  # shape (n_valid, 8), raw values
        degree_control = np.log1p(raw_features[:, 0])
        # Eigenvector centrality as the reference for alignment
        ref_cent = X_cent[:, 2] if CENT_IDX[2] == 4 else X[:, 4]
        rho, rho_t = alignment_metrics(excess, ref_cent, degree_control)
        rho_list.append(rho)
        rho_tilde_list.append(rho_t)

    result = {
        "config": config_name,
        "test_taus": [float(t) for t in test_taus],
        "val_taus": [float(t) for t in val_taus],
        "mean": float(np.mean(test_taus)),
        "std": float(np.std(test_taus, ddof=1)) if len(test_taus) > 1 else 0.0,
        "rho_excess": float(np.mean(rho_list)) if rho_list else None,
        "rho_tilde": float(np.mean(rho_tilde_list)) if rho_tilde_list else None,
    }
    return result


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print(f"[05_injection_experiments] loading data...")
    data = load_data()
    n_valid = len(data["y_sigma"])
    print(f"  n_valid: {n_valid}")

    print(f"[05_injection_experiments] building edge_index...")
    edge_index = build_edges(data["events"], data["valid_nodes"], n_valid)
    print(f"  edge_index: {tuple(edge_index.shape)}")

    all_results = {}
    configs = ["tgn8", "none", "input", "intermediate", "output"]

    for config in configs:
        print(f"\n[05_injection_experiments] config = {config}")
        t0 = time.time()
        result = run_one_config(config, data, edge_index, n_valid, SEEDS)
        elapsed = time.time() - t0
        all_results[config] = result
        print(f"  test tau = {result['mean']:.4f} +/- {result['std']:.4f}"
              f"  ({elapsed:.1f}s)")
        if result["rho_tilde"] is not None:
            print(f"  rho_excess = {result['rho_excess']:+.4f}, "
                  f"rho_tilde = {result['rho_tilde']:+.4f}")

    # Save
    out_path = results_dir / "injection_results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n[05_injection_experiments] saved to {out_path}")

    # Summary
    print(f"\n[05_injection_experiments] SUMMARY")
    print(f"  {'Config':<16s} {'test tau':>20s} {'rho_tilde':>12s}")
    print(f"  {'-'*50}")
    for config in configs:
        r = all_results[config]
        rt = f"{r['rho_tilde']:+.4f}" if r['rho_tilde'] is not None else "---"
        print(f"  {config:<16s} {r['mean']:.4f} +/- {r['std']:.4f}   {rt}")


if __name__ == "__main__":
    main()
