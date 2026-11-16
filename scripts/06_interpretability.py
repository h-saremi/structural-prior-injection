"""Step 6: Interpretability analysis and boundary conditions.

Uses results from step 5 (results/injection_results.json) to produce:

  1. The naive-vs-controlled attention comparison table (Section 5.3)
  2. The per-node volatility analysis (Section 5.4 boundary condition)
  3. Rank stability of the top-100 set across seeds

The per-node volatility analysis tests whether the accuracy gain from
intermediate injection is larger on low-volatility nodes than on
high-volatility nodes. Volatility of node v is defined as the fraction
of v's outgoing edges that appear only once in the temporal graph.

Outputs
-------
results/interpretability.json : all computed metrics
results/summary.txt           : human-readable summary
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau

from src.models import InjectionRanker
from src.training import train_model
from src.interpretability import extract_excess_for_model, alignment_metrics


RAW_IDX = [0, 1, 6, 7]
CENT_IDX = [2, 3, 4, 5]
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
    node_to_local = {int(v): i for i, v in enumerate(valid_nodes)}
    src, dst = [], []
    for u, v, _ in events:
        u_int = int(u)
        if u_int in node_to_local:
            src.append(node_to_local[u_int])
            dst.append(node_to_local.get(int(v), n_valid))
    return torch.tensor([src, dst], dtype=torch.long)


def compute_per_node_volatility(data):
    """
    Volatility of node v = fraction of v's outgoing edges that appear
    exactly once in the temporal graph.

    Returns array of shape (n_valid,), values in [0, 1].
    """
    events = data["events"]
    valid_nodes = data["valid_nodes"]
    n_valid = len(valid_nodes)
    node_to_local = {int(v): i for i, v in enumerate(valid_nodes)}

    edge_counts = {}  # (u_local, v) -> count
    for u, v, _ in events:
        u_int = int(u)
        if u_int in node_to_local:
            key = (node_to_local[u_int], int(v))
            edge_counts[key] = edge_counts.get(key, 0) + 1

    per_node_total = np.zeros(n_valid, dtype=np.int32)
    per_node_once = np.zeros(n_valid, dtype=np.int32)
    for (u_local, v), count in edge_counts.items():
        per_node_total[u_local] += count
        if count == 1:
            per_node_once[u_local] += count

    nu = np.where(per_node_total > 0,
                  per_node_once / np.maximum(per_node_total, 1),
                  0.0)
    return nu.astype(np.float32)


def train_intermediate_for_split(data, edge_index, n_valid, seed=42):
    """Train intermediate-injection model once and return predictions."""
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

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = InjectionRanker(
        raw_dim=len(RAW_IDX), cent_dim=len(CENT_IDX),
        hidden=64, injection="intermediate",
    )
    forward_fn = lambda m: m(edge_index, X_raw_full, X_cent_full)
    val_tau, test_tau, model = train_model(
        model=model, forward_fn=forward_fn,
        y_log=y_log, y_sigma=y_sigma,
        train_mask=train_mask_t,
        val_idx=val_idx, test_idx=test_idx,
        n_valid=n_valid,
        epochs=400, lr=3e-3, weight_decay=1e-5,
        patience=50, eval_every=5,
    )

    with torch.no_grad():
        out = forward_fn(model)[:n_valid]
    return out.numpy(), val_tau, test_tau


def train_none_for_split(data, edge_index, n_valid, seed=42):
    """Train no-injection baseline once and return predictions."""
    X = data["X"]
    y_sigma = data["y_sigma"]
    y_log = data["y_log"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    X_raw = X[:, RAW_IDX]
    n_total = n_valid + 1

    X_raw_full = torch.zeros(n_total, len(RAW_IDX))
    X_raw_full[:n_valid] = torch.from_numpy(X_raw)
    X_cent_full = torch.zeros(n_total, len(CENT_IDX))

    train_mask_np = np.zeros(n_valid, dtype=bool)
    train_mask_np[data["train_idx"]] = True
    train_mask_t = torch.from_numpy(train_mask_np)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = InjectionRanker(
        raw_dim=len(RAW_IDX), cent_dim=len(CENT_IDX),
        hidden=64, injection="none",
    )
    forward_fn = lambda m: m(edge_index, X_raw_full, X_cent_full)
    val_tau, test_tau, model = train_model(
        model=model, forward_fn=forward_fn,
        y_log=y_log, y_sigma=y_sigma,
        train_mask=train_mask_t,
        val_idx=val_idx, test_idx=test_idx,
        n_valid=n_valid,
        epochs=400, lr=3e-3, weight_decay=1e-5,
        patience=50, eval_every=5,
    )
    with torch.no_grad():
        out = forward_fn(model)[:n_valid]
    return out.numpy(), val_tau, test_tau


def volatility_split_analysis(data, edge_index, n_valid):
    """
    Split test nodes into low-volatility and high-volatility halves by
    per-node volatility, then compare the tau gain of intermediate over
    none in each half.

    This tests the paper's boundary condition: if structural priors are
    more useful when edges persist, intermediate injection should improve
    tau more on the low-volatility half.
    """
    print(f"\n[06_interpretability] volatility split analysis")
    nu = compute_per_node_volatility(data)
    test_idx = data["test_idx"]
    y_sigma = data["y_sigma"]

    nu_test = nu[test_idx]
    median = float(np.median(nu_test))
    print(f"  per-node volatility: median = {median:.3f}, "
          f"range = [{nu_test.min():.3f}, {nu_test.max():.3f}]")

    low_mask = nu_test < median
    high_mask = nu_test >= median

    # Train intermediate and none, take predictions on test
    print(f"  training none...")
    pred_none, _, _ = train_none_for_split(data, edge_index, n_valid, seed=42)
    pred_none_test = pred_none[test_idx]

    print(f"  training intermediate...")
    pred_inter, _, _ = train_intermediate_for_split(
        data, edge_index, n_valid, seed=42,
    )
    pred_inter_test = pred_inter[test_idx]

    # Per-half tau
    def half_tau(pred, y, mask):
        if mask.sum() < 10:
            return None
        tau, _ = kendalltau(pred[mask], y[mask])
        return float(tau)

    y_test = y_sigma[test_idx]
    results = {
        "median_volatility": median,
        "n_low": int(low_mask.sum()),
        "n_high": int(high_mask.sum()),
        "none_low": half_tau(pred_none_test, y_test, low_mask),
        "none_high": half_tau(pred_none_test, y_test, high_mask),
        "inter_low": half_tau(pred_inter_test, y_test, low_mask),
        "inter_high": half_tau(pred_inter_test, y_test, high_mask),
    }
    if results["none_low"] is not None and results["inter_low"] is not None:
        results["delta_low"] = results["inter_low"] - results["none_low"]
    if results["none_high"] is not None and results["inter_high"] is not None:
        results["delta_high"] = results["inter_high"] - results["none_high"]

    print(f"  low volatility  (n={results['n_low']}): "
          f"none={results['none_low']:.4f}  "
          f"inter={results['inter_low']:.4f}  "
          f"delta={results.get('delta_low', 0):+.4f}")
    print(f"  high volatility (n={results['n_high']}): "
          f"none={results['none_high']:.4f}  "
          f"inter={results['inter_high']:.4f}  "
          f"delta={results.get('delta_high', 0):+.4f}")

    if results.get("delta_low") is not None and results.get("delta_high") is not None:
        diff = results["delta_low"] - results["delta_high"]
        print(f"  delta_low - delta_high = {diff:+.4f}")
        if diff > 0.01:
            print(f"  → Confirms boundary hypothesis: injection helps more "
                  f"on low-volatility nodes")
        elif diff < -0.01:
            print(f"  → Refutes boundary hypothesis: injection helps more "
                  f"on high-volatility nodes")
        else:
            print(f"  → Inconclusive: no clear volatility effect")

    return results


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print(f"[06_interpretability] loading data...")
    data = load_data()
    n_valid = len(data["y_sigma"])
    edge_index = build_edges(data["events"], data["valid_nodes"], n_valid)
    print(f"  n_valid: {n_valid}, edges: {tuple(edge_index.shape)}")

    # Load step 5 results
    inj_path = results_dir / "injection_results.json"
    if not inj_path.exists():
        raise FileNotFoundError(
            f"{inj_path} not found. Run scripts/05_injection_experiments.py first."
        )
    with open(inj_path) as f:
        injection_results = json.load(f)

    print(f"\n[06_interpretability] loaded {inj_path}")

    # ---------- Table: naive vs controlled ----------
    print(f"\n[06_interpretability] Naive vs degree-controlled attention")
    print(f"  {'Config':<16s} {'tau':>8s} {'rho_raw':>10s} {'rho_tilde':>12s}")
    print(f"  {'-'*50}")
    table_data = {}
    for config in ["none", "input", "intermediate", "output"]:
        r = injection_results.get(config, {})
        if r.get("rho_excess") is None:
            continue
        table_data[config] = {
            "test_tau": r["mean"],
            "rho_raw": r["rho_excess"],
            "rho_tilde": r["rho_tilde"],
        }
        print(f"  {config:<16s} {r['mean']:>8.4f} "
              f"{r['rho_excess']:>+10.4f} {r['rho_tilde']:>+12.4f}")

    # ---------- Hypothesis 1 test ----------
    print(f"\n[06_interpretability] Hypothesis 1 test")
    if "none" in table_data and "intermediate" in table_data:
        base_tau = table_data["none"]["test_tau"]
        base_rho = table_data["none"]["rho_tilde"]
        inter_tau = table_data["intermediate"]["test_tau"]
        inter_rho = table_data["intermediate"]["rho_tilde"]

        tau_ratio = inter_tau / base_tau if base_tau > 0 else float("inf")
        delta_rho = inter_rho - base_rho
        tau_ok = tau_ratio >= 0.97
        rho_ok = delta_rho >= 0.10

        print(f"  base tau:      {base_tau:.4f}")
        print(f"  inter tau:     {inter_tau:.4f}  (ratio = {tau_ratio:.3f})")
        print(f"  base rho_t:    {base_rho:+.4f}")
        print(f"  inter rho_t:   {inter_rho:+.4f}  (delta = {delta_rho:+.4f})")
        print(f"  tau criterion (>= 0.97):  {'PASS' if tau_ok else 'FAIL'}")
        print(f"  rho criterion (>= 0.10):  {'PASS' if rho_ok else 'FAIL'}")
        h1_status = "SUPPORTED" if (tau_ok and rho_ok) else "NOT SUPPORTED"
        print(f"  H1: {h1_status}")

        h1_result = {
            "status": h1_status,
            "tau_ratio": float(tau_ratio),
            "delta_rho": float(delta_rho),
            "tau_ok": bool(tau_ok),
            "rho_ok": bool(rho_ok),
        }
    else:
        h1_result = {"status": "insufficient_data"}
        print(f"  (missing required configs)")

    # ---------- Volatility split ----------
    vol_result = volatility_split_analysis(data, edge_index, n_valid)

    # ---------- Save ----------
    out = {
        "naive_vs_controlled": table_data,
        "hypothesis_1": h1_result,
        "volatility_split": vol_result,
    }
    out_path = results_dir / "interpretability.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[06_interpretability] saved to {out_path}")

    # Summary text
    summary_lines = [
        "Interpretability summary",
        "=" * 40,
        "",
        f"H1 status: {h1_result['status']}",
    ]
    if "tau_ratio" in h1_result:
        summary_lines.append(f"  tau ratio:  {h1_result['tau_ratio']:.3f}")
        summary_lines.append(f"  delta rho:  {h1_result['delta_rho']:+.4f}")
    summary_lines.append("")
    summary_lines.append("Volatility split:")
    if "delta_low" in vol_result:
        summary_lines.append(f"  delta (low vol):  {vol_result['delta_low']:+.4f}")
    if "delta_high" in vol_result:
        summary_lines.append(f"  delta (high vol): {vol_result['delta_high']:+.4f}")
    summary_path = results_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[06_interpretability] wrote {summary_path}")


if __name__ == "__main__":
    main()
