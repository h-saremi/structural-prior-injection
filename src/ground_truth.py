"""Monte Carlo computation of ground-truth spreading power.

Wraps simulator_numba.simulate_sir_numba with checkpointing and progress
reporting for long runs. Produces sigma(v) for every valid node.
"""
import time
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from .simulator_numba import simulate_sir_numba


def compute_ground_truth(
    events: np.ndarray,
    valid_nodes: np.ndarray,
    n_nodes: int,
    beta: float = 0.2,
    mu: float = 0.1,
    n_runs: int = 1000,
    base_seed: int = 42,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 50,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute sigma(v) for every valid node via Monte Carlo.

    Parameters
    ----------
    events : np.ndarray, shape (m, 3)
        Temporal edges, columns [u, v, t]. Will be re-sorted internally.
    valid_nodes : np.ndarray
        Node indices to evaluate (typically out-degree > 0).
    n_nodes : int
        Total number of nodes.
    beta, mu : float
        SIR parameters.
    n_runs : int
        Monte Carlo runs per node.
    base_seed : int
        Base RNG seed. Per-run seed = base_seed + node * n_runs + run.
    checkpoint_path : Path, optional
        If provided, saves progress every checkpoint_every nodes and
        resumes from the last saved state on restart.
    checkpoint_every : int
    verbose : bool

    Returns
    -------
    sigma : np.ndarray, shape (n_nodes,), dtype float32
        sigma[v] = mean number of activated nodes when v is the sole seed.
        Entries for non-valid nodes are 0.
    """
    # Sort events stably by timestamp
    order = np.argsort(events[:, 2], kind="stable")
    events = events[order]
    u_arr = events[:, 0].astype(np.int32)
    v_arr = events[:, 1].astype(np.int32)

    # Warm up JIT compilation (first call is slow)
    _ = simulate_sir_numba(u_arr, v_arr, 0, beta, mu, n_nodes, base_seed)

    sigma = np.zeros(n_nodes, dtype=np.float32)
    done = set()

    if checkpoint_path is not None and checkpoint_path.exists():
        sigma = np.load(checkpoint_path)
        done = set(np.where(sigma > 0)[0].tolist())
        if verbose:
            print(f"Resuming from {checkpoint_path}: {len(done)} nodes already computed")

    t0 = time.time()
    iterator = tqdm(valid_nodes, desc="MC", disable=not verbose)
    for i, node in enumerate(iterator):
        node = int(node)
        if node in done:
            continue
        total = 0
        for r in range(n_runs):
            total += simulate_sir_numba(
                u_arr, v_arr, node, beta, mu, n_nodes,
                base_seed + node * n_runs + r,
            )
        sigma[node] = total / n_runs

        if checkpoint_path is not None and (i + 1) % checkpoint_every == 0:
            np.save(checkpoint_path, sigma)
            if verbose:
                elapsed = time.time() - t0
                rate = (i + 1 - len(done)) / elapsed if elapsed > 0 else 0
                eta = (len(valid_nodes) - i - 1) / rate / 60 if rate > 0 else 0
                print(f"  [{i+1}/{len(valid_nodes)}] "
                      f"{rate:.2f} nodes/s, ETA {eta:.1f} min")

    if checkpoint_path is not None:
        np.save(checkpoint_path, sigma)

    if verbose:
        print(f"Done in {(time.time() - t0) / 60:.1f} min")
        print(f"  mean = {sigma[valid_nodes].mean():.2f}, "
              f"std = {sigma[valid_nodes].std():.2f}, "
              f"range = [{sigma[valid_nodes].min():.2f}, "
              f"{sigma[valid_nodes].max():.2f}]")

    return sigma


def compute_ground_truth_ic(
    events: np.ndarray,
    valid_nodes: np.ndarray,
    n_nodes: int,
    p: float = 0.2,
    n_runs: int = 1000,
    base_seed: int = 42,
) -> np.ndarray:
    """
    Independent-cascade ground truth. Uses the Python simulator because
    the IC model is fast enough without JIT on CollegeMsg-scale data.
    """
    from .simulator import simulate_ic_time_respecting

    sigma = np.zeros(n_nodes, dtype=np.float32)
    import random as _random
    for node in valid_nodes:
        node = int(node)
        rng = _random.Random(base_seed + node)
        total = 0
        for _ in range(n_runs):
            total += simulate_ic_time_respecting(events, node, p, n_nodes, rng)
        sigma[node] = total / n_runs
    return sigma
