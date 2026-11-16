"""Tests for the temporal SIR and IC simulators.

The hand-crafted 5-node graph and its expected outcomes are documented
below. These tests validate both the Python and Numba implementations:

    Edges:
        t=1:  A -> B
        t=2:  B -> C
        t=3:  A -> C
        t=4:  C -> D
        t=5:  B -> D

    Node 4 (E) is isolated.

    Expected SIR results (event-based recovery):
        seed=A, beta=1, mu=0:  {A,B,C,D}       = 4
        seed=B, beta=1, mu=0:  {B,C,D}         = 3
        seed=A, beta=0, mu=0:  {A}             = 1
        seed=A, beta=1, mu=1:  {A,B,C,D}       = 4

    The mu=1 case is subtle: each node transmits exactly once (on its
    first outgoing edge) and then recovers. A transmits on t=1 (to B),
    B on t=2 (to C), C on t=4 (to D). A's edge at t=3 is skipped because
    A has recovered; B's edge at t=5 is skipped because B has recovered.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import numpy as np
import pytest

from src.simulator import (
    geometric,
    simulate_sir_time_respecting,
    simulate_ic_time_respecting,
    simulate_sir_multi_seed,
)


TEST_EVENTS = [
    (0, 1, 1),
    (1, 2, 2),
    (0, 2, 3),
    (2, 3, 4),
    (1, 3, 5),
]


# ---------- Geometric sampler ----------

def test_geometric_mu_zero_returns_large():
    """mu=0 -> never recovers -> very large lifetime."""
    rng = random.Random(42)
    assert geometric(0.0, rng) >= 2**30 - 1


def test_geometric_mu_one_returns_one():
    """mu=1 -> recovers after one step."""
    rng = random.Random(42)
    assert geometric(1.0, rng) == 1


def test_geometric_mean():
    """Sample mean of Geometric(0.1) should be close to 10."""
    rng = random.Random(42)
    samples = [geometric(0.1, rng) for _ in range(10_000)]
    assert 9.0 < np.mean(samples) < 11.0


# ---------- SIR deterministic cases ----------

def test_sir_seed_a_full_transmission():
    r = simulate_sir_time_respecting(
        TEST_EVENTS, 0, beta=1.0, mu=0.0, n_nodes=5, rng=random.Random(42)
    )
    assert r == 4, f"expected 4, got {r}"


def test_sir_seed_b_full_transmission():
    r = simulate_sir_time_respecting(
        TEST_EVENTS, 1, beta=1.0, mu=0.0, n_nodes=5, rng=random.Random(42)
    )
    assert r == 3, f"expected 3, got {r}"


def test_sir_zero_beta():
    r = simulate_sir_time_respecting(
        TEST_EVENTS, 0, beta=0.0, mu=0.0, n_nodes=5, rng=random.Random(42)
    )
    assert r == 1, f"expected 1, got {r}"


def test_sir_mu_one():
    """mu=1 -> each node transmits once on its first outgoing edge."""
    r = simulate_sir_time_respecting(
        TEST_EVENTS, 0, beta=1.0, mu=1.0, n_nodes=5, rng=random.Random(42)
    )
    assert r == 4, f"expected 4, got {r}"


def test_sir_deterministic_with_same_seed():
    r1 = simulate_sir_time_respecting(
        TEST_EVENTS, 0, beta=0.5, mu=0.5, n_nodes=5, rng=random.Random(42)
    )
    r2 = simulate_sir_time_respecting(
        TEST_EVENTS, 0, beta=0.5, mu=0.5, n_nodes=5, rng=random.Random(42)
    )
    assert r1 == r2


# ---------- Multi-seed SIR ----------

def test_sir_multi_seed_two_seeds():
    """Seed set {A, B}, beta=1, mu=0 -> still 4 nodes (A,B,C,D)."""
    r = simulate_sir_multi_seed(
        TEST_EVENTS, {0, 1}, beta=1.0, mu=0.0, n_nodes=5, rng=random.Random(42)
    )
    assert r == 4


# ---------- Independent Cascade ----------

def test_ic_seed_a_full():
    r = simulate_ic_time_respecting(
        TEST_EVENTS, 0, p=1.0, n_nodes=5, rng=random.Random(42)
    )
    assert r == 4


def test_ic_seed_b_full():
    r = simulate_ic_time_respecting(
        TEST_EVENTS, 1, p=1.0, n_nodes=5, rng=random.Random(42)
    )
    assert r == 3


def test_ic_zero_p():
    r = simulate_ic_time_respecting(
        TEST_EVENTS, 0, p=0.0, n_nodes=5, rng=random.Random(42)
    )
    assert r == 1


# ---------- Numba equivalence ----------

def test_numba_matches_python_deterministic():
    """On a 5-node graph, Numba and Python should give the same
    deterministic result when beta=1 and mu=0."""
    try:
        from src.simulator_numba import simulate_sir_numba
    except ImportError:
        pytest.skip("numba not installed")

    events = np.array(TEST_EVENTS, dtype=np.int64)
    events = events[np.argsort(events[:, 2], kind="stable")]
    u_arr = events[:, 0].astype(np.int32)
    v_arr = events[:, 1].astype(np.int32)

    # Warm up JIT
    _ = simulate_sir_numba(u_arr, v_arr, 0, 1.0, 0.0, 5, 42)

    # beta=1, mu=0 -> deterministic
    for seed in range(4):
        r_py = simulate_sir_time_respecting(
            TEST_EVENTS, seed, beta=1.0, mu=0.0, n_nodes=5,
            rng=random.Random(42),
        )
        r_nb = simulate_sir_numba(u_arr, v_arr, seed, 1.0, 0.0, 5, 42)
        assert r_py == r_nb, (
            f"seed={seed}: python={r_py}, numba={r_nb}"
        )


def test_numba_python_distribution_close():
    """On a small graph with beta<1, the two implementations should
    produce statistically indistinguishable distributions (KS test)."""
    try:
        from src.simulator_numba import simulate_sir_numba
        from scipy.stats import ks_2samp
    except ImportError:
        pytest.skip("numba or scipy not installed")

    events = np.array(TEST_EVENTS, dtype=np.int64)
    events = events[np.argsort(events[:, 2], kind="stable")]
    u_arr = events[:, 0].astype(np.int32)
    v_arr = events[:, 1].astype(np.int32)

    _ = simulate_sir_numba(u_arr, v_arr, 0, 0.5, 0.3, 5, 42)

    n_runs = 2000
    py = np.array([
        simulate_sir_time_respecting(
            TEST_EVENTS, 0, 0.5, 0.3, 5, rng=random.Random(i)
        )
        for i in range(n_runs)
    ])
    nb = np.array([
        simulate_sir_numba(u_arr, v_arr, 0, 0.5, 0.3, 5, 1000 + i)
        for i in range(n_runs)
    ])

    stat, pval = ks_2samp(py, nb)
    assert pval > 0.01, (
        f"KS test rejected at p={pval:.4f} (stat={stat:.4f}); "
        f"python mean={py.mean():.2f}, numba mean={nb.mean():.2f}"
    )
