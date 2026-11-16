"""Numba-accelerated temporal SIR for large-scale Monte Carlo estimation.

The logic mirrors simulator.simulate_sir_time_respecting but operates on
NumPy arrays for JIT compilation. The first call triggers compilation
(~15-30 s); subsequent calls are 100-500x faster than the Python version.

A statistical equivalence check (KS test on per-node sigma) is run in
tests/test_simulator.py to ensure the Numba and Python versions produce
the same distribution.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _geom(mu):
    """Sample Geometric(mu) with support {1, 2, ...}. Large int if mu=0."""
    if mu <= 0:
        return np.int32(2**30)
    if mu >= 1:
        return np.int32(1)
    r = np.random.random()
    return np.int32(np.ceil(np.log(1.0 - r) / np.log(1.0 - mu)))


@njit(cache=True)
def simulate_sir_numba(u_arr, v_arr, seed, beta, mu, n_nodes, rng_seed):
    """
    Numba-accelerated temporal SIR.

    Parameters
    ----------
    u_arr : np.ndarray[int32], sorted by timestamp (ties broken by u, v)
        Source nodes of temporal edges.
    v_arr : np.ndarray[int32], same ordering as u_arr
        Target nodes.
    seed : int
        Initially infected node.
    beta : float
        Infection probability per contact.
    mu : float
        Recovery probability per event step.
    n_nodes : int
    rng_seed : int
        Seed for the Numba RNG (note: uses np.random, not Python's random).

    Returns
    -------
    int
        Number of nodes ever infected (including seed).
    """
    np.random.seed(rng_seed)
    state = np.zeros(n_nodes, dtype=np.int8)      # 0 = S, 1 = I, 2 = R
    lifetime = np.zeros(n_nodes, dtype=np.int32)
    state[seed] = 1
    lifetime[seed] = _geom(mu)
    infected = 1

    n = len(u_arr)
    for i in range(n):
        u = u_arr[i]
        if state[u] == 1:
            v = v_arr[i]
            if state[v] == 0:
                if np.random.random() < beta:
                    state[v] = 1
                    infected += 1
                    lifetime[v] = _geom(mu)
            lifetime[u] -= 1
            if lifetime[u] <= 0:
                state[u] = 2
    return infected


@njit(cache=True)
def simulate_sir_numba_multi_seed(u_arr, v_arr, seed_arr, n_seeds,
                                   beta, mu, n_nodes, rng_seed):
    """
    Numba-accelerated temporal SIR starting from a set of seeds.

    Used for spread-ratio evaluation. seed_arr must be a contiguous
    int32 array; n_seeds is its length.
    """
    np.random.seed(rng_seed)
    state = np.zeros(n_nodes, dtype=np.int8)
    lifetime = np.zeros(n_nodes, dtype=np.int32)
    infected = 0
    for k in range(n_seeds):
        s = seed_arr[k]
        if state[s] == 0:
            state[s] = 1
            lifetime[s] = _geom(mu)
            infected += 1

    n = len(u_arr)
    for i in range(n):
        u = u_arr[i]
        if state[u] == 1:
            v = v_arr[i]
            if state[v] == 0:
                if np.random.random() < beta:
                    state[v] = 1
                    infected += 1
                    lifetime[v] = _geom(mu)
            lifetime[u] -= 1
            if lifetime[u] <= 0:
                state[u] = 2
    return infected
