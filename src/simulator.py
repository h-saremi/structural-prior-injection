"""Temporal SIR and IC simulation on time-respecting graphs."""
import math
import random
from typing import Optional


def geometric(p: float, rng: random.Random) -> int:
    """
    Sample from Geometric(p) with support {1, 2, 3, ...}. Mean = 1/p.

    Returns a large integer when p = 0 (never recovers).
    """
    if p <= 0:
        return 2**30
    if p >= 1:
        return 1
    return int(math.ceil(math.log(1 - rng.random()) / math.log(1 - p)))


def simulate_sir_time_respecting(
    events,
    seed_node: int,
    beta: float,
    mu: float,
    n_nodes: int,
    rng: Optional[random.Random] = None,
) -> int:
    """
    Event-based temporal SIR with per-node geometric recovery.

    Each temporal edge contributes one step to the lifetime of its source.
    A node recovers after Geometric(mu) of its OWN outgoing events, so the
    recovery rate is comparable across datasets with very different
    inter-event time distributions.

    Transmission is only possible through the chronologically next edge
    after the source became infected: if (u, v, t) occurs before u's
    infection time, it is not used by u.

    Parameters
    ----------
    events : iterable of (source, target, timestamp)
    seed_node : int
        Initially infected node.
    beta : float
        Infection probability per contact (0 to 1).
    mu : float
        Recovery probability per event step (0 = never recovers).
    n_nodes : int
        Total number of nodes.
    rng : random.Random, optional
        Random generator for reproducibility.

    Returns
    -------
    int
        Number of nodes ever infected (including the seed).

    Notes
    -----
    The algorithm iterates over the sorted event list once. When an edge
    (u, v, t) is processed, if u is currently infectious, it attempts to
    infect v (if v is susceptible) with probability beta, then its lifetime
    counter decreases by one. When the counter reaches zero, u recovers and
    no longer transmits on later edges.

    This event-driven loop is equivalent to a priority-queue implementation
    but simpler and faster for a single seed. It is also the version
    validated against the hand-crafted 5-node test in tests/test_simulator.py.
    """
    if rng is None:
        rng = random.Random()

    events = sorted(events, key=lambda e: (e[2], e[0], e[1]))

    state = {seed_node: "I"}
    lifetime = {seed_node: geometric(mu, rng)}

    for u, v, _ in events:
        if state.get(u) != "I":
            continue
        # Attempt transmission
        if state.get(v) not in ("I", "R"):
            if rng.random() < beta:
                state[v] = "I"
                lifetime[v] = geometric(mu, rng)
        # Decrement source lifetime
        lifetime[u] -= 1
        if lifetime[u] <= 0:
            state[u] = "R"

    return sum(1 for s in state.values() if s in ("I", "R"))


def simulate_ic_time_respecting(
    events,
    seed_node: int,
    p: float,
    n_nodes: int,
    rng: Optional[random.Random] = None,
    same_time_transmission: bool = False,
) -> int:
    """
    Temporal Independent Cascade.

    Each temporal edge (u, v, t) is attempted exactly once, in chronological
    order, succeeding with probability p. A node activated at time t cannot
    transmit on its own outgoing edges at the same time t unless
    same_time_transmission=True.

    Parameters
    ----------
    events : iterable of (source, target, timestamp)
    seed_node : int
    p : float
        Infection probability per edge (0 to 1).
    n_nodes : int
    rng : random.Random, optional
    same_time_transmission : bool, default False

    Returns
    -------
    int
        Number of nodes activated (including the seed).
    """
    if rng is None:
        rng = random.Random()

    events_sorted = sorted(events, key=lambda e: (e[2], e[0], e[1]))
    active = {seed_node: 0}  # node -> activation time

    for u, v, t in events_sorted:
        if u not in active:
            continue
        if not same_time_transmission and active[u] >= t:
            continue
        if v in active:
            continue
        if rng.random() < p:
            active[v] = t

    return len(active)


def simulate_sir_multi_seed(
    events,
    seed_set,
    beta: float,
    mu: float,
    n_nodes: int,
    rng: Optional[random.Random] = None,
) -> int:
    """
    Temporal SIR starting from a SET of seed nodes.

    Used for spread-ratio evaluation: given the top-k ranked nodes,
    simulate the diffusion and measure how many nodes are reached.
    """
    if rng is None:
        rng = random.Random()

    events = sorted(events, key=lambda e: (e[2], e[0], e[1]))
    state = {}
    lifetime = {}
    for s in seed_set:
        state[s] = "I"
        lifetime[s] = geometric(mu, rng)

    for u, v, _ in events:
        if state.get(u) != "I":
            continue
        if state.get(v) not in ("I", "R"):
            if rng.random() < beta:
                state[v] = "I"
                lifetime[v] = geometric(mu, rng)
        lifetime[u] -= 1
        if lifetime[u] <= 0:
            state[u] = "R"

    return sum(1 for s in state.values() if s in ("I", "R"))
