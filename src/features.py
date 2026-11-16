"""Static centrality and temporal feature extraction.

Computes the 8 features used throughout the paper:
  out_deg, in_deg, betweenness, closeness, eigenvector, k_shell,
  event_count, activity_span

and applies log1p + standardization to the heavy-tailed ones.
"""
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "out_deg", "in_deg", "betweenness", "closeness",
    "eigenvector", "k_shell", "event_count", "activity_span",
]

# Indices of features that receive log1p before standardization
LOG_FEATURES = {
    "out_deg", "in_deg", "betweenness", "eigenvector",
    "event_count", "activity_span",
}


def build_aggregated_graph(events: np.ndarray, n_nodes: int) -> nx.DiGraph:
    """Build the aggregated directed graph from temporal edges."""
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    for u, v, _ in events:
        G.add_edge(int(u), int(v))
    return G


def extract_features(
    events: np.ndarray,
    n_nodes: int,
    valid_nodes: np.ndarray,
) -> tuple:
    """
    Compute the 8 static features for every valid node.

    Parameters
    ----------
    events : np.ndarray, shape (m, 3)
        Temporal edges [u, v, t].
    n_nodes : int
    valid_nodes : np.ndarray
        Node indices to compute features for.

    Returns
    -------
    X : np.ndarray, shape (n_valid, 8), dtype float32
        Raw (un-standardized) features.
    names : list of str
        Column names, in the same order as X.
    """
    G = build_aggregated_graph(events, n_nodes)
    G_undirected = G.to_undirected()

    out_deg = dict(G.out_degree())
    in_deg = dict(G.in_degree())

    # Betweenness: use k-sampling for speed on large graphs
    k_sample = min(100, n_nodes)
    betweenness = nx.betweenness_centrality(
        G_undirected, k=k_sample, seed=42
    )

    closeness = nx.closeness_centrality(G_undirected)

    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {v: 0.0 for v in G.nodes()}

    core_number = nx.core_number(G_undirected)

    # Temporal features
    event_count = {}
    first_seen = {}
    last_seen = {}
    for u, _, t in events:
        u = int(u)
        t = int(t)
        event_count[u] = event_count.get(u, 0) + 1
        if u not in first_seen:
            first_seen[u] = t
        last_seen[u] = t
    activity_span = {u: last_seen[u] - first_seen[u] for u in first_seen}

    X = np.zeros((len(valid_nodes), len(FEATURE_NAMES)), dtype=np.float32)
    for i, v in enumerate(valid_nodes):
        v = int(v)
        X[i, 0] = out_deg.get(v, 0)
        X[i, 1] = in_deg.get(v, 0)
        X[i, 2] = betweenness.get(v, 0.0)
        X[i, 3] = closeness.get(v, 0.0)
        X[i, 4] = eigenvector.get(v, 0.0)
        X[i, 5] = core_number.get(v, 0)
        X[i, 6] = event_count.get(v, 0)
        X[i, 7] = activity_span.get(v, 0)

    return X, FEATURE_NAMES


def standardize(
    X: np.ndarray,
    feature_names: list,
    scaler: StandardScaler = None,
) -> tuple:
    """
    Apply log1p to heavy-tailed features, then standardize all columns.

    Parameters
    ----------
    X : np.ndarray, shape (n, 8)
    feature_names : list of str
    scaler : StandardScaler, optional
        If provided, uses this scaler (for applying the same transform
        to validation or test data). If None, fits a new one.

    Returns
    -------
    X_std : np.ndarray, shape (n, 8)
    scaler : StandardScaler
    """
    X = X.copy()
    for i, name in enumerate(feature_names):
        if name in LOG_FEATURES:
            X[:, i] = np.log1p(X[:, i])

    if scaler is None:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X).astype(np.float32)
    else:
        X_std = scaler.transform(X).astype(np.float32)

    return X_std, scaler


def split_indices(
    n: int,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict:
    """
    Split n indices into train / val / test by fixed permutation.

    The same split is used across all methods and seeds for fair
    comparison. Test fraction = 1 - train_frac - val_frac.
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    return {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }
