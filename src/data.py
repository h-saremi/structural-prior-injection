"""Data loading, download, and preprocessing for SNAP temporal networks."""
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


SNAP_URLS = {
    "CollegeMsg": "https://snap.stanford.edu/data/CollegeMsg.txt.gz",
    "email-Eu": "https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz",
    "wiki-talk": "https://snap.stanford.edu/data/wiki-talk-temporal.txt.gz",
    "sx-mathoverflow": "https://snap.stanford.edu/data/sx-mathoverflow-a2q.txt.gz",
}


def download_dataset(name: str, data_dir: Path) -> Path:
    """Download a SNAP temporal network if not already present."""
    if name not in SNAP_URLS:
        raise ValueError(f"Unknown dataset: {name}. "
                         f"Available: {list(SNAP_URLS.keys())}")
    url = SNAP_URLS[name]
    dest = Path(data_dir) / "raw" / f"{name}.txt.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size < 1000:
        print(f"Downloading {name} from {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            dest.write_bytes(r.read())
        print(f"  Saved {dest.stat().st_size / 1e6:.1f} MB to {dest}")
    else:
        print(f"  Using cached {dest}")
    return dest


def load_events(name: str, data_dir: Path) -> Tuple[np.ndarray, int]:
    """
    Load a SNAP temporal network and relabel nodes to 0..n-1.

    Parameters
    ----------
    name : str
        Dataset name (key of SNAP_URLS).
    data_dir : Path
        Root data directory.

    Returns
    -------
    events : np.ndarray, shape (m, 3), dtypes [int32, int32, int64]
        Sorted by timestamp (stable sort, ties broken by source then target).
    n_nodes : int
        Number of distinct nodes in the raw data.
    """
    path = download_dataset(name, data_dir)

    try:
        df = pd.read_csv(
            path, sep=r"\s+", header=None, names=["u", "v", "t"],
            compression="gzip", engine="python", comment="#",
        )
    except Exception as e:
        # Manual fallback for malformed files
        print(f"pandas parser failed ({e}); trying manual gzip reader")
        import gzip
        rows = []
        with gzip.open(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
        df = pd.DataFrame(rows, columns=["u", "v", "t"])

    # Relabel nodes to contiguous 0..n-1
    nodes = sorted(set(df.u.unique()) | set(df.v.unique()))
    mapping = {n: i for i, n in enumerate(nodes)}
    df["u"] = df.u.map(mapping).astype(np.int32)
    df["v"] = df.v.map(mapping).astype(np.int32)
    df["t"] = df.t.astype(np.int64)

    # Sort by timestamp (stable sort preserves within-timestamp order)
    df = df.sort_values("t", kind="stable")
    events = df[["u", "v", "t"]].values
    return events, len(nodes)


def filter_valid_nodes(events: np.ndarray, n_nodes: int) -> np.ndarray:
    """
    Return indices of nodes with at least one outgoing temporal edge.

    Nodes with zero out-degree cannot spread information by construction
    (sigma(v) = 1) and produce artificial ties in ranking metrics.
    """
    out_deg = np.zeros(n_nodes, dtype=np.int32)
    for u in events[:, 0]:
        out_deg[u] += 1
    return np.where(out_deg > 0)[0]


def build_synthetic_events(
    n_nodes: int = 1000,
    avg_degree: int = 8,
    topology: str = "BA",
    alpha_ev: float = 2.0,
    persistence: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, int]:
    """
    Generate a synthetic temporal graph.

    Parameters
    ----------
    n_nodes : int
    avg_degree : int
    topology : str, 'ER' or 'BA'
    alpha_ev : float
        Power-law exponent for inter-event times (larger = less volatile).
    persistence : float in [0, 1]
        Probability that an edge remains active in the next window.
    seed : int

    Returns
    -------
    events : np.ndarray (m, 3)
    n_nodes : int
    """
    import networkx as nx
    rng = np.random.default_rng(seed)

    if topology == "ER":
        G = nx.erdos_renyi_graph(n_nodes, avg_degree / n_nodes, seed=seed)
    elif topology == "BA":
        G = nx.barabasi_albert_graph(n_nodes, avg_degree // 2, seed=seed)
    else:
        raise ValueError(f"Unknown topology: {topology}")

    # Number of windows and events per window
    n_windows = 50
    events_per_window = max(10, int(G.number_of_edges() * avg_degree / 4))

    events = []
    t_counter = 0
    for w in range(n_windows):
        active_edges = [e for e in G.edges() if rng.random() < persistence or w == 0]
        if not active_edges:
            active_edges = list(G.edges())[:max(1, len(G.edges()) // 10)]
        # Draw inter-event times from power law (Pareto)
        n_draw = min(events_per_window, len(active_edges) * 5)
        edge_indices = rng.integers(0, len(active_edges), size=n_draw)
        inter_times = (rng.pareto(alpha_ev - 1, size=n_draw) + 1).astype(np.int64)
        for idx, dt in zip(edge_indices, inter_times):
            u, v = active_edges[idx]
            t_counter += int(dt)
            events.append((int(u), int(v), t_counter))

    events = np.array(sorted(events, key=lambda e: (e[2], e[0], e[1])),
                      dtype=np.int64)
    # Relabel to int32 for u, v
    out = np.zeros_like(events)
    out[:, 0] = events[:, 0].astype(np.int32)
    out[:, 1] = events[:, 1].astype(np.int32)
    out[:, 2] = events[:, 2]
    return out, n_nodes
