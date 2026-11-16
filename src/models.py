"""Baseline and injection models.

Contains:
  - MLPRanker: 2-layer MLP over static features (shallow learned baseline)
  - InjectionRanker: temporal GNN with optional centrality prior injection
    at input, intermediate, or output level
"""
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class MLPRanker(nn.Module):
    """2-layer MLP ranker over static features."""

    def __init__(self, in_dim: int, hidden: int = 48, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class InjectionRanker(nn.Module):
    """
    Temporal GNN with optional centrality-prior injection.

    Parameters
    ----------
    raw_dim : int
        Number of raw activity features (out_deg, in_deg, event_count,
        activity_span). Always fed to the model.
    cent_dim : int
        Number of centrality features (betweenness, closeness, eigenvector,
        k_shell). Used only when injection != 'none'.
    hidden : int
        Hidden dimension for the temporal attention blocks.
    injection : {'none', 'input', 'intermediate', 'output'}
        Where to inject the centrality prior:
          - 'none':          baseline, centrality not used
          - 'input':         concatenated to raw features before proj
          - 'intermediate':  added to hidden representation after conv1
          - 'output':        added as linear bias to the final score

    Architecture
    ------------
    Both input and intermediate variants use a 2-layer residual temporal
    attention backbone:

        h_static = raw_proj(x_in)
        h        = h_static + LN(Conv1(h_static))      [ + cent_proj if intermediate ]
        h        = h        + LN(Conv2(h))              [ last layer ]
        score    = head([h_static || h])

    The concatenation [h_static || h] at the head preserves the static
    signal through message passing, which we found essential: without it,
    the attention blocks dilute the activity features and accuracy drops
    below the MLP baseline.
    """

    def __init__(
        self,
        raw_dim: int,
        cent_dim: int,
        hidden: int = 64,
        injection: str = "none",
        dropout: float = 0.1,
    ):
        super().__init__()
        if injection not in ("none", "input", "intermediate", "output"):
            raise ValueError(f"Unknown injection level: {injection}")
        self.injection = injection

        in_dim = raw_dim + (cent_dim if injection == "input" else 0)
        self.raw_proj = nn.Linear(in_dim, hidden)

        self.cent_proj_mid = (
            nn.Linear(cent_dim, hidden) if injection == "intermediate" else None
        )
        self.cent_proj_out = (
            nn.Linear(cent_dim, 1) if injection == "output" else None
        )

        self.conv1 = TransformerConv(hidden, hidden, heads=2, concat=False)
        self.conv2 = TransformerConv(hidden, hidden, heads=2, concat=False)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, edge_index, x_raw, x_cent):
        """
        Parameters
        ----------
        edge_index : torch.Tensor, shape (2, E)
            Temporal edges, cols [source, target]. Sources must be valid
            nodes (0..n_valid-1); targets can include a sink index n_valid.
        x_raw : torch.Tensor, shape (n_total, raw_dim)
        x_cent : torch.Tensor, shape (n_total, cent_dim)

        Returns
        -------
        scores : torch.Tensor, shape (n_total,)
            Ranking scores. Only the first n_valid entries are used.
        """
        if self.injection == "input":
            x_in = torch.cat([x_raw, x_cent], dim=-1)
        else:
            x_in = x_raw

        h_static = self.raw_proj(x_in)

        # First residual block
        h = h_static + self.ln1(self.conv1(h_static, edge_index))

        # Intermediate injection: after first attention block
        if self.injection == "intermediate":
            h = h + self.cent_proj_mid(x_cent)

        # Second residual block
        h = h + self.ln2(self.conv2(h, edge_index))

        # Ranking head with residual static signal
        h = torch.cat([h_static, h], dim=-1)
        score = self.head(h).squeeze(-1)

        # Output injection: additive linear bias
        if self.injection == "output":
            score = score + self.cent_proj_out(x_cent).squeeze(-1)

        return score
