"""Degree-controlled excess-attention analysis.

Implements Definition 1 of the paper:

    beta_v = sum_l sum_{(v,u) in E} alpha_vu^(l)(T)
           - sum_l sum_{(v,u) in E} 1 / d_in(u)

The first term is the actual attention mass that v sends over its outgoing
edges; the second is the expected mass under uniform allocation. A node
has positive excess attention when the model concentrates attention on
its outgoing edges beyond what uniform allocation would predict.

The naive alternative---mean attention per outgoing edge---is mechanically
anti-correlated with out-degree (high-degree nodes spread attention over
many edges) and produces a spurious negative correlation with centrality.
The excess-attention metric removes this confound by construction.
"""
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression


def get_attention_weights(
    conv,
    h_in: torch.Tensor,
    edge_index: torch.Tensor,
) -> tuple:
    """
    Extract per-edge attention weights from a TransformerConv layer.

    Parameters
    ----------
    conv : torch_geometric.nn.TransformerConv
    h_in : torch.Tensor, shape (n_nodes, hidden)
    edge_index : torch.Tensor, shape (2, E)

    Returns
    -------
    edge_index : torch.Tensor, shape (2, E)
    alpha : torch.Tensor, shape (E,)
        Mean attention weight across heads, detached from the graph.
    """
    _, (ei, alpha) = conv(h_in, edge_index, return_attention_weights=True)
    if alpha.dim() > 1:
        alpha = alpha.mean(dim=-1)
    return ei, alpha.detach()


def excess_attention(
    conv1,
    conv2,
    h1_in: torch.Tensor,
    h2_in: torch.Tensor,
    edge_index: torch.Tensor,
    n_total: int,
) -> np.ndarray:
    """
    Compute excess attention per node (Definition 1 of the paper).

    Parameters
    ----------
    conv1, conv2 : TransformerConv layers
    h1_in : input to conv1
    h2_in : input to conv2
    edge_index : torch.Tensor, shape (2, E)
    n_total : int
        Number of nodes including the sink index.

    Returns
    -------
    excess : np.ndarray, shape (n_total,)
    """
    with torch.no_grad():
        ei1, a1 = get_attention_weights(conv1, h1_in, edge_index)
        ei2, a2 = get_attention_weights(conv2, h2_in, edge_index)

    actual = torch.zeros(n_total)
    expected = torch.zeros(n_total)

    for ei, alpha in [(ei1, a1), (ei2, a2)]:
        src = ei[0].cpu()
        dst = ei[1].cpu()
        actual.index_add_(0, src, alpha.cpu())

        # Expected attention under uniform allocation is 1 / in_deg(dst)
        in_deg = torch.zeros(n_total)
        in_deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        in_deg = in_deg.clamp(min=1)
        expected.index_add_(0, src, 1.0 / in_deg[dst])

    excess = (actual - expected).numpy()
    return excess


def alignment_metrics(
    excess: np.ndarray,
    ref_cent: np.ndarray,
    degree_control: np.ndarray,
) -> tuple:
    """
    Compute raw and degree-controlled Spearman correlations.

    Parameters
    ----------
    excess : np.ndarray, shape (n,)
        Excess attention per node (Definition 1).
    ref_cent : np.ndarray, shape (n,)
        Reference centrality scores (e.g. eigenvector centrality).
    degree_control : np.ndarray, shape (n,)
        Degree proxy used as the control variable. In the paper we use
        log(1 + out_degree). Residualizing against this removes the
        degree confound that mechanically correlates attention with
        centrality.

    Returns
    -------
    rho : float
        Raw Spearman correlation between excess attention and centrality.
    rho_tilde : float
        Degree-controlled Spearman correlation (on residuals).
    """
    rho, _ = spearmanr(excess, ref_cent)

    dc = degree_control.reshape(-1, 1)
    r_excess = excess - LinearRegression().fit(dc, excess).predict(dc)
    r_cent = ref_cent - LinearRegression().fit(dc, ref_cent).predict(dc)
    rho_tilde, _ = spearmanr(r_excess, r_cent)

    return float(rho), float(rho_tilde)


def extract_excess_for_model(
    model,
    edge_index: torch.Tensor,
    x_raw_full: torch.Tensor,
    x_cent_full: torch.Tensor,
    n_valid: int,
) -> np.ndarray:
    """
    Convenience wrapper: run the model up to layer 1, extract excess
    attention for the valid nodes only.

    Parameters
    ----------
    model : InjectionRanker
    edge_index, x_raw_full, x_cent_full : see models.InjectionRanker
    n_valid : int

    Returns
    -------
    excess : np.ndarray, shape (n_valid,)
    """
    n_total = x_raw_full.shape[0]
    model.eval()

    with torch.no_grad():
        if model.injection == "input":
            x_in = torch.cat([x_raw_full, x_cent_full], dim=-1)
        else:
            x_in = x_raw_full
        h1_in = model.raw_proj(x_in)
        h1_mid = h1_in + model.ln1(model.conv1(h1_in, edge_index))
        if model.injection == "intermediate":
            h1_mid = h1_mid + model.cent_proj_mid(x_cent_full)

    excess = excess_attention(
        model.conv1, model.conv2, h1_in, h1_mid, edge_index, n_total
    )
    return excess[:n_valid]
