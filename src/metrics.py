"""Evaluation metrics: Kendall's tau, precision@k, spread ratio, CI."""
import numpy as np
from scipy.stats import kendalltau, wilcoxon


def kendall_tau_b(pred: np.ndarray, truth: np.ndarray) -> float:
    """Kendall's tau_b between predicted scores and ground-truth sigma."""
    tau, _ = kendalltau(pred, truth)
    return float(tau)


def precision_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    """
    Overlap between the top-k predicted set and the top-k ground-truth set.
    """
    pred_top = set(np.argsort(-np.asarray(pred))[:k])
    true_top = set(np.argsort(-np.asarray(truth))[:k])
    return len(pred_top & true_top) / k


def spread_ratio(
    pred: np.ndarray,
    true_sigma: np.ndarray,
    k: int,
) -> float:
    """
    Approximate spread ratio: sum of ground-truth sigma of the top-k
    predicted nodes, divided by the sum of sigma of the top-k ground-truth
    nodes. A value of 1.0 means the predicted top-k captures the same
    total spreading power as the true top-k.

    For a full Monte Carlo spread ratio (simulating the diffusion from the
    seed set), see scripts/06_interpretability.py which uses
    simulate_sir_numba_multi_seed.
    """
    pred_top = np.argsort(-np.asarray(pred))[:k]
    true_top = np.argsort(-np.asarray(true_sigma))[:k]
    return float(true_sigma[pred_top].sum() / true_sigma[true_top].sum())


def bootstrap_ci(
    values,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple:
    """
    Bootstrap confidence interval on the mean.

    Returns (ci_low, ci_high) at level (1 - alpha).
    """
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_resamples)]
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def wilcoxon_test(
    a,
    b,
    alternative: str = "two-sided",
) -> dict:
    """
    Wilcoxon signed-rank test between paired samples a and b.

    Returns dict with statistic and p-value. If all differences are zero,
    returns p=1.0 (no evidence of difference).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a - b
    if np.allclose(diff, 0):
        return {"statistic": 0.0, "pvalue": 1.0}
    try:
        stat, p = wilcoxon(a, b, alternative=alternative)
        return {"statistic": float(stat), "pvalue": float(p)}
    except ValueError:
        return {"statistic": float("nan"), "pvalue": float("nan")}


def holm_bonferroni(
    pvalues: list,
    alpha: float = 0.05,
) -> list:
    """
    Holm-Bonferroni correction for multiple comparisons.

    Returns list of booleans (True = significant after correction).
    """
    m = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    significant = [False] * m
    for rank, idx in enumerate(sorted_idx):
        threshold = alpha / (m - rank)
        if pvalues[idx] < threshold:
            significant[idx] = True
        else:
            break
    return significant


def summarize(
    taus: list,
    label: str = "",
) -> str:
    """Format a list of tau values as 'mean ± std [CI]'."""
    taus = np.asarray(taus)
    mean = taus.mean()
    std = taus.std(ddof=1) if len(taus) > 1 else 0.0
    if len(taus) > 1:
        lo, hi = bootstrap_ci(taus)
        ci_str = f", 95% CI [{lo:.4f}, {hi:.4f}]"
    else:
        ci_str = ""
    prefix = f"{label}: " if label else ""
    return f"{prefix}tau = {mean:.4f} +/- {std:.4f}{ci_str}"
