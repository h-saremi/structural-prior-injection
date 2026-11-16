"""Training loop with proper validation/test alignment.

Key design decision
-------------------
Validation and test predictions are indexed with `val_idx` / `test_idx`
(integer arrays) rather than with boolean masks. Indexing with a mask
returns predictions in ascending node-position order, while `y[val_idx]`
returns labels in the shuffled permutation order. The two are not aligned
and Kendall's tau computed on them is meaningless.

This alignment bug is easy to introduce and produces near-zero tau
despite a correctly-trained model. We use integer indexing everywhere
to avoid it.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau


def train_model(
    model: torch.nn.Module,
    forward_fn,
    y_log: np.ndarray,
    y_sigma: np.ndarray,
    train_mask: torch.Tensor,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    n_valid: int,
    device: str = "cpu",
    epochs: int = 400,
    lr: float = 3e-3,
    weight_decay: float = 1e-5,
    patience: int = 50,
    grad_clip: float = 1.0,
    eval_every: int = 5,
    verbose: bool = False,
) -> tuple:
    """
    Train a model with MSE-on-log(sigma) and early stopping on val tau.

    Parameters
    ----------
    model : torch.nn.Module
    forward_fn : callable
        Maps model -> scores tensor of shape (n_total,). Usually a lambda
        that closes over edge_index and the feature tensors.
    y_log : np.ndarray, shape (n_valid,)
        log(sigma) targets.
    y_sigma : np.ndarray, shape (n_valid,)
        Raw sigma targets, used for tau on the test split.
    train_mask : torch.Tensor, bool, shape (n_valid,)
    val_idx, test_idx : np.ndarray
        Integer index arrays into the (n_valid,) ordering.
    n_valid : int
    device : str
    epochs, lr, weight_decay, patience, grad_clip : training hyperparams
    eval_every : int
        Evaluate validation tau every `eval_every` epochs.
    verbose : bool

    Returns
    -------
    best_val_tau : float
    best_test_tau : float
    best_model : torch.nn.Module
        Model with the best-validation weights loaded.
    """
    model = model.to(device)
    y_log_t = torch.from_numpy(y_log).to(device)
    train_mask_t = train_mask.to(device) if isinstance(train_mask, torch.Tensor) else torch.from_numpy(train_mask).to(device)
    val_idx_t = torch.from_numpy(val_idx)
    test_idx_t = torch.from_numpy(test_idx)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_tau = -1.0
    best_test_tau = -1.0
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = forward_fn(model)[:n_valid]
        loss = F.mse_loss(pred[train_mask_t], y_log_t[train_mask_t])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if (epoch + 1) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                out = forward_fn(model)[:n_valid]
                # INTEGER indexing, not mask indexing
                pred_val = out[val_idx_t].cpu().numpy()
            tau_val, _ = kendalltau(pred_val, y_log[val_idx])

            if tau_val > best_val_tau:
                best_val_tau = tau_val
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += eval_every

            if verbose and (epoch + 1) % 25 == 0:
                print(f"  epoch {epoch+1:3d}  loss={loss.item():.4f}  "
                      f"val_tau={tau_val:.4f}  best={best_val_tau:.4f}@{best_epoch}")

            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch+1}")
                break

    # Load best and compute test tau
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = forward_fn(model)[:n_valid]
        pred_test = out[test_idx_t].cpu().numpy()
    best_test_tau, _ = kendalltau(pred_test, y_sigma[test_idx])

    return float(best_val_tau), float(best_test_tau), model


def train_with_seeds(
    model_factory,
    forward_fn_factory,
    y_log: np.ndarray,
    y_sigma: np.ndarray,
    train_mask: torch.Tensor,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    n_valid: int,
    seeds=(42, 123, 456, 789, 1011),
    **train_kwargs,
) -> dict:
    """
    Run training across multiple seeds and aggregate.

    Parameters
    ----------
    model_factory : callable
        Returns a freshly initialized model. Called once per seed.
    forward_fn_factory : callable
        Given a model, returns the forward function.
    seeds : tuple of int
    **train_kwargs
        Passed to train_model.

    Returns
    -------
    dict with keys:
        test_taus : list of float (one per seed)
        val_taus  : list of float
        mean      : float
        std       : float
        ci_low, ci_high : 95% bootstrap CI on the mean
    """
    test_taus = []
    val_taus = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = model_factory()
        val_tau, test_tau, _ = train_model(
            model,
            forward_fn_factory(model),
            y_log, y_sigma, train_mask, val_idx, test_idx, n_valid,
            **train_kwargs,
        )
        val_taus.append(val_tau)
        test_taus.append(test_tau)

    taus = np.array(test_taus)
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(taus, size=len(taus), replace=True).mean()
                  for _ in range(10_000)]

    return {
        "test_taus": test_taus,
        "val_taus": val_taus,
        "mean": float(taus.mean()),
        "std": float(taus.std(ddof=1)) if len(taus) > 1 else 0.0,
        "ci_low": float(np.quantile(boot_means, 0.025)),
        "ci_high": float(np.quantile(boot_means, 0.975)),
    }
