"""
Reliability diagrams for multiclass classifiers.

A reliability diagram (Niculescu-Mizil & Caruana, 2005; Guo et al., 2017)
visualises calibration by binning predictions by their confidence and
plotting the empirical accuracy in each bin against the bin's mean confidence.

A perfectly calibrated model lies on the diagonal y=x.

We follow the multiclass formulation:
    - For each sample, take the predicted class (argmax) and its confidence
      (max softmax probability).
    - Bin samples by their confidence.
    - In each bin, compute the empirical accuracy of the argmax prediction.

This is the "top-label calibration" formulation (Guo et al. 2017), which is
the most common in the deep-learning literature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReliabilityCurve:
    """Output of reliability_curve()."""
    bin_centres: np.ndarray         # (n_bins,)
    bin_accuracies: np.ndarray      # (n_bins,) — empirical acc per bin (NaN if empty)
    bin_confidences: np.ndarray     # (n_bins,) — mean confidence per bin (NaN if empty)
    bin_counts: np.ndarray          # (n_bins,) — number of samples in each bin
    ece: float                      # overall ECE
    n_samples: int


def reliability_curve(
    y_true: np.ndarray,         # (N,) int labels
    y_prob: np.ndarray,          # (N, K) softmax probs
    n_bins: int = 10,
) -> ReliabilityCurve:
    """Compute a reliability curve from soft predictions.

    Args:
        y_true: integer labels in [0, K).
        y_prob: probabilities summing to 1 per row.
        n_bins: number of equal-width bins on the [0, 1] confidence axis.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_prob.ndim != 2:
        raise ValueError(f"y_prob must be (N, K); got shape {y_prob.shape}")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob length mismatch")

    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    accs = np.full(n_bins, np.nan)
    confs = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    ece_total = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include right edge in the last bin (covers conf == 1.0)
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        counts[i] = int(mask.sum())
        if counts[i] == 0:
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(confidences[mask].mean())
        accs[i] = bin_acc
        confs[i] = bin_conf
        ece_total += (counts[i] / n) * abs(bin_acc - bin_conf)

    return ReliabilityCurve(
        bin_centres=bin_centres,
        bin_accuracies=accs,
        bin_confidences=confs,
        bin_counts=counts,
        ece=float(ece_total),
        n_samples=n,
    )


def plot_reliability(
    curves: dict[str, ReliabilityCurve],
    title: str = "Reliability diagram",
    figsize: tuple[float, float] = (8, 7),
    ax=None,
):
    """Plot multiple reliability curves overlaid for visual comparison.

    Args:
        curves: dict of {model_name: ReliabilityCurve}.
        title: plot title.
        figsize: matplotlib figure size.
        ax: optional axes to draw on (if None, create one).

    Returns:
        The axes object.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    # Diagonal
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="perfect calibration")
    for name, c in curves.items():
        # Skip NaN bins so the line isn't broken
        valid = ~np.isnan(c.bin_accuracies)
        ax.plot(
            c.bin_confidences[valid], c.bin_accuracies[valid],
            "o-", label=f"{name} (ECE={c.ece:.4f}, n={c.n_samples})",
        )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence (mean per bin)")
    ax.set_ylabel("Empirical accuracy (per bin)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Temperature scaling (Platt-style post-hoc calibration)
# ---------------------------------------------------------------------------

def temperature_scale(
    logits: np.ndarray,    # (N, K) raw logits (pre-softmax)
    y_true: np.ndarray,    # (N,) int labels
    n_iter: int = 50,
    lr: float = 0.01,
) -> tuple[float, np.ndarray]:
    """Fit a single scalar T such that softmax(logits / T) minimises NLL.

    From Guo et al. (2017) "On Calibration of Modern Neural Networks".
    Returns:
        (T*, calibrated_probs)
    """
    import torch
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(y_true, dtype=torch.long)
    T = torch.tensor(1.0, requires_grad=True)
    optimizer = torch.optim.LBFGS([T], lr=lr, max_iter=n_iter)

    def closure():
        optimizer.zero_grad()
        scaled = logits_t / T.clamp(min=1e-4)
        loss = torch.nn.functional.cross_entropy(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    T_value = float(T.detach().item())
    calibrated = torch.softmax(logits_t / max(T_value, 1e-4), dim=-1).numpy()
    return T_value, calibrated
