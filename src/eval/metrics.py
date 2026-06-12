"""
Evaluation metrics for probabilistic match prediction.

The metrics here are appropriate for a 3-class classification problem with
soft predictions (probabilities) — NOT just accuracy. The reason is that for
football prediction we care about how well-calibrated the probabilities are,
not just whether the most-likely class matches the outcome.

Implemented:
    - multiclass_log_loss        : Cross-entropy / NLL, the primary metric
    - multiclass_brier_score     : Mean squared error of probability vectors
    - expected_calibration_error : ECE with adaptive or equal-width binning
    - reliability_curve_data     : Per-bin observed-vs-predicted for plotting
    - accuracy                   : Top-1 accuracy (reference only)

References:
    Naeini et al. (2015). "Obtaining Well Calibrated Probabilities Using
        Bayesian Binning." AAAI. — ECE definition.
    Guo et al. (2017). "On Calibration of Modern Neural Networks." ICML.
        — reliability diagrams and temperature scaling.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def _validate_probs(y_prob: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Ensure y_prob is (N, C), sums to 1, no NaNs."""
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.ndim != 2 or y_prob.shape[1] != n_classes:
        raise ValueError(f"y_prob must be (N, {n_classes}), got {y_prob.shape}")
    if np.any(np.isnan(y_prob)):
        raise ValueError("y_prob contains NaN")
    row_sums = y_prob.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError(f"Rows of y_prob must sum to 1; min={row_sums.min()}, max={row_sums.max()}")
    return y_prob


def multiclass_log_loss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Multi-class log loss (negative log-likelihood, cross-entropy).

    L = -1/N * sum_i log p_i[y_i]

    Lower is better. A model that outputs uniform 1/3 probabilities has
    log loss = log(3) ≈ 1.0986. Anything below that is meaningful.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = _validate_probs(y_prob)
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
    n = len(y_true)
    return float(-np.log(y_prob_clipped[np.arange(n), y_true]).mean())


def multiclass_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int = 3,
) -> float:
    """Multi-class Brier score: mean squared error of probability vectors.

    B = 1/N * sum_i ||p_i - onehot(y_i)||^2

    Lower is better. Range: [0, 2] for 3 classes. Uniform predictor scores ~0.67.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = _validate_probs(y_prob, n_classes=n_classes)
    n = len(y_true)
    onehot = np.zeros_like(y_prob)
    onehot[np.arange(n), y_true] = 1.0
    return float(((y_prob - onehot) ** 2).sum(axis=1).mean())


def accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Top-1 accuracy. Reference metric only — calibration matters more."""
    return float((np.asarray(y_true) == np.asarray(y_prob).argmax(axis=1)).mean())


# ---------------------------------------------------------------------------
# Calibration: ECE and reliability diagram
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Multi-class ECE based on top-1 confidence (Guo et al. 2017).

    For each prediction, take the max probability (confidence) and the
    predicted class. Bin by confidence, then compare bin-averaged confidence
    against bin-averaged accuracy.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Lower is better. 0 = perfect calibration.

    strategy : 'uniform' (equal-width bins) or 'quantile' (equal-count bins).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = _validate_probs(y_prob)
    confidence = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        bin_edges = np.quantile(confidence, np.linspace(0.0, 1.0, n_bins + 1))
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0 + 1e-9
    else:
        raise ValueError(f"strategy must be 'uniform' or 'quantile', got {strategy}")

    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidence[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


def reliability_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict:
    """Per-bin data for plotting a reliability diagram.

    Returns dict with arrays:
        bin_centers, bin_confidence, bin_accuracy, bin_counts
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = _validate_probs(y_prob)
    confidence = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        bin_edges = np.quantile(confidence, np.linspace(0.0, 1.0, n_bins + 1))
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0 + 1e-9

    bin_centers, bin_conf, bin_acc, bin_counts = [], [], [], []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        bin_centers.append((lo + hi) / 2)
        bin_counts.append(int(mask.sum()))
        if mask.sum() > 0:
            bin_conf.append(float(confidence[mask].mean()))
            bin_acc.append(float(correct[mask].mean()))
        else:
            bin_conf.append(np.nan)
            bin_acc.append(np.nan)

    return {
        "bin_centers": np.array(bin_centers),
        "bin_confidence": np.array(bin_conf),
        "bin_accuracy": np.array(bin_acc),
        "bin_counts": np.array(bin_counts),
    }


# ---------------------------------------------------------------------------
# Convenience: a single dict with all metrics
# ---------------------------------------------------------------------------

def evaluate_all(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    name: str | None = None,
) -> dict:
    """Compute all metrics at once. Returns dict ready for tabulation."""
    return {
        **({"model": name} if name else {}),
        "log_loss": multiclass_log_loss(y_true, y_prob),
        "brier": multiclass_brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=10),
        "accuracy": accuracy(y_true, y_prob),
        "n": len(y_true),
    }


def uniform_baseline_metrics(y_true: np.ndarray) -> dict:
    """Metrics for a uniform 1/3 predictor — the absolute floor for any model."""
    n = len(y_true)
    y_prob_uniform = np.ones((n, 3)) / 3.0
    return evaluate_all(y_true, y_prob_uniform, name="uniform_1/3")
