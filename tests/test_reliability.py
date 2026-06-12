"""Tests for src/eval/reliability.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval.reliability import reliability_curve, temperature_scale  # noqa: E402


def test_perfect_calibration_has_zero_ece():
    """If predictions are well-calibrated, ECE should be near zero."""
    np.random.seed(42)
    n = 5000
    # Synthetic: confidence == empirical accuracy by design
    y_prob = np.random.dirichlet([1, 1, 1], size=n)
    confidences = y_prob.max(axis=1)
    preds = y_prob.argmax(axis=1)
    y_true = np.array([
        preds[i] if np.random.random() < confidences[i] else (preds[i] + 1) % 3
        for i in range(n)
    ])
    curve = reliability_curve(y_true, y_prob, n_bins=10)
    assert curve.ece < 0.05, f"Expected near-zero ECE, got {curve.ece}"


def test_overconfident_model_has_high_ece():
    """If the model is overconfident (claims 0.95 but only 0.5 accuracy), ECE is high."""
    n = 1000
    y_prob = np.zeros((n, 3))
    y_prob[:, 0] = 0.95
    y_prob[:, 1] = 0.025
    y_prob[:, 2] = 0.025
    # Only 50% actually correct
    y_true = np.where(np.arange(n) < n // 2, 0, 1)
    curve = reliability_curve(y_true, y_prob, n_bins=10)
    # Expected ECE ≈ |0.95 - 0.5| ≈ 0.45
    assert curve.ece > 0.3, f"Expected high ECE for overconfident model, got {curve.ece}"


def test_bin_counts_sum_to_n():
    n = 200
    y_prob = np.random.dirichlet([1, 1, 1], size=n)
    y_true = np.random.randint(0, 3, n)
    curve = reliability_curve(y_true, y_prob, n_bins=10)
    assert int(curve.bin_counts.sum()) == n


def test_shape_validation():
    import pytest
    with pytest.raises(ValueError):
        reliability_curve(np.array([0, 1]), np.array([0.5, 0.5]))   # 1-D y_prob
    with pytest.raises(ValueError):
        reliability_curve(np.array([0, 1, 2]), np.array([[0.3, 0.7]]))   # length mismatch


def test_temperature_scaling_runs():
    np.random.seed(0)
    n, k = 200, 3
    logits = np.random.randn(n, k)
    y_true = np.random.randint(0, k, n)
    T, probs = temperature_scale(logits, y_true, n_iter=10)
    assert T > 0
    assert probs.shape == (n, k)
    # Each row sums to ~1
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_temperature_scaling_reduces_ece_on_overconfident():
    """Smoke test: on a synthetically overconfident model, T > 1 should reduce ECE."""
    np.random.seed(42)
    n, k = 1000, 3
    # Overconfident logits: large magnitudes
    true_logits = np.random.randn(n, k) * 0.5
    y_true = true_logits.argmax(axis=1)
    # Flip 50% of labels (so confidence is way above accuracy)
    flip_mask = np.random.rand(n) < 0.5
    y_true[flip_mask] = (y_true[flip_mask] + 1) % k
    inflated_logits = true_logits * 5.0   # massively overconfident

    # ECE before calibration
    probs_before = np.exp(inflated_logits) / np.exp(inflated_logits).sum(axis=1, keepdims=True)
    ece_before = reliability_curve(y_true, probs_before).ece

    T, probs_after = temperature_scale(inflated_logits, y_true, n_iter=50)
    ece_after = reliability_curve(y_true, probs_after).ece

    assert T > 1.0, f"Expected T > 1 to reduce overconfidence; got T={T}"
    assert ece_after < ece_before, f"ECE did not decrease: {ece_before} → {ece_after}"
