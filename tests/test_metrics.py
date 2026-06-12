"""Tests for src/eval/metrics.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval.metrics import (  # noqa: E402
    multiclass_log_loss,
    multiclass_brier_score,
    expected_calibration_error,
    reliability_curve_data,
    accuracy,
    evaluate_all,
    uniform_baseline_metrics,
)


def test_log_loss_of_uniform_is_log3():
    """A uniform 1/3 predictor has log loss = ln(3) ≈ 1.0986."""
    y_true = np.array([0, 1, 2, 0, 1])
    y_prob = np.ones((5, 3)) / 3.0
    assert abs(multiclass_log_loss(y_true, y_prob) - np.log(3)) < 1e-6


def test_log_loss_of_perfect_is_near_zero():
    y_true = np.array([0, 1, 2, 0, 1])
    y_prob = np.eye(3)[y_true].astype(float)
    y_prob = y_prob * 0.999 + 0.0005
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    assert multiclass_log_loss(y_true, y_prob) < 0.01


def test_brier_perfect_is_zero():
    y_true = np.array([0, 1, 2])
    y_prob = np.eye(3)[y_true].astype(float)
    assert multiclass_brier_score(y_true, y_prob) < 1e-9


def test_brier_uniform_is_two_thirds():
    """For 3 classes and uniform 1/3 prediction, Brier = 2/3 exactly."""
    y_true = np.array([0, 1, 2])
    y_prob = np.ones((3, 3)) / 3.0
    assert abs(multiclass_brier_score(y_true, y_prob) - 2/3) < 1e-9


def test_ece_perfect_calibration():
    """Confidence equals empirical accuracy → ECE near 0."""
    n = 1000
    rng = np.random.default_rng(42)
    confidences = rng.uniform(0.4, 0.95, n)
    # Generate predictions correct with prob equal to confidence
    correct = rng.binomial(1, confidences)
    predicted = rng.integers(0, 3, n)
    y_true = np.where(correct == 1, predicted, (predicted + 1) % 3)
    # Build prob matrix with `confidence` mass on predicted, rest split
    y_prob = np.zeros((n, 3))
    for i in range(n):
        y_prob[i, predicted[i]] = confidences[i]
        others = [j for j in range(3) if j != predicted[i]]
        y_prob[i, others] = (1 - confidences[i]) / 2
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece < 0.05  # well calibrated by construction


def test_validate_probs_rejects_non_sum_one():
    """Probs must sum to 1."""
    bad = np.array([[0.5, 0.3, 0.1]])  # sums to 0.9
    with pytest.raises(ValueError, match="sum to 1"):
        multiclass_log_loss(np.array([0]), bad)


def test_validate_probs_rejects_nan():
    bad = np.array([[0.5, np.nan, 0.5]])
    with pytest.raises(ValueError):
        multiclass_log_loss(np.array([0]), bad)


def test_accuracy():
    y_true = np.array([0, 1, 2, 0])
    y_prob = np.array([
        [0.7, 0.2, 0.1],   # predicts 0 (correct)
        [0.1, 0.8, 0.1],   # predicts 1 (correct)
        [0.4, 0.4, 0.2],   # predicts 0 (wrong, truth=2)
        [0.5, 0.3, 0.2],   # predicts 0 (correct)
    ])
    assert accuracy(y_true, y_prob) == 0.75


def test_evaluate_all_returns_complete_dict():
    y_true = np.array([0, 1, 2, 0, 1])
    y_prob = np.ones((5, 3)) / 3.0
    result = evaluate_all(y_true, y_prob, name="test")
    for key in ("model", "log_loss", "brier", "ece", "accuracy", "n"):
        assert key in result


def test_uniform_baseline_metrics():
    y = np.array([0, 1, 2, 0, 1, 2])
    res = uniform_baseline_metrics(y)
    assert abs(res["log_loss"] - np.log(3)) < 1e-6
    assert abs(res["brier"] - 2/3) < 1e-6


def test_reliability_curve_data_returns_arrays():
    y_true = np.array([0, 1, 2, 0, 1])
    y_prob = np.ones((5, 3)) / 3.0
    data = reliability_curve_data(y_true, y_prob, n_bins=5)
    assert set(data.keys()) == {"bin_centers", "bin_confidence", "bin_accuracy", "bin_counts"}
    assert len(data["bin_centers"]) == 5
