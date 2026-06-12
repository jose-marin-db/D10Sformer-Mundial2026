"""Tests for src/training/training_metrics.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.training_metrics import (  # noqa: E402
    mlm_loss_from_logits,
    mlm_accuracy,
    perplexity_from_loss,
)


def test_mlm_loss_ignores_minus_100():
    B, T, V = 2, 4, 10
    logits = torch.randn(B, T, V)
    labels = torch.tensor([[0, -100, 2, -100],
                           [1, -100, -100, 3]], dtype=torch.long)
    # Loss with the same logits but DIFFERENT -100 positions must equal
    loss_a = mlm_loss_from_logits(logits, labels)
    # Now flip a -100 position's label to something arbitrary — loss should NOT change
    labels_b = labels.clone()
    labels_b[0, 1] = 5
    labels_b[0, 1] = -100   # ignored
    loss_b = mlm_loss_from_logits(logits, labels_b)
    assert torch.isclose(loss_a, loss_b)


def test_mlm_accuracy_perfect():
    """If logits are argmax-correct on every masked position, accuracy = 1.0."""
    B, T, V = 1, 3, 5
    labels = torch.tensor([[1, -100, 3]], dtype=torch.long)
    logits = torch.full((B, T, V), -10.0)
    # Make argmax = label at positions 0 and 2
    logits[0, 0, 1] = 10.0
    logits[0, 2, 3] = 10.0
    acc, n = mlm_accuracy(logits, labels)
    assert acc == 1.0
    assert n == 2


def test_mlm_accuracy_no_masked_positions():
    labels = torch.full((1, 5), -100, dtype=torch.long)
    logits = torch.randn(1, 5, 10)
    acc, n = mlm_accuracy(logits, labels)
    assert n == 0
    assert acc == 0.0


def test_perplexity_from_loss():
    assert math.isclose(perplexity_from_loss(0.0), 1.0)
    assert math.isclose(perplexity_from_loss(math.log(7.0)), 7.0)


def test_perplexity_overflow_returns_inf():
    assert perplexity_from_loss(1e6) == float("inf")
