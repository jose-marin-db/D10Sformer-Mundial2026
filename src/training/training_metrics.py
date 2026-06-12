"""
Training-time metrics for D10Sformer (MLM perplexity, MLM accuracy).

Unlike `src/eval/metrics.py` (which computes log-loss / Brier / ECE for the
result classifier), this module focuses on **online metrics during training**,
mainly for monitoring MLM convergence.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def mlm_loss_from_logits(
    logits: torch.Tensor,    # (B, T, V)
    labels: torch.Tensor,    # (B, T) — int ids; -100 for ignored positions
) -> torch.Tensor:
    """Standard masked cross-entropy with ignore_index=-100."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )


def mlm_accuracy(
    logits: torch.Tensor,    # (B, T, V)
    labels: torch.Tensor,    # (B, T)
) -> tuple[float, int]:
    """Top-1 accuracy on the non-ignored positions.

    Returns (accuracy, n_evaluated). If there are no masked positions, returns (0.0, 0).
    """
    mask = (labels != -100)
    n = int(mask.sum().item())
    if n == 0:
        return 0.0, 0
    preds = logits.argmax(dim=-1)
    correct = ((preds == labels) & mask).sum().item()
    return correct / n, n


def perplexity_from_loss(loss: float) -> float:
    """Perplexity = exp(loss). Common reporting unit in MLM (Bengio et al., 2003)."""
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")
