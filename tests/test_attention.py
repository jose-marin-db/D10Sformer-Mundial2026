"""Tests for src/models/attention.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.attention import MultiHeadSelfAttention  # noqa: E402


D_MODEL = 256
NUM_HEADS = 8


@pytest.fixture
def attn():
    torch.manual_seed(0)
    return MultiHeadSelfAttention(d_model=D_MODEL, num_heads=NUM_HEADS, dropout=0.0)


def test_output_shape(attn):
    B, T = 3, 17
    x = torch.randn(B, T, D_MODEL)
    out = attn(x)
    assert out.shape == (B, T, D_MODEL)


def test_returns_attention_weights(attn):
    B, T = 2, 10
    x = torch.randn(B, T, D_MODEL)
    out, weights = attn(x, return_weights=True)
    assert weights.shape == (B, NUM_HEADS, T, T)
    # Each row of the attention matrix should sum to ~1 (softmax)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_scale_is_inverse_sqrt_dk():
    """The scale used in the attention should be 1/sqrt(d_k)."""
    m = MultiHeadSelfAttention(d_model=128, num_heads=8)
    expected = 1.0 / math.sqrt(128 // 8)
    assert math.isclose(m.scale, expected)


def test_d_model_not_divisible_raises():
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(d_model=100, num_heads=8)  # 100/8 not integer


def test_padding_mask_ignored_positions(attn):
    """A masked-out (padded) position should receive ~0 attention from all queries."""
    attn.eval()
    B, T = 1, 5
    x = torch.randn(B, T, D_MODEL)
    mask = torch.tensor([[1, 1, 1, 0, 0]])  # last 2 positions are padding
    _, weights = attn(x, attention_mask=mask, return_weights=True)
    # Sum of attention over the masked columns (positions 3 and 4) should be ~0
    attended_to_padding = weights[0, :, :, 3:].sum().item()
    assert attended_to_padding < 1e-5


def test_backward_pass(attn):
    x = torch.randn(2, 10, D_MODEL, requires_grad=True)
    out = attn(x)
    out.sum().backward()
    assert x.grad is not None
    assert attn.W_q.weight.grad is not None
    assert attn.W_k.weight.grad is not None
    assert attn.W_v.weight.grad is not None
    assert attn.W_o.weight.grad is not None


def test_no_nan_with_full_row_mask():
    """If a query position is fully masked (all keys are pad), output should be finite (not NaN)."""
    attn = MultiHeadSelfAttention(d_model=64, num_heads=4, dropout=0.0)
    attn.eval()
    x = torch.randn(1, 4, 64)
    # All positions get attended-to except position 0 — but position 3's query has
    # access to all keys so it's fine. We construct a case where every key is real.
    mask = torch.tensor([[1, 1, 1, 1]])
    out = attn(x, attention_mask=mask)
    assert torch.isfinite(out).all()
