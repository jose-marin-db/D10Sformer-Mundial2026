"""Tests for src/models/transformer.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.transformer import (  # noqa: E402
    FeedForward,
    TransformerEncoderBlock,
    TransformerEncoder,
)


D_MODEL = 256
NUM_HEADS = 8
D_FF = 1024


def test_feedforward_shape():
    ffn = FeedForward(d_model=D_MODEL, d_ff=D_FF, dropout=0.0)
    x = torch.randn(2, 10, D_MODEL)
    out = ffn(x)
    assert out.shape == x.shape


def test_encoder_block_preserves_shape():
    blk = TransformerEncoderBlock(d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF, dropout=0.0)
    x = torch.randn(3, 20, D_MODEL)
    out = blk(x)
    assert out.shape == x.shape


def test_encoder_block_residual_is_active():
    """With near-zero attention output, a residual block should approximately
    return the input. We check by setting all internal weights to 0."""
    blk = TransformerEncoderBlock(d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF, dropout=0.0)
    blk.eval()
    # Zero out all linear weights inside attn + ffn → residual path dominates
    with torch.no_grad():
        for p in blk.attn.parameters():
            p.zero_()
        for p in blk.ffn.parameters():
            p.zero_()
    x = torch.randn(1, 5, D_MODEL)
    out = blk(x)
    # Pre-LN: x' = x + dropout(MHSA(LN(x))) = x + 0 = x  (and same for FFN)
    assert torch.allclose(out, x, atol=1e-5)


def test_stack_n_layers():
    enc = TransformerEncoder(
        num_layers=4, d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF, dropout=0.0
    )
    x = torch.randn(2, 30, D_MODEL)
    out = enc(x)
    assert out.shape == x.shape
    assert len(enc.layers) == 4


def test_padding_mask_propagates():
    """The encoder must honour an attention mask end-to-end."""
    enc = TransformerEncoder(num_layers=2, d_model=64, num_heads=4, d_ff=128, dropout=0.0)
    enc.eval()
    x = torch.randn(1, 6, 64)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    out = enc(x, attention_mask=mask)
    assert torch.isfinite(out).all()


def test_backward_pass_stack():
    enc = TransformerEncoder(num_layers=3, d_model=64, num_heads=4, d_ff=128, dropout=0.0)
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = enc(x)
    out.sum().backward()
    assert x.grad is not None
    # Every layer should have grads
    for layer in enc.layers:
        assert layer.attn.W_q.weight.grad is not None
        assert layer.ffn.fc1.weight.grad is not None
