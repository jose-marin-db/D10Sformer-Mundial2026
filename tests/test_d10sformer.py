"""Tests for src/models/d10sformer.py — end-to-end model."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.d10sformer import D10Sformer, D10SformerConfig  # noqa: E402


VOCAB_SIZE = 4521


@pytest.fixture
def small_config():
    """A tiny config so tests run fast."""
    return D10SformerConfig(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        max_seq_length=128,
        num_segments=8,
        dropout=0.0,
        attention_dropout=0.0,
    )


@pytest.fixture
def small_model(small_config):
    torch.manual_seed(42)
    return D10Sformer(small_config)


def test_forward_returns_all_logits(small_model):
    B, T = 3, 40
    tok = torch.randint(1, VOCAB_SIZE, (B, T))
    seg = torch.randint(0, 8, (B, T))
    out = small_model(tok, seg)

    assert "result_logits" in out
    assert "score_logits" in out
    assert "mlm_logits" in out

    assert out["result_logits"].shape == (B, 3)
    assert out["score_logits"].shape == (B, 36)
    assert out["mlm_logits"].shape == (B, T, VOCAB_SIZE)


def test_attention_mask_keeps_outputs_finite(small_model):
    B, T = 2, 50
    tok = torch.randint(1, VOCAB_SIZE, (B, T))
    seg = torch.randint(0, 8, (B, T))
    mask = torch.ones(B, T)
    mask[:, 30:] = 0   # second half is padding
    out = small_model(tok, seg, attention_mask=mask)
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"NaN/Inf in {k}"


def test_return_hidden(small_model, small_config):
    tok = torch.randint(1, VOCAB_SIZE, (1, 10))
    seg = torch.zeros(1, 10, dtype=torch.long)
    out = small_model(tok, seg, return_hidden=True)
    assert out["hidden"].shape == (1, 10, small_config.d_model)


def test_backward_pass_full_model(small_model):
    """Full multi-task backward pass should populate grads everywhere."""
    B, T = 2, 30
    tok = torch.randint(1, VOCAB_SIZE, (B, T))
    seg = torch.randint(0, 8, (B, T))
    out = small_model(tok, seg)

    # Synthetic targets
    result_target = torch.randint(0, 3, (B,))
    score_target = torch.randint(0, 36, (B,))
    mlm_target = torch.randint(0, VOCAB_SIZE, (B, T))

    loss = (
        F.cross_entropy(out["result_logits"], result_target)
        + F.cross_entropy(out["score_logits"], score_target)
        + F.cross_entropy(out["mlm_logits"].reshape(-1, VOCAB_SIZE), mlm_target.reshape(-1))
    )
    loss.backward()

    # At least the embedding + first layer + heads must have grads
    assert small_model.embeddings.token_embedding.weight.grad is not None
    assert small_model.encoder.layers[0].attn.W_q.weight.grad is not None
    assert small_model.result_head.classifier.weight.grad is not None
    assert small_model.score_head.classifier.weight.grad is not None


def test_parameter_breakdown(small_model):
    breakdown = small_model.parameter_breakdown()
    assert "embeddings" in breakdown
    assert "encoder" in breakdown
    assert "TOTAL" in breakdown
    # Sanity: TOTAL must equal sum of trainable parameters
    assert breakdown["TOTAL"] == small_model.num_parameters(only_trainable=True)


def test_tied_mlm_weights_share_storage(small_config):
    cfg = small_config
    cfg.tie_mlm_weights = True
    m = D10Sformer(cfg)
    assert (
        m.mlm_head.decoder.weight.data_ptr()
        == m.embeddings.token_embedding.weight.data_ptr()
    )


def test_untied_mlm_weights_dont_share_storage(small_config):
    cfg = small_config
    cfg.tie_mlm_weights = False
    m = D10Sformer(cfg)
    assert (
        m.mlm_head.decoder.weight.data_ptr()
        != m.embeddings.token_embedding.weight.data_ptr()
    )


def test_default_config_param_count_is_in_expected_range():
    """At d=256, layers=6, the model should be 6-15M parameters."""
    cfg = D10SformerConfig(vocab_size=VOCAB_SIZE)   # all defaults
    m = D10Sformer(cfg)
    n = m.num_parameters()
    assert 4_000_000 < n < 20_000_000, f"Unexpected param count: {n:,}"
