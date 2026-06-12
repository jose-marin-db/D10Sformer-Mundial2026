"""Tests for src/models/heads.py."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.heads import MLMHead, ResultHead, ScoreHead  # noqa: E402


D_MODEL = 256
VOCAB_SIZE = 4521


def test_mlm_head_shape():
    head = MLMHead(d_model=D_MODEL, vocab_size=VOCAB_SIZE)
    h = torch.randn(2, 50, D_MODEL)
    logits = head(h)
    assert logits.shape == (2, 50, VOCAB_SIZE)


def test_mlm_head_weight_tying():
    """When tied_embedding is given, the decoder weight is the same tensor."""
    emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
    head = MLMHead(d_model=D_MODEL, vocab_size=VOCAB_SIZE, tied_embedding=emb)
    assert head.decoder.weight.data_ptr() == emb.weight.data_ptr()


def test_mlm_head_independent_when_not_tied():
    head = MLMHead(d_model=D_MODEL, vocab_size=VOCAB_SIZE, tied_embedding=None)
    emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
    assert head.decoder.weight.data_ptr() != emb.weight.data_ptr()


def test_result_head_outputs_3_classes():
    head = ResultHead(d_model=D_MODEL)
    h = torch.randn(4, 30, D_MODEL)
    logits = head(h)
    assert logits.shape == (4, 3)


def test_score_head_outputs_36_classes():
    head = ScoreHead(d_model=D_MODEL, num_score_classes=36)
    h = torch.randn(4, 30, D_MODEL)
    logits = head(h)
    assert logits.shape == (4, 36)


def test_classification_head_uses_cls_only():
    """Changing tokens at positions > 0 should NOT affect the result head output."""
    torch.manual_seed(0)
    head = ResultHead(d_model=D_MODEL)
    head.eval()
    h1 = torch.randn(1, 10, D_MODEL)
    h2 = h1.clone()
    h2[:, 1:, :] = torch.randn(1, 9, D_MODEL)   # mutate everything except CLS
    out1 = head(h1)
    out2 = head(h2)
    assert torch.allclose(out1, out2)


def test_backward_pass_all_heads():
    h = torch.randn(2, 20, D_MODEL, requires_grad=True)
    mlm = MLMHead(d_model=D_MODEL, vocab_size=VOCAB_SIZE)
    res = ResultHead(d_model=D_MODEL)
    sco = ScoreHead(d_model=D_MODEL)
    loss = mlm(h).sum() + res(h).sum() + sco(h).sum()
    loss.backward()
    assert h.grad is not None
