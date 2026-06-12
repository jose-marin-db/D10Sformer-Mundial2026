"""Tests for src/models/embeddings.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.embeddings import MatchEmbedding, DEFAULT_NUM_SEGMENTS  # noqa: E402


VOCAB_SIZE = 4521
D_MODEL = 256
MAX_LEN = 512


@pytest.fixture
def embedder():
    torch.manual_seed(42)
    return MatchEmbedding(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        max_seq_length=MAX_LEN,
        num_segments=DEFAULT_NUM_SEGMENTS,
    )


def test_forward_shape(embedder):
    B, T = 4, 50
    tok = torch.randint(0, VOCAB_SIZE, (B, T))
    seg = torch.randint(0, DEFAULT_NUM_SEGMENTS, (B, T))
    out = embedder(tok, seg)
    assert out.shape == (B, T, D_MODEL)


def test_pad_token_zero_embedding(embedder):
    """[PAD] token should have a zero token-embedding after init."""
    pad_emb = embedder.token_embedding.weight[embedder.pad_token_id]
    assert torch.allclose(pad_emb, torch.zeros_like(pad_emb))


def test_position_embedding_independent_of_batch(embedder):
    """Same token+segment at the same position in two batches → same embedding."""
    tok = torch.tensor([[10, 20, 30, 40]])
    seg = torch.tensor([[0, 0, 1, 1]])
    embedder.eval()  # disable dropout for deterministic comparison
    out1 = embedder(tok, seg)
    out2 = embedder(tok, seg)
    assert torch.allclose(out1, out2)


def test_position_changes_output(embedder):
    """Different positions of the same token should give different embeddings."""
    embedder.eval()
    tok_a = torch.tensor([[42]])
    seg_a = torch.tensor([[0]])
    # Same token, different position — emulate by padding with another token
    tok_b = torch.tensor([[42, 42]])
    seg_b = torch.tensor([[0, 0]])
    out_a = embedder(tok_a, seg_a)        # at pos 0
    out_b = embedder(tok_b, seg_b)        # pos 0 and pos 1
    # Position 0 embeddings should be equal across both calls
    assert torch.allclose(out_a[0, 0], out_b[0, 0])
    # Position 1 embedding (same token, different position) must differ
    assert not torch.allclose(out_a[0, 0], out_b[0, 1])


def test_segment_changes_output(embedder):
    """Same token at same position but different segment → different embedding."""
    embedder.eval()
    tok = torch.tensor([[42]])
    out_seg0 = embedder(tok, torch.tensor([[0]]))
    out_seg1 = embedder(tok, torch.tensor([[1]]))
    assert not torch.allclose(out_seg0, out_seg1)


def test_sequence_too_long_raises(embedder):
    bad = torch.zeros(1, MAX_LEN + 1, dtype=torch.long)
    seg = torch.zeros(1, MAX_LEN + 1, dtype=torch.long)
    with pytest.raises(ValueError):
        embedder(bad, seg)


def test_backward_pass(embedder):
    """Loss.backward() should populate grads on all embedding tables."""
    tok = torch.randint(1, VOCAB_SIZE, (2, 10))   # avoid pad_token_id=0
    seg = torch.randint(0, DEFAULT_NUM_SEGMENTS, (2, 10))
    out = embedder(tok, seg)
    loss = out.sum()
    loss.backward()
    assert embedder.token_embedding.weight.grad is not None
    assert embedder.position_embedding.weight.grad is not None
    assert embedder.segment_embedding.weight.grad is not None
