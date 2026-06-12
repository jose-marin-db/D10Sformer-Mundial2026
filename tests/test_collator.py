"""Tests for src/data/collator.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.vocabulary import FootballVocab, SPECIAL_TOKENS  # noqa: E402
from data.tokenizer import MatchTokenizer, MatchDocument, PlayerRef  # noqa: E402
from data.dataset import MatchDataset  # noqa: E402
from data.collator import MLMCollator, CollatedBatch  # noqa: E402


def _vocab() -> FootballVocab:
    return FootballVocab.build_from_data(
        teams=["Argentina", "France", "Brazil"],
        tournaments=["Friendly", "FIFA World Cup"],
        stages=["group", "final"],
        player_appearances={f"100{i}": 50 for i in range(1, 30)},
        player_positions={f"100{i}": "FW" for i in range(1, 30)},
        k_player_threshold=10,
    )


def _make_dataset(vocab):
    matches = [
        MatchDocument(tournament="Friendly", team_a="Argentina", team_b="France",
                      venue="home", result="home_win", home_score=2, away_score=1,
                      lineup_a=[PlayerRef(player_id=f"100{i+1}", position="FW") for i in range(11)]),
        MatchDocument(tournament="FIFA World Cup", stage="final",
                      team_a="Brazil", team_b="Argentina", venue="neutral",
                      result="draw", home_score=1, away_score=1),
        MatchDocument(tournament="Friendly", team_a="Argentina", team_b="France",
                      venue="home"),   # no targets
    ]
    return MatchDataset(matches, MatchTokenizer(vocab))


def test_collator_returns_collated_batch():
    vocab = _vocab()
    ds = _make_dataset(vocab)
    collator = MLMCollator(vocab, seed=42)
    batch = collator([ds[0], ds[1], ds[2]])
    assert isinstance(batch, CollatedBatch)


def test_padding_to_max_length():
    vocab = _vocab()
    ds = _make_dataset(vocab)
    collator = MLMCollator(vocab, seed=42)
    samples = [ds[0], ds[1], ds[2]]
    max_len = max(s.length for s in samples)
    batch = collator(samples)
    assert batch.token_ids.shape == (3, max_len)
    assert batch.segment_ids.shape == (3, max_len)
    assert batch.attention_mask.shape == (3, max_len)


def test_attention_mask_marks_padding():
    vocab = _vocab()
    ds = _make_dataset(vocab)
    collator = MLMCollator(vocab, seed=42)
    samples = [ds[0], ds[1]]   # different lengths
    batch = collator(samples)
    # Per row, sum of attention_mask must equal the original sample length
    for i, s in enumerate(samples):
        assert batch.attention_mask[i].sum().item() == s.length


def test_mlm_labels_ignore_unmasked():
    vocab = _vocab()
    ds = _make_dataset(vocab)
    collator = MLMCollator(vocab, mlm_probability=0.15, seed=42)
    batch = collator([ds[0], ds[1], ds[2]])
    # All positions that were NOT chosen for masking must have label = -100
    # The non-ignored positions should be a small fraction (~15% of real tokens)
    non_ignored = (batch.mlm_labels != -100).sum().item()
    total_real = batch.attention_mask.sum().item()
    # With seed 42 we should be roughly within (5%, 30%)
    assert 0 < non_ignored < total_real
    frac = non_ignored / total_real
    assert 0.02 < frac < 0.35, f"Masking ratio out of bounds: {frac}"


def test_special_tokens_never_masked():
    vocab = _vocab()
    ds = _make_dataset(vocab)
    collator = MLMCollator(vocab, mlm_probability=0.5, seed=7)   # high prob to stress test
    batch = collator([ds[0], ds[1], ds[2]])

    special_ids = {vocab.token_to_id[t] for t in SPECIAL_TOKENS if t in vocab.token_to_id}
    # For every position whose ORIGINAL token was a special, mlm_labels must be -100
    # (i.e., it wasn't selected for masking). We need the original tokens — those
    # are still in mlm_labels where != -100, but for the special check we look at
    # the attention-masked positions and verify the collator didn't sample them.
    for b in range(batch.token_ids.shape[0]):
        for t in range(batch.token_ids.shape[1]):
            if batch.mlm_labels[b, t].item() != -100:
                orig = batch.mlm_labels[b, t].item()
                assert orig not in special_ids, (
                    f"Special token id {orig} was sampled for masking — forbidden"
                )


def test_pad_positions_never_masked():
    vocab = _vocab()
    ds = _make_dataset(vocab)
    collator = MLMCollator(vocab, mlm_probability=0.5, seed=7)
    batch = collator([ds[0], ds[1]])    # different lengths → there's real padding
    # Every position where attention_mask == 0 must have mlm_label == -100
    pad_positions = (batch.attention_mask == 0)
    labels_at_pad = batch.mlm_labels[pad_positions]
    assert (labels_at_pad == -100).all().item()


def test_result_labels_use_minus_100_for_missing():
    vocab = _vocab()
    ds = _make_dataset(vocab)
    collator = MLMCollator(vocab, seed=42)
    batch = collator([ds[0], ds[1], ds[2]])
    assert batch.result_labels[0].item() != -100
    assert batch.result_labels[1].item() != -100
    assert batch.result_labels[2].item() == -100   # third match had no target
    assert batch.score_labels[2].item() == -100


def test_reproducibility_with_seed():
    vocab = _vocab()
    ds = _make_dataset(vocab)
    collator_a = MLMCollator(vocab, seed=123)
    collator_b = MLMCollator(vocab, seed=123)
    batch_a = collator_a([ds[0], ds[1]])
    batch_b = collator_b([ds[0], ds[1]])
    assert torch.equal(batch_a.token_ids, batch_b.token_ids)
    assert torch.equal(batch_a.mlm_labels, batch_b.mlm_labels)


def test_full_mlm_probability_picks_most_real_tokens():
    """At mlm_probability=1.0 every non-special, non-pad token should be selected."""
    vocab = _vocab()
    ds = _make_dataset(vocab)
    collator = MLMCollator(vocab, mlm_probability=1.0, seed=0)
    batch = collator([ds[0]])
    # Counts
    real_tokens = batch.attention_mask.sum().item()
    masked_positions = (batch.mlm_labels != -100).sum().item()
    # Every real, non-special token should be selected
    sample = ds[0]
    special_ids = {vocab.token_to_id[t] for t in SPECIAL_TOKENS if t in vocab.token_to_id}
    n_specials_in_sample = sum(1 for tid in sample.token_ids if tid in special_ids)
    expected_max_masked = real_tokens - n_specials_in_sample
    assert masked_positions == expected_max_masked
