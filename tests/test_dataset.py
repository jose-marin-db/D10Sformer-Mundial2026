"""Tests for src/data/dataset.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.vocabulary import FootballVocab  # noqa: E402
from data.tokenizer import MatchTokenizer, MatchDocument, PlayerRef  # noqa: E402
from data.dataset import MatchDataset, MatchSample  # noqa: E402


def _vocab() -> FootballVocab:
    return FootballVocab.build_from_data(
        teams=["Argentina", "France", "Brazil"],
        tournaments=["Friendly", "FIFA World Cup"],
        stages=["group", "final"],
        player_appearances={f"100{i}": 50 for i in range(1, 30)},
        player_positions={f"100{i}": "FW" for i in range(1, 30)},
        k_player_threshold=10,
    )


def _make_matches() -> list[MatchDocument]:
    return [
        MatchDocument(tournament="Friendly", team_a="Argentina", team_b="France",
                      venue="home", result="home_win", home_score=2, away_score=1),
        MatchDocument(tournament="FIFA World Cup", stage="final",
                      team_a="Brazil", team_b="Argentina", venue="neutral",
                      result="draw", home_score=1, away_score=1,
                      lineup_a=[PlayerRef(player_id=f"100{i+1}", position="FW") for i in range(11)]),
        # One match without a target (should still tokenize, target_*_id = None)
        MatchDocument(tournament="Friendly", team_a="Argentina", team_b="France", venue="home"),
    ]


def test_dataset_length():
    ds = MatchDataset(_make_matches(), MatchTokenizer(_vocab()))
    assert len(ds) == 3


def test_getitem_returns_match_sample():
    ds = MatchDataset(_make_matches(), MatchTokenizer(_vocab()))
    sample = ds[0]
    assert isinstance(sample, MatchSample)
    assert sample.length == len(sample.token_ids) == len(sample.segment_ids)
    assert sample.target_result_id is not None
    assert sample.target_score_id is not None


def test_sample_without_target_has_none():
    ds = MatchDataset(_make_matches(), MatchTokenizer(_vocab()))
    sample = ds[2]   # third match has no result/scores
    assert sample.target_result_id is None
    assert sample.target_score_id is None


def test_drop_no_target_filters():
    ds_kept = MatchDataset(_make_matches(), MatchTokenizer(_vocab()), drop_no_target=False)
    ds_dropped = MatchDataset(_make_matches(), MatchTokenizer(_vocab()), drop_no_target=True)
    assert len(ds_kept) == 3
    assert len(ds_dropped) == 2


def test_length_stats_returns_full_dict():
    ds = MatchDataset(_make_matches(), MatchTokenizer(_vocab()))
    s = ds.length_stats()
    for k in ["n", "min", "max", "mean", "median", "p90", "p99"]:
        assert k in s
    assert s["n"] == 3
    assert s["min"] <= s["median"] <= s["max"]
