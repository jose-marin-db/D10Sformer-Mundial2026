"""Tests for src/data/match_corpus_builder.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.match_corpus_builder import (  # noqa: E402
    build_international_documents,
    build_statsbomb_document,
)


def test_builder_uses_elo_before_columns():
    """Real parquet uses `home_elo_before` / `away_elo_before` — must work."""
    df = pd.DataFrame({
        "date": ["2022-06-01", "2022-06-15"],
        "home_team": ["Argentina", "Brazil"],
        "away_team": ["France", "Spain"],
        "home_score": [2, 1],
        "away_score": [1, 1],
        "tournament": ["Friendly", "Friendly"],
        "neutral": [False, True],
        "home_elo_before": [2100.0, 2050.0],
        "away_elo_before": [2080.0, 2000.0],
    })
    docs = build_international_documents(df, include_features=True)
    assert len(docs) == 2
    assert docs[0].features is not None
    assert docs[0].features.home_elo == 2100.0
    assert docs[0].features.away_elo == 2080.0


def test_builder_uses_legacy_elo_columns():
    """Legacy column names `home_elo` / `away_elo` should still work."""
    df = pd.DataFrame({
        "date": ["2022-06-01"],
        "home_team": ["Argentina"], "away_team": ["France"],
        "home_score": [2], "away_score": [1],
        "tournament": ["Friendly"], "neutral": [False],
        "home_elo": [2100.0], "away_elo": [2080.0],
    })
    docs = build_international_documents(df, include_features=True)
    assert docs[0].features.home_elo == 2100.0
    assert docs[0].features.away_elo == 2080.0


def test_builder_handles_missing_elo_gracefully():
    df = pd.DataFrame({
        "date": ["2022-06-01"],
        "home_team": ["A"], "away_team": ["B"],
        "home_score": [0], "away_score": [0],
        "tournament": ["Friendly"], "neutral": [False],
    })
    docs = build_international_documents(df, include_features=True)
    assert docs[0].features is None


def test_builder_picks_up_form_and_goals():
    df = pd.DataFrame({
        "date": ["2022-06-01"],
        "home_team": ["A"], "away_team": ["B"],
        "home_score": [0], "away_score": [0],
        "tournament": ["Friendly"], "neutral": [False],
        "home_elo_before": [2000.0], "away_elo_before": [1900.0],
        "home_form_pts_5": [2.4], "away_form_pts_5": [1.0],
        "home_recent_goals_5": [1.8], "away_recent_goals_5": [0.6],
    })
    docs = build_international_documents(df, include_features=True)
    f = docs[0].features
    assert f.home_form_pts == 2.4
    assert f.away_form_pts == 1.0
    assert f.home_recent_goals == 1.8
    assert f.away_recent_goals == 0.6


def test_result_inferred_from_scores():
    df = pd.DataFrame({
        "date": ["2022-06-01", "2022-06-02", "2022-06-03"],
        "home_team": ["A", "C", "E"],
        "away_team": ["B", "D", "F"],
        "home_score": [2, 1, 0],
        "away_score": [1, 1, 1],
        "tournament": ["Friendly"] * 3,
        "neutral": [False, False, False],
    })
    docs = build_international_documents(df, include_features=False)
    assert docs[0].result == "home_win"
    assert docs[1].result == "draw"
    assert docs[2].result == "away_win"


def test_neutral_gives_neutral_venue():
    df = pd.DataFrame({
        "date": ["2022-06-01"],
        "home_team": ["A"], "away_team": ["B"],
        "home_score": [1], "away_score": [0],
        "tournament": ["WC"], "neutral": [True],
    })
    docs = build_international_documents(df)
    assert docs[0].venue == "neutral"


def test_min_date_filter():
    df = pd.DataFrame({
        "date": ["2010-01-01", "2020-01-01", "2025-01-01"],
        "home_team": ["A", "C", "E"], "away_team": ["B", "D", "F"],
        "home_score": [0, 1, 2], "away_score": [0, 1, 2],
        "tournament": ["Friendly"] * 3, "neutral": [False] * 3,
    })
    docs = build_international_documents(df, min_date="2015-01-01")
    assert len(docs) == 2   # 2020 + 2025
