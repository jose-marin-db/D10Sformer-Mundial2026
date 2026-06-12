"""Tests for src/data/feature_engineering.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.feature_engineering import (  # noqa: E402
    build_feature_matrix,
    temporal_split,
    prepare_xy,
    get_feature_columns,
    RESULT_HOME_WIN,
    RESULT_DRAW,
    RESULT_AWAY_WIN,
)


@pytest.fixture
def sample_matches() -> pd.DataFrame:
    """5 matches across 3 teams with known outcomes."""
    return pd.DataFrame([
        {"date": "2020-01-01", "home_team": "Argentina", "away_team": "Brazil",
         "home_score": 2, "away_score": 1, "tournament": "Friendly",
         "tournament_class": "friendly", "neutral": False,
         "home_elo_before": 1800, "away_elo_before": 1820,
         "expected_home_win_prob": 0.55},
        {"date": "2020-03-01", "home_team": "Argentina", "away_team": "France",
         "home_score": 1, "away_score": 1, "tournament": "Friendly",
         "tournament_class": "friendly", "neutral": False,
         "home_elo_before": 1810, "away_elo_before": 1830,
         "expected_home_win_prob": 0.52},
        {"date": "2020-06-01", "home_team": "Brazil", "away_team": "France",
         "home_score": 0, "away_score": 1, "tournament": "Friendly",
         "tournament_class": "friendly", "neutral": True,
         "home_elo_before": 1810, "away_elo_before": 1830,
         "expected_home_win_prob": 0.43},
        {"date": "2020-09-01", "home_team": "Argentina", "away_team": "Brazil",
         "home_score": 3, "away_score": 0, "tournament": "WC qual",
         "tournament_class": "wc_qualifier", "neutral": False,
         "home_elo_before": 1815, "away_elo_before": 1795,
         "expected_home_win_prob": 0.62},
        {"date": "2020-12-01", "home_team": "Brazil", "away_team": "France",
         "home_score": 2, "away_score": 2, "tournament": "Friendly",
         "tournament_class": "friendly", "neutral": False,
         "home_elo_before": 1810, "away_elo_before": 1840,
         "expected_home_win_prob": 0.48},
    ]).assign(date=lambda d: pd.to_datetime(d["date"]))


def test_build_feature_matrix_basic(sample_matches):
    fm = build_feature_matrix(sample_matches, min_date=None)
    assert len(fm) == 5
    assert "result" in fm.columns
    assert "home_elo" in fm.columns
    assert "elo_diff" in fm.columns


def test_no_leakage_first_match_is_cold_start(sample_matches):
    """First match for each team should have NaN form features (no history)."""
    fm = build_feature_matrix(sample_matches, min_date=None)
    first = fm.iloc[0]
    assert first["home_team"] == "Argentina"
    assert pd.isna(first["home_form5_pts"])
    assert first["home_form5_n"] == 0


def test_form_computation_correct(sample_matches):
    """Argentina's 2nd match should have form from match 1 (won)."""
    fm = build_feature_matrix(sample_matches, min_date=None)
    arg_match_2 = fm[fm["home_team"] == "Argentina"].iloc[1]  # vs France
    assert arg_match_2["home_form5_pts"] == 3.0  # 1 win = 3 points avg
    assert arg_match_2["home_form5_n"] == 1


def test_h2h_finds_prior_meeting(sample_matches):
    """The rematch ARG vs BRA should see 1 prior meeting where ARG won."""
    fm = build_feature_matrix(sample_matches, min_date=None)
    rematch = fm[
        (fm["home_team"] == "Argentina") & (fm["away_team"] == "Brazil")
    ].iloc[1]
    assert rematch["h2h_n_matches"] == 1
    assert rematch["h2h_home_wins"] == 1


def test_result_encoding(sample_matches):
    """Result column matches expected values."""
    fm = build_feature_matrix(sample_matches, min_date=None)
    expected = [RESULT_HOME_WIN, RESULT_DRAW, RESULT_AWAY_WIN, RESULT_HOME_WIN, RESULT_DRAW]
    assert fm["result"].tolist() == expected


def test_temporal_split_disjoint():
    """train, val, test must be disjoint in time."""
    dates = pd.date_range("2022-01-01", "2025-01-01", freq="MS")
    n = len(dates)
    df = pd.DataFrame({
        "date": dates,
        "home_team": ["A"] * n, "away_team": ["B"] * n,
        "tournament": ["Friendly"] * n, "tournament_class": ["friendly"] * n,
        "neutral": [False] * n,
        "home_elo_before": [1500.0] * n, "away_elo_before": [1500.0] * n,
        "expected_home_win_prob": [0.5] * n,
        "result": [0] * n, "home_score": [1] * n, "away_score": [0] * n,
    })
    train, val, test = temporal_split(df, "2023-01-01", "2024-01-01")
    assert (train["date"] < pd.Timestamp("2023-01-01")).all()
    assert (val["date"] >= pd.Timestamp("2023-01-01")).all()
    assert (val["date"] < pd.Timestamp("2024-01-01")).all()
    assert (test["date"] >= pd.Timestamp("2024-01-01")).all()
    # All three splits should have data
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0


def test_prepare_xy_no_leakage_columns(sample_matches):
    """X should NOT contain identifiers or target."""
    fm = build_feature_matrix(sample_matches, min_date=None)
    X, y = prepare_xy(fm)
    forbidden = {"date", "home_team", "away_team", "result", "home_score", "away_score"}
    assert not (forbidden & set(X.columns)), f"X has forbidden cols: {forbidden & set(X.columns)}"
    assert len(y) == len(X)


def test_prepare_xy_onehot(sample_matches):
    """tournament_class should be expanded to one-hot when requested."""
    fm = build_feature_matrix(sample_matches, min_date=None)
    X, _ = prepare_xy(fm, onehot_categoricals=True)
    # Should have at least one tournament_class_* column
    assert any(c.startswith("tournament_class_") for c in X.columns)
    assert "tournament_class" not in X.columns
