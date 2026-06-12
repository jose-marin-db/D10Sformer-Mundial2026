import sys
import tempfile
import pickle
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.tokenizer import MatchDocument, RollingFeatures
from training.ft_training_pipeline import rmean, compute_rolling_features, extract_df_from_pkl


def test_rmean():
    # Empty history
    assert np.isnan(rmean([], 0))
    
    # History with fewer than WINDOW items
    history = [(3, 2.0), (1, 1.0)]
    # rmean for pts (idx 0)
    assert rmean(history, 0) == (3 + 1) / 2
    # rmean for goals (idx 1)
    assert rmean(history, 1) == (2.0 + 1.0) / 2
    
    # History with more than WINDOW items (WINDOW = 5)
    history = [(1, 0.0), (3, 1.0), (0, 0.0), (1, 1.0), (3, 4.0), (3, 2.0)]
    # last 5 items are: (3, 1.0), (0, 0.0), (1, 1.0), (3, 4.0), (3, 2.0)
    assert rmean(history, 0) == (3 + 0 + 1 + 3 + 3) / 5
    assert rmean(history, 1) == (1.0 + 0.0 + 1.0 + 4.0 + 2.0) / 5


def test_compute_rolling_features():
    df_int = pd.DataFrame({
        'date': pd.to_datetime(['2026-06-01', '2026-06-02', '2026-06-03']),
        'home_team': ['Argentina', 'Brazil', 'Argentina'],
        'away_team': ['Brazil', 'Chile', 'Chile'],
        'home_score': [2.0, 1.0, 3.0],
        'away_score': [1.0, 1.0, 0.0],
        'tournament': ['Friendly', 'Friendly', 'Friendly'],
        'tournament_class': ['friendly', 'friendly', 'friendly'],
        'neutral': [0, 0, 0]
    })
    
    df_sorted = compute_rolling_features(df_int)
    assert 'home_form_pts_5' in df_sorted.columns
    assert 'away_form_pts_5' in df_sorted.columns
    assert 'home_recent_goals_5' in df_sorted.columns
    assert 'away_recent_goals_5' in df_sorted.columns
    assert 'home_form10_pts' in df_sorted.columns
    assert 'away_form10_pts' in df_sorted.columns
    
    # First match Argentina vs Brazil: no history yet, so should be nan
    assert np.isnan(df_sorted.loc[0, 'home_form_pts_5'])
    assert np.isnan(df_sorted.loc[0, 'away_form_pts_5'])
    
    # Third match Argentina vs Chile: Argentina has played 1 match (won 2-1 against Brazil)
    # So home_form_pts_5 for Argentina should be 3.0 and home_recent_goals_5 should be 2.0
    assert df_sorted.loc[2, 'home_form_pts_5'] == 3.0
    assert df_sorted.loc[2, 'home_recent_goals_5'] == 2.0


def test_extract_df_from_pkl():
    doc1 = MatchDocument(
        tournament='friendly',
        team_a='Argentina',
        team_b='Brazil',
        venue='home',
        features=RollingFeatures(
            home_elo=1800.0,
            away_elo=1700.0,
            home_form_pts=2.5,
            away_form_pts=1.8,
            home_recent_goals=2.2,
            away_recent_goals=1.1
        ),
        home_score=2,
        away_score=1,
        result='home_win'
    )
    
    doc2 = MatchDocument(
        tournament='friendly',
        team_a='Brazil',
        team_b='Chile',
        venue='neutral',
        features=None,
        home_score=None,
        away_score=None,
        result=None
    )
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        pkl_path = Path(tmp_dir) / "test_corpus.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump([doc1, doc2], f)
            
        df = extract_df_from_pkl(pkl_path)
        
        assert len(df) == 2
        assert df.loc[0, 'tournament_class'] == 'friendly'
        assert df.loc[0, 'neutral'] == 0
        assert df.loc[0, 'venue'] == 'home'
        assert df.loc[0, 'home_elo'] == 1800.0
        assert df.loc[0, 'away_elo'] == 1700.0
        assert df.loc[0, 'elo_diff'] == 100.0
        assert df.loc[0, 'home_form5_pts'] == 2.5
        assert df.loc[0, 'away_form5_pts'] == 1.8
        assert df.loc[0, 'home_form5_gf'] == 2.2
        assert df.loc[0, 'away_form5_gf'] == 1.1
        assert df.loc[0, 'home_form10_pts'] == 2.5
        assert df.loc[0, 'away_form10_pts'] == 1.8
        assert df.loc[0, 'score_h'] == 2
        assert df.loc[0, 'score_a'] == 1
        
        # Missing features in second doc
        assert df.loc[1, 'neutral'] == 1
        assert df.loc[1, 'venue'] == 'neutral'
        assert df.loc[1, 'home_elo'] == 1500.0
        assert df.loc[1, 'away_elo'] == 1500.0
        assert df.loc[1, 'elo_diff'] == 0.0
        assert df.loc[1, 'home_form5_pts'] == 1.0
        assert df.loc[1, 'away_form5_pts'] == 1.0
        assert df.loc[1, 'score_h'] == 0
        assert df.loc[1, 'score_a'] == 0
