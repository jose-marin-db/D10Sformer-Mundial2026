import sys
from pathlib import Path

import pytest
import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.ft_dataset import FTDataset, FTCollator  # noqa: E402


def test_ft_dataset_item_format():
    df = pd.DataFrame({
        'tournament_class': ['friendly', 'world_cup'],
        'neutral': [1, 0],
        'venue': ['neutral', 'home'],
        'home_elo': [1600.0, 1800.0],
        'away_elo': [1500.0, 1750.0],
        'elo_diff': [100.0, 50.0],
        'home_form5_pts': [1.5, 2.2],
        'away_form5_pts': [1.2, 1.8],
        'home_form5_gf': [1.4, 2.0],
        'away_form5_gf': [1.0, 1.6],
        'home_form10_pts': [1.5, 2.1],
        'away_form10_pts': [1.3, 1.9],
        'score_h': [2, 1],
        'score_a': [1, 1]
    })
    
    tourn_map = {'friendly': 0, 'world_cup': 1}
    ds = FTDataset(df, tourn_map)
    assert len(ds) == 2
    
    cat, cont, label = ds[0]
    assert cat.shape == (3,)
    assert cont.shape == (9,)
    assert label == 2 * 6 + 1 # SCORE_2_1 is index 13


def test_ft_collator():
    df = pd.DataFrame({
        'tournament_class': ['friendly', 'world_cup'],
        'neutral': [1, 0],
        'venue': ['neutral', 'home'],
        'home_elo': [1600.0, 1800.0],
        'away_elo': [1500.0, 1750.0],
        'elo_diff': [100.0, 50.0],
        'home_form5_pts': [1.5, 2.2],
        'away_form5_pts': [1.2, 1.8],
        'home_form5_gf': [1.4, 2.0],
        'away_form5_gf': [1.0, 1.6],
        'home_form10_pts': [1.5, 2.1],
        'away_form10_pts': [1.3, 1.9],
        'score_h': [2, 1],
        'score_a': [1, 1]
    })
    
    tourn_map = {'friendly': 0, 'world_cup': 1}
    ds = FTDataset(df, tourn_map)
    collator = FTCollator()
    
    cats, conts, labels = collator([ds[0], ds[1]])
    
    assert cats.shape == (2, 3)
    assert conts.shape == (2, 9)
    assert labels.shape == (2,)
    assert labels[0].item() == 2 * 6 + 1
    assert labels[1].item() == 1 * 6 + 1


def test_ft_dataset_nan_handling():
    # Test handling of NaNs in continuous columns
    df = pd.DataFrame({
        'tournament_class': ['friendly'],
        'neutral': [1],
        'venue': ['neutral'],
        'home_elo': [float('nan')],
        'away_elo': [float('nan')],
        'elo_diff': [float('nan')],
        'home_form5_pts': [float('nan')],
        'away_form5_pts': [float('nan')],
        'home_form5_gf': [float('nan')],
        'away_form5_gf': [float('nan')],
        'home_form10_pts': [float('nan')],
        'away_form10_pts': [float('nan')],
        'score_h': [2],
        'score_a': [1]
    })
    
    tourn_map = {'friendly': 0}
    ds = FTDataset(df, tourn_map)
    cat, cont, label = ds[0]
    
    # Assert NaN values are filled with expected defaults:
    # 'home_elo', 'away_elo', 'elo_diff' -> 1500.0
    # 'home_form5_gf', 'away_form5_gf' -> 1.0 (since gf is in the name)
    # others -> 1.0
    assert cont[0].item() == 1500.0  # home_elo
    assert cont[1].item() == 1500.0  # away_elo
    assert cont[2].item() == 1500.0  # elo_diff
    assert cont[3].item() == 1.0     # home_form5_pts
    assert cont[4].item() == 1.0     # away_form5_pts
    assert cont[5].item() == 1.0     # home_form5_gf
    assert cont[6].item() == 1.0     # away_form5_gf
    assert cont[7].item() == 1.0     # home_form10_pts
    assert cont[8].item() == 1.0     # away_form10_pts
