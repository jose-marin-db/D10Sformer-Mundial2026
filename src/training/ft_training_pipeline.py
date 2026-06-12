import sys
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path('src')))

from data.vocabulary import FootballVocab
from models.ft_transformer import FTD10Sformer, FTD10SformerConfig
from data.ft_dataset import FTDataset, FTCollator

DATA_INTERIM = Path('interim')
CORPUS_DIR = Path('processed/corpus')

WINDOW = 5

def rmean(history, idx):
    recent = history[-WINDOW:]
    if not recent: return np.nan
    return sum(x[idx] for x in recent) / len(recent)

def compute_rolling_features(df_int):
    df_sorted = df_int.sort_values('date').reset_index(drop=True).copy()
    team_history = defaultdict(list)
    home_form, away_form, home_goals, away_goals, home_form_10, away_form_10 = [], [], [], [], [], []
    for _, row in df_sorted.iterrows():
        h, a = row.home_team, row.away_team
        hs, as_ = row.home_score, row.away_score
        home_form.append(rmean(team_history[h], 0))
        away_form.append(rmean(team_history[a], 0))
        home_goals.append(rmean(team_history[h], 1))
        away_goals.append(rmean(team_history[a], 1))
        home_form_10.append(rmean(team_history[h][-10:], 0))
        away_form_10.append(rmean(team_history[a][-10:], 0))
        if pd.notna(hs) and pd.notna(as_):
            if hs > as_:   h_pts, a_pts = 3, 0
            elif hs < as_: h_pts, a_pts = 0, 3
            else:          h_pts, a_pts = 1, 1
            team_history[h].append((h_pts, float(hs)))
            team_history[a].append((a_pts, float(as_)))
    df_sorted['home_form_pts_5']     = home_form
    df_sorted['away_form_pts_5']     = away_form
    df_sorted['home_recent_goals_5'] = home_goals
    df_sorted['away_recent_goals_5'] = away_goals
    df_sorted['home_form10_pts']     = home_form_10
    df_sorted['away_form10_pts']     = away_form_10
    return df_sorted

# Filter to match StatsBomb corpus splits
def extract_df_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        docs = pickle.load(f)
    records = []
    for m in docs:
        elo_diff = (m.features.home_elo - m.features.away_elo) if m.features else 0.0
        records.append({
            'tournament_class': m.tournament,
            'neutral': 1 if m.venue == 'neutral' else 0,
            'venue': m.venue,
            'home_elo': m.features.home_elo if m.features else 1500.0,
            'away_elo': m.features.away_elo if m.features else 1500.0,
            'elo_diff': elo_diff,
            'home_form5_pts': m.features.home_form_pts if m.features else 1.0,
            'away_form5_pts': m.features.away_form_pts if m.features else 1.0,
            'home_form5_gf': m.features.home_recent_goals if m.features else 1.0,
            'away_form5_gf': m.features.away_recent_goals if m.features else 1.0,
            'home_form10_pts': m.features.home_form_pts if m.features else 1.0,
            'away_form10_pts': m.features.away_form_pts if m.features else 1.0,
            'score_h': m.home_score if m.home_score is not None else 0,
            'score_a': m.away_score if m.away_score is not None else 0
        })
    return pd.DataFrame(records)

def train_pipeline(epochs=15, batch_size=64, save_path='checkpoints/ft_transformer_best.pt'):
    # 1. Feature matrix assembly
    print("Loading historical results and computing features...")
    df_int = pd.read_parquet(DATA_INTERIM / 'international_matches_with_elo.parquet')
    df_int['date'] = pd.to_datetime(df_int['date'])
    df_sorted = compute_rolling_features(df_int)

    print("Preparing train/val/test dataframes...")
    df_train = extract_df_from_pkl(CORPUS_DIR / 'finetune_train.pkl')
    df_val = extract_df_from_pkl(CORPUS_DIR / 'val.pkl')
    df_test = extract_df_from_pkl(CORPUS_DIR / 'test.pkl')

    vocab = FootballVocab.load('processed/vocab.json')
    tourn_map = {t: i for i, t in enumerate(sorted(df_train['tournament_class'].unique()))}

    ds_train = FTDataset(df_train, tourn_map)
    ds_val = FTDataset(df_val, tourn_map)
    ds_test = FTDataset(df_test, tourn_map)

    collator = FTCollator()
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=collator)

    # Compute inverse class weights for scores
    labels = [ds_train[i][2] for i in range(len(ds_train))]
    classes, counts = np.unique(labels, return_counts=True)
    weight_dict = {c: len(labels) / (36 * count) for c, count in zip(classes, counts)}
    class_weights = torch.ones(36, dtype=torch.float32)
    for c, w in weight_dict.items():
        class_weights[c] = w

    # 2. Build and train FT-Transformer
    config = FTD10SformerConfig(
        vocab_size=len(vocab),
        num_tournament_classes=len(tourn_map) + 1,
        d_model=128, num_layers=3, num_heads=4, d_ff=256, dropout=0.15
    )
    model = FTD10Sformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    print(f"\nStarting training loop ({epochs} epochs)...")
    best_loss = float('inf')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    class_weights = class_weights.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for cats, conts, labels in train_loader:
            cats, conts, labels = cats.to(device), conts.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(cats, conts)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * cats.size(0)
            
        # Evaluate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for cats, conts, labels in val_loader:
                cats, conts, labels = cats.to(device), conts.to(device), labels.to(device)
                logits = model(cats, conts)
                loss = F.cross_entropy(logits, labels)
                val_loss += loss.item() * cats.size(0)
                
        train_loss = total_loss / len(ds_train)
        val_loss = val_loss / len(ds_val)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("  ✓ Saved best model checkpoint")

    print("\nTraining Complete!")

if __name__ == '__main__':
    train_pipeline()
