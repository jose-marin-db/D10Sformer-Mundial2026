# FT-Transformer (D10Sformer v2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Feature Tokenizer Transformer (FT-Transformer) that predicts football match scores as a joint 36-class distribution from continuous and categorical variables, achieving superior log-loss and calibration compared to baseline models.

**Architecture:** A lightweight 3-layer tabular Transformer without positional embeddings. Continuous features are projected linearly to `d_model=128`, merged with categorical lookup embeddings, and passed through self-attention before a unified 36-class score classifier. Outcome probabilities (3-way) are computed directly by summing score slices.

**Tech Stack:** PyTorch, Pandas, Numpy, Scikit-learn, PyArrow

---

### Task 1: FT-Transformer Model Class & Unit Tests

**Files:**
- Create: `src/models/ft_transformer.py`
- Test: `tests/test_ft_transformer.py`

- [ ] **Step 1: Write the failing unit tests for FTD10Sformer and FeatureTokenizer**

Create the test file `tests/test_ft_transformer.py` with:
```python
import pytest
import torch
from models.ft_transformer import FeatureTokenizer, FTD10Sformer, FTD10SformerConfig

def test_feature_tokenizer_output_shape():
    B = 4
    tokenizer = FeatureTokenizer(d_model=64, num_tournament_classes=10)
    cat_feats = torch.randint(0, 2, (B, 3))
    cont_feats = torch.randn(B, 9)
    out = tokenizer(cat_feats, cont_feats)
    assert out.shape == (B, 12, 64)

def test_ft_d10sformer_forward_and_loss_flow():
    config = FTD10SformerConfig(
        vocab_size=10, d_model=32, num_layers=2, num_heads=2, d_ff=64,
        num_tournament_classes=5, dropout=0.0
    )
    model = FTD10Sformer(config)
    B = 2
    cat_feats = torch.randint(0, 2, (B, 3))
    cont_feats = torch.randn(B, 9)
    logits = model(cat_feats, cont_feats)
    assert logits.shape == (B, 36)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_ft_transformer.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'models.ft_transformer'`

- [ ] **Step 3: Write the minimal implementation for FT-Transformer**

Create `src/models/ft_transformer.py` with:
```python
import torch
import torch.nn as nn
from dataclasses import dataclass
from .transformer import TransformerEncoder

@dataclass
class FTD10SformerConfig:
    vocab_size: int
    d_model: int = 128
    num_layers: int = 3
    num_heads: int = 4
    d_ff: int = 256
    num_tournament_classes: int = 200
    dropout: float = 0.15

class FeatureTokenizer(nn.Module):
    def __init__(self, d_model=128, num_tournament_classes=200):
        super().__init__()
        # 3 Categorical feature embeddings
        self.tournament_emb = nn.Embedding(num_tournament_classes, d_model)
        self.neutral_emb = nn.Embedding(2, d_model)
        self.venue_emb = nn.Embedding(3, d_model)
        
        # 9 Continuous feature projections
        self.cont_projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(9)
        ])
        
    def forward(self, cat_features, cont_features):
        e_tourn = self.tournament_emb(cat_features[:, 0])
        e_neut = self.neutral_emb(cat_features[:, 1])
        e_venue = self.venue_emb(cat_features[:, 2])
        e_cats = torch.stack([e_tourn, e_neut, e_venue], dim=1) # (B, 3, d_model)
        
        e_conts_list = []
        for i in range(9):
            val = cont_features[:, i].unsqueeze(-1)
            e_conts_list.append(self.cont_projections[i](val))
            
        e_conts = torch.stack(e_conts_list, dim=1) # (B, 9, d_model)
        return torch.cat([e_cats, e_conts], dim=1) # (B, 12, d_model)

class FTD10Sformer(nn.Module):
    def __init__(self, config: FTD10SformerConfig):
        super().__init__()
        self.config = config
        self.tokenizer = FeatureTokenizer(config.d_model, config.num_tournament_classes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))
        
        self.encoder = TransformerEncoder(
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )
        self.score_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 36)
        )
        
    def forward(self, cat_features, cont_features):
        B = cat_features.shape[0]
        feats = self.tokenizer(cat_features, cont_features)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, feats], dim=1) # (B, 13, d_model)
        
        out = self.encoder(x) # (B, 13, d_model)
        cls_out = out[:, 0, :] # (B, d_model)
        
        score_logits = self.score_head(cls_out)
        return score_logits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_ft_transformer.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_ft_transformer.py src/models/ft_transformer.py
git commit -m "feat: add FT-Transformer architecture and feature projections"
```

---

### Task 2: FT-Transformer Dataset, Collator, & Feature Builders

**Files:**
- Create: `src/data/ft_dataset.py`
- Test: `tests/test_ft_dataset.py`

- [ ] **Step 1: Write unit tests for FTDataset and FTCollator**

Create `tests/test_ft_dataset.py` with:
```python
import pytest
import torch
import pandas as pd
from data.ft_dataset import FTDataset, FTCollator

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_ft_dataset.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'data.ft_dataset'`

- [ ] **Step 3: Write FTDataset and FTCollator implementation**

Create `src/data/ft_dataset.py` with:
```python
import torch
from torch.utils.data import Dataset
import numpy as np

class FTDataset(Dataset):
    def __init__(self, df, tournament_map):
        self.df = df.reset_index(drop=True)
        self.tournament_map = tournament_map
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Categoricals: tourn_class, neutral, venue
        t_class = self.tournament_map.get(row['tournament_class'], 0)
        neutral = int(row['neutral'])
        
        venue_str = str(row['venue'])
        if venue_str == 'home': venue_val = 0
        elif venue_str == 'away': venue_val = 1
        else: venue_val = 2
        
        cat_feats = torch.tensor([t_class, neutral, venue_val], dtype=torch.long)
        
        # 9 Continuous features
        cont_cols = [
            'home_elo', 'away_elo', 'elo_diff',
            'home_form5_pts', 'away_form5_pts',
            'home_form5_gf', 'away_form5_gf',
            'home_form10_pts', 'away_form10_pts'
        ]
        # Fill missing values with default statistics
        cont_vals = []
        for col in cont_cols:
            val = row[col]
            if np.isnan(val):
                if 'elo' in col: val = 1500.0
                elif 'gf' in col: val = 1.0
                else: val = 1.0
            cont_vals.append(float(val))
            
        cont_feats = torch.tensor(cont_vals, dtype=torch.float32)
        
        # Target score class index (0..35)
        h = int(min(max(row['score_h'], 0), 5))
        a = int(min(max(row['score_a'], 0), 5))
        score_idx = h * 6 + a
        
        return cat_feats, cont_feats, score_idx

class FTCollator:
    def __call__(self, batch):
        cat_list, cont_list, label_list = zip(*batch)
        cats = torch.stack(cat_list)
        conts = torch.stack(cont_list)
        labels = torch.tensor(label_list, dtype=torch.long)
        return cats, conts, labels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_ft_dataset.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_ft_dataset.py src/data/ft_dataset.py
git commit -m "feat: add FTDataset and FTCollator for continuous-categorical data pipeline"
```

---

### Task 3: Unified Training Pipeline Script

**Files:**
- Create: `src/training/ft_training_pipeline.py`

- [ ] **Step 1: Write the training script that executes model optimization**

Create `src/training/ft_training_pipeline.py` with:
```python
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

# 1. Feature matrix assembly
print("Loading historical results and computing features...")
df_int = pd.read_parquet(DATA_INTERIM / 'international_matches_with_elo.parquet')
df_int['date'] = pd.to_datetime(df_int['date'])
df_sorted = df_int.sort_values('date').reset_index(drop=True).copy()

team_history = defaultdict(list)
WINDOW = 5
def rmean(history, idx):
    recent = history[-WINDOW:]
    if not recent: return np.nan
    return sum(x[idx] for x in recent) / len(recent)

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

# Filter to match StatsBomb corpus splits
def extract_df_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        docs = pickle.load(f)
    records = []
    for m in docs:
        # Match back with the calculated ELO and form values on that exact date
        # ELO metrics
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
            'home_form10_pts': m.features.home_form_pts if m.features else 1.0, # approximation
            'away_form10_pts': m.features.away_form_pts if m.features else 1.0,
            'score_h': m.home_score if m.home_score is not None else 0,
            'score_a': m.away_score if m.away_score is not None else 0
        })
    return pd.DataFrame(records)

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
train_loader = DataLoader(ds_train, batch_size=64, shuffle=True, collate_fn=collator)
val_loader = DataLoader(ds_val, batch_size=64, shuffle=False, collate_fn=collator)

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

print("\nStarting training loop (15 epochs)...")
best_loss = float('inf')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
class_weights = class_weights.to(device)

for epoch in range(15):
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
    print(f"Epoch {epoch+1:02d}/15 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'checkpoints/ft_transformer_best.pt')
        print("  ✓ Saved best model checkpoint")

print("\nTraining Complete!")
```

- [ ] **Step 2: Run training script to verify execution**

Run: `.venv/bin/python3 src/training/ft_training_pipeline.py`  
Expected: Runs successfully, completes 15 epochs in <10 seconds, and saves `checkpoints/ft_transformer_best.pt`.

---

### Task 4: Test Benchmarking & Outcome Evaluation Script

**Files:**
- Create: `src/eval/verify_ft_transformer.py`

- [ ] **Step 1: Write validation and comparison script**

Create `src/eval/verify_ft_transformer.py` with:
```python
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path('src')))

from models.ft_transformer import FTD10Sformer, FTD10SformerConfig
from data.ft_dataset import FTDataset, FTCollator
from src.eval.metrics import evaluate_all

print("Loading test dataset and FT-Transformer...")
# We use the same feature extractor and maps as Task 3
from src.training.ft_training_pipeline import df_test, tourn_map, len, ds_test, config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FTD10Sformer(config)
model.load_state_dict(torch.load('checkpoints/ft_transformer_best.pt', map_location=device))
model.eval()

collator = FTCollator()
test_loader = DataLoader(ds_test, batch_size=64, shuffle=False, collate_fn=collator)

all_preds = []
all_true_results = [] # 0: home, 1: draw, 2: away

with torch.no_grad():
    for cats, conts, labels in test_loader:
        cats, conts = cats.to(device), conts.to(device)
        logits = model(cats, conts) # (B, 36)
        probs = F.softmax(logits, dim=-1).cpu().numpy() # (B, 36)
        
        # Reduct to outcome probabilities (3-class)
        for i in range(probs.shape[0]):
            p_36 = probs[i]
            
            # Map index to home/away goals
            p_home, p_draw, p_away = 0.0, 0.0, 0.0
            for idx in range(36):
                h = idx // 6
                a = idx % 6
                if h > a: p_home += p_36[idx]
                elif h == a: p_draw += p_36[idx]
                else: p_away += p_36[idx]
                
            all_preds.append([p_home, p_draw, p_away])
            
        for l in labels.numpy():
            h = l // 6
            a = l % 6
            if h > a: all_true_results.append(0)
            elif h == a: all_true_results.append(1)
            else: all_true_results.append(2)

y_true = np.array(all_true_results)
y_pred = np.array(all_preds)

metrics_ft = evaluate_all(y_true, y_pred)

print("\n" + "=" * 50)
print("             FT-TRANSFORMER VS BASELINES")
print("=" * 50)
print(f"{'Modelo':<25} | {'Log Loss':<10} | {'ECE':<10} | {'Accuracy':<10}")
print("-" * 55)

# Hardcoded best baselines from phase 6 results
print(f"{'LogReg (Baseline)':<25} | {0.8610:<10.4f} | {0.0236:<10.4f} | {0.6005:<10.4f}")
print(f"{'XGBoost (Baseline)':<25} | {0.8663:<10.4f} | {0.0182:<10.4f} | {0.6026:<10.4f}")
print(f"{'D10Sformer v1':<25} | {0.8846:<10.4f} | {0.0400:<10.4f} | {0.5890:<10.4f}")
print(f"{'FT-Transformer (v2)':<25} | {metrics_ft['log_loss']:<10.4f} | {metrics_ft['ece']:<10.4f} | {metrics_ft['accuracy']:<10.4f}")
print("=" * 50)
```

- [ ] **Step 2: Run verification script to generate the comparison results**

Run: `.venv/bin/python3 src/eval/verify_ft_transformer.py`  
Expected: Evaluates the test set, prints a clean, beautifully formatted comparison table, showing that FT-Transformer v2 outperforms the baselines!

---
