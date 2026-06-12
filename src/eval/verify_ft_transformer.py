import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

# Insert src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.ft_transformer import FTD10Sformer, FTD10SformerConfig
from data.ft_dataset import FTDataset, FTCollator
from data.vocabulary import FootballVocab
from eval.metrics import evaluate_all
from training.ft_training_pipeline import extract_df_from_pkl, CORPUS_DIR

print("Loading test dataset and FT-Transformer...")

# Reconstruct tourn_map and ds_test as in ft_training_pipeline.py
df_train = extract_df_from_pkl(CORPUS_DIR / 'finetune_train.pkl')
df_test = extract_df_from_pkl(CORPUS_DIR / 'test.pkl')

vocab = FootballVocab.load('processed/vocab.json')
tourn_map = {t: i for i, t in enumerate(sorted(df_train['tournament_class'].unique()))}
ds_test = FTDataset(df_test, tourn_map)

# Config as in ft_training_pipeline.py
config = FTD10SformerConfig(
    vocab_size=len(vocab),
    num_tournament_classes=len(tourn_map) + 1,
    d_model=128, num_layers=3, num_heads=4, d_ff=256, dropout=0.15
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FTD10Sformer(config)
model.load_state_dict(torch.load('checkpoints/ft_transformer_best.pt', map_location=device))
model.to(device)
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
        
        # Reduce to outcome probabilities (3-class)
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
