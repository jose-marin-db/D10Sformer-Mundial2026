import sys
import json
import pickle
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Add src/ to python path
sys.path.insert(0, str(Path('src')))

from data.vocabulary import FootballVocab
from data.tokenizer import MatchTokenizer, MatchDocument, RollingFeatures
from models.d10sformer import D10Sformer, D10SformerConfig

DATA_INTERIM = Path('interim')

# -------------------------------------------------------------
# 1. Load data & compute team rolling features
# -------------------------------------------------------------
print("Loading data...")
df_int = pd.read_parquet(DATA_INTERIM / 'international_matches_with_elo.parquet')
df_int['date'] = pd.to_datetime(df_int['date'])
df_sorted = df_int.sort_values('date').reset_index(drop=True).copy()

team_history = defaultdict(list)
WINDOW = 5
def rmean(history, idx):
    recent = history[-WINDOW:]
    if not recent: return math.nan
    return sum(x[idx] for x in recent) / len(recent)

home_form, away_form, home_goals, away_goals = [], [], [], []
for _, row in df_sorted.iterrows():
    h, a = row.home_team, row.away_team
    hs, as_ = row.home_score, row.away_score
    home_form.append(rmean(team_history[h], 0))
    away_form.append(rmean(team_history[a], 0))
    home_goals.append(rmean(team_history[h], 1))
    away_goals.append(rmean(team_history[a], 1))
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

# Last state per team
from simulation.bracket import WC2026_GROUPS
WC_TEAMS = sorted({t for grp in WC2026_GROUPS.values() for t in grp})
def latest_state(team):
    rows = df_sorted[(df_sorted.home_team == team) | (df_sorted.away_team == team)]
    if len(rows) == 0: return None
    r = rows.iloc[-1]
    if r.home_team == team:
        return {'team': team, 'elo': float(r.home_elo_after) if pd.notna(r.home_elo_after) else float(r.home_elo_before),
                'form_pts': r.home_form_pts_5, 'recent_goals': r.home_recent_goals_5}
    return {'team': team, 'elo': float(r.away_elo_after) if pd.notna(r.away_elo_after) else float(r.away_elo_before),
            'form_pts': r.away_form_pts_5, 'recent_goals': r.away_recent_goals_5}

team_features = {}
for t in WC_TEAMS:
    st = latest_state(t)
    if st is None:
        team_features[t] = {'team': t, 'elo': 1500.0, 'form_pts': 1.0, 'recent_goals': 1.0}
    else:
        team_features[t] = st

# -------------------------------------------------------------
# 2. Load trained D10Sformer model
# -------------------------------------------------------------
print("Loading trained D10Sformer model...")
vocab = FootballVocab.load('processed/vocab.json')
tokenizer = MatchTokenizer(vocab, max_seq_length=80)

model_config = D10SformerConfig(
    vocab_size=len(vocab),
    d_model=256, num_layers=6, num_heads=8, d_ff=1024,
    max_seq_length=80, num_segments=8,
    dropout=0.1, attention_dropout=0.1,
    pad_token_id=vocab.encode('[PAD]'),
    tie_mlm_weights=True,
)
model = D10Sformer(model_config)
ckpt = torch.load('checkpoints/finetune_weighted_15ep/best.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Local mapping helpers for D10Sformer outputs
# Result head: 0 -> RESULT_HOME_WIN, 1 -> RESULT_DRAW, 2 -> RESULT_AWAY_WIN
# Score head: index i corresponds to SCORE_h_a where h = i // 6, a = i % 6

# -------------------------------------------------------------
# 3. Matches to predict (Fase 1)
# -------------------------------------------------------------
matches_fase1 = [
    ("Group_A", "México", "Mexico", "Sudáfrica", "South Africa"),
    ("Group_A", "República de Corea", "South Korea", "República Checa", "Czech Republic"),
    ("Group_B", "Canadá", "Canada", "Bosnia y Herzegovina", "Bosnia and Herzegovina"),
    ("Group_D", "Estados Unidos", "United States", "Paraguay", "Paraguay"),
    ("Group_B", "Qatar", "Qatar", "Suiza", "Switzerland"),
    ("Group_C", "Brasil", "Brazil", "Marruecos", "Morocco"),
    ("Group_C", "Haití", "Haiti", "Escocia", "Scotland"),
    ("Group_D", "Australia", "Australia", "Turquía", "Turkey"),
    ("Group_E", "Alemania", "Germany", "Curazao", "Curaçao"),
    ("Group_F", "Paises Bajos", "Netherlands", "Japón", "Japan"),
    ("Group_E", "Costa de Marfil", "Ivory Coast", "Ecuador", "Ecuador"),
    ("Group_F", "Suecia", "Sweden", "Tunez", "Tunisia"),
    ("Group_H", "España", "Spain", "Cabo Verde", "Cape Verde"),
    ("Group_G", "Bélgica", "Belgium", "Egipto", "Egypt"),
    ("Group_H", "Arabia Saudita", "Saudi Arabia", "Uruguay", "Uruguay"),
    ("Group_G", "Irán", "Iran", "Nueva Zelanda", "New Zealand"),
    ("Group_I", "Francia", "France", "Senegal", "Senegal"),
    ("Group_I", "Iraq", "Iraq", "Noruega", "Norway"),
    ("Group_J", "Argentina", "Argentina", "Argelia", "Algeria"),
    ("Group_J", "Austria", "Austria", "Jordania", "Jordan"),
    ("Group_K", "Portugal", "Portugal", "República Democrática del Congo", "DR Congo"),
    ("Group_L", "Inglaterra", "England", "Croacia", "Croatia"),
    ("Group_L", "Ghana", "Ghana", "Panamá", "Panama"),
    ("Group_K", "Uzbekistán", "Uzbekistan", "Colombia", "Colombia"),
]

print("\n=== PREDICCIONES FASE 1 CON EL D10Sformer ===")
print(f"{'Grupo':<8} | {'Local (web)':<25} vs {'Visitante (web)':<25} | {'Probabilidades (H/D/A)':<25} | {'D10Sformer Score':<15}")
print("-" * 115)

results = []
for grp, h_web, h_eng, a_web, a_eng in matches_fase1:
    fa = team_features.get(h_eng, {'elo': 1500, 'form_pts': 1.0, 'recent_goals': 1.0})
    fb = team_features.get(a_eng, {'elo': 1500, 'form_pts': 1.0, 'recent_goals': 1.0})
    
    doc = MatchDocument(
        tournament="FIFA World Cup",
        team_a=h_eng,
        team_b=a_eng,
        venue="neutral",
        stage=None,
        features=RollingFeatures(
            home_elo=fa['elo'],
            away_elo=fb['elo'],
            home_form_pts=fa['form_pts'],
            away_form_pts=fb['form_pts'],
            home_recent_goals=fa['recent_goals'],
            away_recent_goals=fb['recent_goals']
        )
    )
    
    out = tokenizer.tokenize(doc)
    
    tok_tensor = torch.tensor([out.token_ids], dtype=torch.long)
    seg_tensor = torch.tensor([out.segment_ids], dtype=torch.long)
    
    with torch.no_grad():
        out_model = model(tok_tensor, seg_tensor)
        
    res_probs = F.softmax(out_model["result_logits"], dim=-1)[0].numpy()
    score_probs = F.softmax(out_model["score_logits"], dim=-1)[0].numpy()
    
    p_home, p_draw, p_away = res_probs[0], res_probs[1], res_probs[2]
    
    # Most probable score index
    best_score_idx = np.argmax(score_probs)
    g_h = best_score_idx // 6
    g_a = best_score_idx % 6
    p_score = score_probs[best_score_idx]
    
    prob_str = f"{p_home:.2f} / {p_draw:.2f} / {p_away:.2f}"
    score_str = f"{g_h} - {g_a} (P={p_score:.2f})"
    
    print(f"{grp:<8} | {h_web:<25} vs {a_web:<25} | {prob_str:<25} | {score_str:<15}")
    results.append({
        "group": grp,
        "home_web": h_web,
        "away_web": a_web,
        "p_home": float(p_home),
        "p_draw": float(p_draw),
        "p_away": float(p_away),
        "score_h": int(g_h),
        "score_a": int(g_a),
        "p_score": float(p_score)
    })

# Save to json
with open("d10sformer_fase1_predictions_output.json", "w") as f:
    json.dump(results, f, indent=2)
