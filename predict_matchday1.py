import sys
import json
import pickle
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# Add src/ to python path
sys.path.insert(0, str(Path('src')))

from simulation.bracket import WC2026_GROUPS, assert_bracket_consistency
from data.feature_engineering import build_feature_matrix, get_feature_columns, prepare_xy
from models.baselines import LogisticRegressionBaseline

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
# 2. Train Logistic Regression
# -------------------------------------------------------------
print("Training Logistic Regression model...")
df_full = build_feature_matrix(
    df_sorted, windows_form=(5, 10), h2h_window=5, min_date='2014-01-01',
)
X_full, y_full = prepare_xy(df_full, onehot_categoricals=True)
FEATURE_COLS = list(X_full.columns)

logreg = LogisticRegressionBaseline()
logreg.fit(X_full, y_full)

# -------------------------------------------------------------
# 3. Define Predictor and Helper functions
# -------------------------------------------------------------
def build_feature_vector(team_a, team_b, venue='neutral'):
    fa = team_features.get(team_a, {'elo': 1500, 'form_pts': 1.0, 'recent_goals': 1.0})
    fb = team_features.get(team_b, {'elo': 1500, 'form_pts': 1.0, 'recent_goals': 1.0})
    elo_diff = fa['elo'] - fb['elo']
    raw = {
        'tournament_class': 'world_cup_final',
        'neutral': 1 if venue == 'neutral' else 0,
        'home_elo': fa['elo'], 'away_elo': fb['elo'], 'elo_diff': elo_diff,
        'expected_home_win_prob': 1 / (1 + 10 ** (-elo_diff / 400)),
        'home_rest_days': 7, 'away_rest_days': 7,
        'home_form5_pts': fa['form_pts'] if pd.notna(fa['form_pts']) else 1.0,
        'home_form5_gf':  fa['recent_goals'] if pd.notna(fa['recent_goals']) else 1.0,
        'home_form5_ga': 1.0, 'home_form5_gd': 0.0, 'home_form5_n': 5,
        'away_form5_pts': fb['form_pts'] if pd.notna(fb['form_pts']) else 1.0,
        'away_form5_gf':  fb['recent_goals'] if pd.notna(fb['recent_goals']) else 1.0,
        'away_form5_ga': 1.0, 'away_form5_gd': 0.0, 'away_form5_n': 5,
        'home_form10_pts': fa['form_pts'] if pd.notna(fa['form_pts']) else 1.0,
        'home_form10_gf':  fa['recent_goals'] if pd.notna(fa['recent_goals']) else 1.0,
        'home_form10_ga': 1.0, 'home_form10_gd': 0.0, 'home_form10_n': 10,
        'away_form10_pts': fb['form_pts'] if pd.notna(fb['form_pts']) else 1.0,
        'away_form10_gf':  fb['recent_goals'] if pd.notna(fb['recent_goals']) else 1.0,
        'away_form10_ga': 1.0, 'away_form10_gd': 0.0, 'away_form10_n': 10,
        'h2h_n_matches': 0, 'h2h_home_wins': 0, 'h2h_draws': 0,
        'h2h_away_wins': 0, 'h2h_avg_gd_for_home': 0.0,
    }
    df_row = pd.DataFrame([raw])
    return df_row

def logreg_predictor_raw(team_a, team_b, venue='neutral'):
    x = build_feature_vector(team_a, team_b, venue=venue)
    x = pd.get_dummies(x, columns=['tournament_class', 'neutral'], dtype=int)
    x = x.reindex(columns=FEATURE_COLS, fill_value=0.0)
    return logreg.predict_proba(x)[0]

# Expected scores based on calibrated expected goals
def get_expected_score(p_home, p_draw, p_away, base_total=2.5):
    p_h_eff = p_home + p_draw / 2
    p_a_eff = p_away + p_draw / 2
    rate_h = base_total * p_h_eff
    rate_a = base_total * p_a_eff
    
    # We round to the nearest integer to find the most probable scoreline
    goals_h = round(rate_h)
    goals_a = round(rate_a)
    
    # If the win probabilities are extremely close (diff < 0.25), we recommend a draw (e.g. 1 - 1)
    if abs(p_home - p_away) < 0.25:
        avg_g = round((rate_h + rate_a) / 2)
        # Avoid 0-0 unless total rate is very low, prefer 1-1 as standard draw
        if avg_g == 0:
            avg_g = 1
        return avg_g, avg_g
    
    # If the rounded scores are equal but the match is not balanced, we break the tie.
    probs = [p_home, p_draw, p_away]
    outcome = np.argmax(probs) # 0: home win, 1: draw, 2: away win
    
    if outcome == 0 and goals_h <= goals_a:
        goals_h = goals_a + 1
    elif outcome == 2 and goals_a <= goals_h:
        goals_a = goals_h + 1
        
    return goals_h, goals_a

# -------------------------------------------------------------
# 4. Matches to predict (Fase 1)
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

print("\n=== PREDICCIONES FASE 1 (Mundialitopeya) ===")
print(f"{'Grupo':<8} | {'Local (web)':<25} vs {'Visitante (web)':<25} | {'Probabilidades (H/D/A)':<25} | {'Marcador Recomendado':<12}")
print("-" * 110)

results = []
for grp, h_web, h_eng, a_web, a_eng in matches_fase1:
    probs = logreg_predictor_raw(h_eng, a_eng)
    p_home, p_draw, p_away = probs[0], probs[1], probs[2]
    
    g_h, g_a = get_expected_score(p_home, p_draw, p_away)
    
    prob_str = f"{p_home:.2f} / {p_draw:.2f} / {p_away:.2f}"
    score_str = f"{g_h} - {g_a}"
    
    print(f"{grp:<8} | {h_web:<25} vs {a_web:<25} | {prob_str:<25} | {score_str:<12}")
    results.append({
        "group": grp,
        "home_web": h_web,
        "away_web": a_web,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "score_h": g_h,
        "score_a": g_a
    })

# Save to json just in case
with open("fase1_predictions_output.json", "w") as f:
    json.dump(results, f, indent=2)
