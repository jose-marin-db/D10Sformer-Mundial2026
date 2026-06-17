import sys
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# Add src/ to python path
sys.path.insert(0, str(Path('src')))
from simulation.bracket import WC2026_GROUPS, SPANISH_TO_ENGLISH

def run_bulk_load():
    print("=== Inicializando Estado de Fuerza ELO original (Pre-Mundial) ===")
    
    # 1. Load historical data from interim (initial state as of June 11)
    DATA_INTERIM = Path('interim')
    df_int = pd.read_parquet(DATA_INTERIM / 'international_matches_with_elo.parquet')
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

    WC_TEAMS = sorted({t for grp in WC2026_GROUPS.values() for t in grp})
    def latest_state(team):
        rows = df_sorted[(df_sorted.home_team == team) | (df_sorted.away_team == team)]
        if len(rows) == 0: return None
        r = rows.iloc[-1]
        if r.home_team == team:
            return {'team': team, 'elo': float(r.home_elo_after) if pd.notna(r.home_elo_after) else float(r.home_elo_before),
                    'form_pts': r.home_form_pts_5, 'recent_goals': r.home_recent_goals_5, 'form10_pts': r.home_form10_pts}
        return {'team': team, 'elo': float(r.away_elo_after) if pd.notna(r.away_elo_after) else float(r.away_elo_before),
                'form_pts': r.away_form_pts_5, 'recent_goals': r.away_recent_goals_5, 'form10_pts': r.away_form10_pts}

    team_features = {}
    for t in WC_TEAMS:
        st = latest_state(t)
        if st is None:
            team_features[t] = {'team': t, 'elo': 1500.0, 'form_pts': 1.0, 'recent_goals': 1.0, 'form10_pts': 1.0}
        else:
            team_features[t] = st

    # 2. Translation map for web interface
    spanish_to_english_web = SPANISH_TO_ENGLISH.copy()
    spanish_to_english_web["Paises Bajos"] = "Netherlands"
    spanish_to_english_web["Países Bajos"] = "Netherlands"
    spanish_to_english_web["Tunez"] = "Tunisia"
    spanish_to_english_web["Túnez"] = "Tunisia"
    spanish_to_english_web["Iraq"] = "Iraq"
    spanish_to_english_web["Irak"] = "Iraq"
    spanish_to_english_web["República de Corea"] = "South Korea"
    spanish_to_english_web["Corea del Sur"] = "South Korea"

    # 3. Real World Results
    real_results_data = [
        {"home": "México", "away": "Sudáfrica", "home_score": 2, "away_score": 0},
        {"home": "Corea del Sur", "away": "República Checa", "home_score": 2, "away_score": 1},
        {"home": "Canadá", "away": "Bosnia y Herzegovina", "home_score": 1, "away_score": 1},
        {"home": "Estados Unidos", "away": "Paraguay", "home_score": 4, "away_score": 1},
        {"home": "Catar", "away": "Suiza", "home_score": 1, "away_score": 1},
        {"home": "Brasil", "away": "Marruecos", "home_score": 1, "away_score": 1},
        {"home": "Haití", "away": "Escocia", "home_score": 0, "away_score": 1},
        {"home": "Australia", "away": "Turquía", "home_score": 2, "away_score": 0},
        {"home": "Alemania", "away": "Curazao", "home_score": 7, "away_score": 1},
        {"home": "Países Bajos", "away": "Japón", "home_score": 2, "away_score": 2},
        {"home": "Costa de Marfil", "away": "Ecuador", "home_score": 1, "away_score": 0},
        {"home": "Suecia", "away": "Túnez", "home_score": 5, "away_score": 1},
        {"home": "España", "away": "Cabo Verde", "home_score": 0, "away_score": 0},
        {"home": "Bélgica", "away": "Egipto", "home_score": 1, "away_score": 1},
        {"home": "Arabia Saudita", "away": "Uruguay", "home_score": 1, "away_score": 1},
        {"home": "Irán", "away": "Nueva Zelanda", "home_score": 2, "away_score": 2},
        {"home": "Francia", "away": "Senegal", "home_score": 3, "away_score": 1},
        {"home": "Irak", "away": "Noruega", "home_score": 1, "away_score": 4},
        {"home": "Argentina", "away": "Argelia", "home_score": 3, "away_score": 0},
        {"home": "Austria", "away": "Jordania", "home_score": 3, "away_score": 1}
    ]

    loaded_matches = []

    print("\n=== Procesando partidos en lote y recalculando ELO ===")
    for m in real_results_data:
        h_spa = m["home"]
        a_spa = m["away"]
        gh, ga = m["home_score"], m["away_score"]
        
        h_eng = spanish_to_english_web[h_spa]
        a_eng = spanish_to_english_web[a_spa]
        
        fa = team_features[h_eng]
        fb = team_features[a_eng]
        
        elo_a = fa['elo']
        elo_b = fb['elo']
        
        # Official ELO formula from app
        dr = elo_a - elo_b
        expected_a = 1 / (1 + 10 ** (-dr / 400))
        expected_b = 1.0 - expected_a
        
        actual_a = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        actual_b = 1.0 - actual_a
        
        # Goal difference multiplier
        gd = abs(gh - ga)
        if gd <= 1:
            G = 1.0
        elif gd == 2:
            G = 1.5
        else:
            G = 1.75 + (gd - 3) / 8.0
            
        K = 40.0 # High multiplier for World Cup matches
        
        new_elo_a = elo_a + K * G * (actual_a - expected_a)
        new_elo_b = elo_b + K * G * (actual_b - expected_b)
        
        team_features[h_eng]['elo'] = new_elo_a
        team_features[a_eng]['elo'] = new_elo_b
        
        # History / form update
        if 'history' not in team_features[h_eng]:
            team_features[h_eng]['history'] = [(float(fa['form_pts']), float(fa['recent_goals']))] * 5
        if 'history' not in team_features[a_eng]:
            team_features[a_eng]['history'] = [(float(fb['form_pts']), float(fb['recent_goals']))] * 5
            
        pts_a = 3 if gh > ga else (1 if gh == ga else 0)
        pts_b = 3 if ga > gh else (1 if gh == ga else 0)
        
        team_features[h_eng]['history'].append((float(pts_a), float(gh)))
        team_features[a_eng]['history'].append((float(pts_b), float(ga)))
        
        # Recalculate averages
        team_features[h_eng]['form_pts'] = sum(x[0] for x in team_features[h_eng]['history'][-5:]) / 5.0
        team_features[h_eng]['recent_goals'] = sum(x[1] for x in team_features[h_eng]['history'][-5:]) / 5.0
        team_features[a_eng]['form_pts'] = sum(x[0] for x in team_features[a_eng]['history'][-5:]) / 5.0
        team_features[a_eng]['recent_goals'] = sum(x[1] for x in team_features[a_eng]['history'][-5:]) / 5.0
        
        # Record match log
        loaded_matches.append({
            "home": h_spa,
            "away": a_spa,
            "home_score": int(gh),
            "away_score": int(ga)
        })
        print(f"✔ Cargado: {h_spa} {gh} - {ga} {a_spa} | ELO {h_spa}: {int(elo_a)} ➜ {int(new_elo_a)} | ELO {a_spa}: {int(elo_b)} ➜ {int(new_elo_b)}")

    # 4. Save to live state file
    live_dir = Path('live')
    live_dir.mkdir(exist_ok=True)
    state_file = live_dir / 'streamlit_live_state.pkl'
    
    with open(state_file, 'wb') as f:
        pickle.dump({
            'team_features': team_features,
            'WC_TEAMS': WC_TEAMS,
            'loaded_matches': loaded_matches
        }, f)
        
    print(f"\n🎉 ¡Carga masiva completada con éxito! Archivo '{state_file}' guardado.")

if __name__ == '__main__':
    run_bulk_load()
