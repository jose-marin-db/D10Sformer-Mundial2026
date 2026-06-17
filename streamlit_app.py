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
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add src/ to python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data.vocabulary import FootballVocab
from models.ft_transformer import FTD10Sformer, FTD10SformerConfig
from data.ft_dataset import FTDataset
from simulation.bracket import WC2026_GROUPS, SPANISH_TO_ENGLISH

st.set_page_config(
    page_title="D10Sformer v2 — Dashboard Mundial 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling (Vercel/Stripe Minimalist Standard)
st.markdown("""
<style>
    /* Global Page Font & Reset */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }
    
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #0F172A; /* Slate 900 */
        text-align: center;
        margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B; /* Slate 500 */
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Premium Minimalist Card Component */
    .metric-card {
        background-color: #F8FAFC; /* Slate 50 */
        border: 1px solid #E2E8F0; /* Slate 200 */
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02);
        text-align: center;
        transition: all 0.2s ease-in-out;
    }
    .metric-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #10B981; /* Emerald 500 */
        letter-spacing: -1px;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748B; /* Slate 500 */
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.2rem;
    }
    
    /* Flat Modern Style Overrides for Native Streamlit Widgets */
    .stTabs [data-basetabs="tablist"] {
        justify-content: center;
        border-bottom: 1px solid #E2E8F0 !important;
        gap: 1rem !important;
    }
    .stTabs [data-basetabs="tab"] {
        font-weight: 700 !important;
        color: #64748B !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s !important;
    }
    .stTabs [data-basetabs="tab"][aria-selected="true"] {
        color: #2563EB !important; /* Royal Blue */
        border-bottom: 2px solid #2563EB !important;
    }
    
    div[data-testid="stExpander"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02) !important;
        margin-bottom: 1rem !important;
    }
    
    /* Sleek flat buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
        transform: translateY(-0.5px) !important;
    }
    .stButton > button[kind="secondary"] {
        background-color: #F8FAFC !important;
        color: #1E293B !important;
        border: 1px solid #E2E8F0 !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #F1F5F9 !important;
        border-color: #CBD5E1 !important;
    }

    /* Mobile-first and responsive typography */
    @media (max-width: 768px) {
        .main-title {
            font-size: 1.4rem !important;
            letter-spacing: -0.3px !important;
        }
        .subtitle {
            font-size: 0.82rem !important;
            margin-bottom: 1.0rem !important;
        }
        .metric-value {
            font-size: 1.8rem !important;
        }
        .metric-card {
            padding: 0.6rem !important;
        }
        /* Scale down native headers and texts on mobile */
        h1 {
            font-size: 1.2rem !important;
            font-weight: 800 !important;
        }
        h2 {
            font-size: 1.05rem !important;
            font-weight: 700 !important;
        }
        h3 {
            font-size: 0.9rem !important;
            font-weight: 700 !important;
        }
        p, li, span, label, div[data-testid="stExpander"], div[data-testid="stMarkdownContainer"] {
            font-size: 0.82rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 1. Caching Data and Models
# -------------------------------------------------------------
@st.cache_resource
def load_transformer_model():
    # Construct maps exactly as training pipeline
    CORPUS_DIR = Path('processed/corpus')
    with open(CORPUS_DIR / 'finetune_train.pkl', 'rb') as f:
        docs = pickle.load(f)
    tournaments = sorted(list(set(m.tournament for m in docs)))
    tourn_map = {t: i for i, t in enumerate(tournaments)}
    
    vocab = FootballVocab.load('processed/vocab.json')
    
    config = FTD10SformerConfig(
        vocab_size=len(vocab),
        num_tournament_classes=len(tourn_map) + 1,
        d_model=128, num_layers=3, num_heads=4, d_ff=256, dropout=0.15
    )
    model = FTD10Sformer(config)
    model.load_state_dict(torch.load('checkpoints/ft_transformer_best.pt', map_location='cpu'))
    model.eval()
    
    return model, vocab, tourn_map

@st.cache_resource
def load_d10sformer_v1(vocab):
    from models.d10sformer import D10Sformer, D10SformerConfig
    config = D10SformerConfig(
        vocab_size=len(vocab),
        d_model=256, num_layers=6, num_heads=8, d_ff=1024,
        max_seq_length=80, num_segments=8,
        dropout=0.1, attention_dropout=0.1,
        pad_token_id=vocab.encode('[PAD]'),
        tie_mlm_weights=True,
    )
    model_v1 = D10Sformer(config)
    ckpt = torch.load('checkpoints/finetune_weighted_15ep/best.pt', map_location='cpu')
    model_v1.load_state_dict(ckpt['model_state_dict'])
    model_v1.eval()
    return model_v1

@st.cache_data
def load_historical_data_and_features():
    # Check if live state exists
    live_state_path = Path('live/streamlit_live_state.pkl')
    if live_state_path.exists():
        with open(live_state_path, 'rb') as f:
            state = pickle.load(f)
        loaded_matches = state.get('loaded_matches', [])
        return state['team_features'], state['WC_TEAMS'], loaded_matches

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

    # Get latest stats for all World Cup teams
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
            
    return team_features, WC_TEAMS, []

# Load cached items
try:
    model, vocab, tourn_map = load_transformer_model()
    model_v1 = load_d10sformer_v1(vocab)
    team_features, WC_TEAMS, loaded_matches = load_historical_data_and_features()
except Exception as e:
    st.error(f"Error loading models or dataset. Make sure you trained the model first: {e}")
    st.stop()

# Global Spanish to English web names mapping (KeyError Hotfix)
spanish_to_english_web = SPANISH_TO_ENGLISH.copy()
spanish_to_english_web["Paises Bajos"] = "Netherlands"
spanish_to_english_web["Países Bajos"] = "Netherlands"
spanish_to_english_web["Tunez"] = "Tunisia"
spanish_to_english_web["Túnez"] = "Tunisia"
spanish_to_english_web["Iraq"] = "Iraq"
spanish_to_english_web["República de Corea"] = "South Korea"
spanish_to_english_web["Corea del Sur"] = "South Korea"

web_spanish_names = sorted(list(spanish_to_english_web.keys()))

def p_to_score(p_home, p_draw, p_away, elo_a, elo_b, goals_a, goals_b):
    # Calculate a dynamic match goals rate based on offensive form and ELO discrepancy
    base_rate = 2.6
    combined_history = float(goals_a) + float(goals_b)
    elo_diff = abs(float(elo_a) - float(elo_b))
    
    # Gap of 400 ELO adds about 0.4 expected goals
    elo_adjustment = (elo_diff / 400.0) * 0.4
    
    base_total = (combined_history * 0.7) + (base_rate * 0.3) + elo_adjustment
    base_total = max(1.8, min(base_total, 4.5))
    
    p_h_eff = p_home + p_draw / 2
    p_a_eff = p_away + p_draw / 2
    
    rate_h = base_total * p_h_eff
    rate_a = base_total * p_a_eff
    
    # Highly balanced matches (win probability diff < 0.25) round to draws
    if abs(p_home - p_away) < 0.25:
        avg_g = round((rate_h + rate_a) / 2)
        if avg_g == 0:
            avg_g = 1
        return avg_g, avg_g
        
    goals_h = round(rate_h)
    goals_a_out = round(rate_a)
    
    # Enforce favorite has more goals
    if p_home > p_away and goals_h <= goals_a_out:
        goals_h = goals_a_out + 1
    elif p_away > p_home and goals_a_out <= goals_h:
        goals_a_out = goals_h + 1
        
    # Inject variance for major favorites
    if p_home > 0.70 and goals_h < 2:
        goals_h = 2
    if p_home > 0.85 and goals_h < 3:
        goals_h = 3
        
    if p_away > 0.70 and goals_a_out < 2:
        goals_a_out = 2
    if p_away > 0.85 and goals_a_out < 3:
        goals_a_out = 3
        
    return goals_h, goals_a_out

def get_available_matches_to_load(team_features, loaded_matches):
    eng_to_spanish = {v: k for k, v in SPANISH_TO_ENGLISH.items()}
    eng_to_spanish["Netherlands"] = "Países Bajos"
    eng_to_spanish["Tunisia"] = "Túnez"
    eng_to_spanish["South Korea"] = "República de Corea"
    eng_to_spanish["Czech Republic"] = "República Checa"
    eng_to_spanish["South Africa"] = "Sudáfrica"
    
    # Track played matches: frozenset({home, away}) -> m
    played_group_map = {}
    played_ko_map = {}
    for m in loaded_matches:
        if "match_id" in m:
            played_ko_map[m["match_id"]] = m
        else:
            played_group_map[frozenset({m["home"], m["away"]})] = m
            
    unplayed = []
    played = []
    
    # 1. Chronological Group Stage Generation (Fase 1, then Fase 2, then Fase 3)
    for fase in [1, 2, 3]:
        for grp, teams in WC2026_GROUPS.items():
            if fase == 1:
                pairs = [(teams[0], teams[1]), (teams[2], teams[3])]
            elif fase == 2:
                pairs = [(teams[0], teams[2]), (teams[1], teams[3])]
            else:
                pairs = [(teams[0], teams[3]), (teams[1], teams[2])]
                
            for a_spa, b_spa in pairs:
                key = frozenset({a_spa, b_spa})
                if key in played_group_map:
                    m = played_group_map[key]
                    played.append({
                        "id": f"group_{grp}_{a_spa}_{b_spa}",
                        "type": "group",
                        "group": grp,
                        "home": a_spa,
                        "away": b_spa,
                        "home_score": m["home_score"],
                        "away_score": m["away_score"],
                        "label": f"✅ Grupo {grp[-1]} (Fecha {fase}): {a_spa} {m['home_score']} - {m['away_score']} {b_spa} [Cargado]"
                    })
                else:
                    unplayed.append({
                        "id": f"group_{grp}_{a_spa}_{b_spa}",
                        "type": "group",
                        "group": grp,
                        "home": a_spa,
                        "away": b_spa,
                        "label": f"🏆 Grupo {grp[-1]} (Fecha {fase}): {a_spa} vs {b_spa}"
                    })
                    
    # If any group matches remain unplayed, return unplayed + played group matches
    if len(unplayed) > 0:
        return unplayed + played
        
    # 2. All 72 group stage matches are loaded! Resolve Standings dynamically!
    from simulation.simulator import GroupStanding, select_best_thirds, assign_thirds_to_slots, resolve_slot
    from simulation.bracket import ROUND_OF_32, ROUND_OF_16, QUARTERFINALS, SEMIFINALS, THIRD_PLACE, FINAL, ALL_KNOCKOUT_MATCHES, round_of_match
    
    group_standings = {}
    for grp, teams in WC2026_GROUPS.items():
        standings = {t: GroupStanding(team=SPANISH_TO_ENGLISH.get(t, t)) for t in teams}
        grp_teams_set = set(teams)
        for m in loaded_matches:
            if m["home"] in grp_teams_set and m["away"] in grp_teams_set:
                h_spa, a_spa = m["home"], m["away"]
                gh, ga = m["home_score"], m["away_score"]
                
                standings[h_spa].goals_for += gh
                standings[h_spa].goals_against += ga
                standings[a_spa].goals_for += ga
                standings[a_spa].goals_against += gh
                
                if gh > ga:
                    standings[h_spa].points += 3
                elif gh < ga:
                    standings[a_spa].points += 3
                else:
                    standings[h_spa].points += 1
                    standings[a_spa].points += 1
                    
        group_standings[grp] = sorted(standings.values(), key=lambda s: s.sort_key)
        
    ctx = {}
    for grp, standings in group_standings.items():
        ctx[f"1st_{grp}"] = standings[0].team
        ctx[f"2nd_{grp}"] = standings[1].team
        
    rng_det = np.random.default_rng(42)
    best_thirds = select_best_thirds(group_standings, n=8)
    third_assignment = assign_thirds_to_slots(best_thirds, ROUND_OF_32, rng_det)
    ctx.update(third_assignment)
    
    # 3. Resolve parent matches to unlock Knockouts recursively
    for mid, m in played_ko_map.items():
        gh, ga = m["home_score"], m["away_score"]
        h_eng = SPANISH_TO_ENGLISH.get(m["home"], m["home"])
        a_eng = SPANISH_TO_ENGLISH.get(m["away"], m["away"])
        if gh > ga:
            winner, loser = h_eng, a_eng
        else:
            winner, loser = a_eng, h_eng
        ctx[f"winner_match_{mid}"] = winner
        ctx[f"loser_match_{mid}"] = loser
        
    unplayed_ko = []
    played_ko = []
    
    for match in ALL_KNOCKOUT_MATCHES:
        try:
            team_a_eng = resolve_slot(match.slot_a, ctx)
            team_b_eng = resolve_slot(match.slot_b, ctx)
            
            a_spa = eng_to_spanish.get(team_a_eng, team_a_eng)
            b_spa = eng_to_spanish.get(team_b_eng, team_b_eng)
            
            round_label = round_of_match(match.match_id)
            round_spa = {
                "round_of_32": "16avos de Final",
                "round_of_16": "Octavos de Final",
                "quarterfinals": "Cuartos de Final",
                "semifinals": "Semifinales",
                "third_place": "Tercer Puesto",
                "final": "Gran Final"
            }.get(round_label, round_label)
            
            if match.match_id in played_ko_map:
                m = played_ko_map[match.match_id]
                played_ko.append({
                    "id": f"ko_{match.match_id}",
                    "type": "knockout",
                    "match_id": match.match_id,
                    "home": a_spa,
                    "away": b_spa,
                    "home_score": m["home_score"],
                    "away_score": m["away_score"],
                    "label": f"✅ {round_spa} (Partido {match.match_id}): {a_spa} {m['home_score']} - {m['away_score']} {b_spa} [Cargado]"
                })
            else:
                unplayed_ko.append({
                    "id": f"ko_{match.match_id}",
                    "type": "knockout",
                    "match_id": match.match_id,
                    "home": a_spa,
                    "away": b_spa,
                    "label": f"⚔️ {round_spa} (Partido {match.match_id}): {a_spa} vs {b_spa}"
                })
        except KeyError:
            continue
            
    return unplayed_ko + played_ko

# Sidebar Settings
st.sidebar.header("⚙️ Configuración del Predictor")
prediction_mode = st.sidebar.radio(
    "Modelo de Goles (Marcadores):",
    ["Directo del Transformer (Conservador)", "Simulación Poisson (Realista y Goleador)"],
    index=1 # Default to exciting Poisson simulation
)

# Collapsible ELO Rankings in Sidebar
with st.sidebar.expander("📈 Rankings de Fuerza (ELO Oficial)"):
    lead_records = []
    for t, feat in team_features.items():
        spa_name = SPANISH_TO_ENGLISH.get(t, t)
        lead_records.append({
            "Selección": spa_name,
            "ELO": int(feat["elo"])
        })
    df_leaderboard = pd.DataFrame(lead_records).sort_values("ELO", ascending=False).reset_index(drop=True)
    df_leaderboard.index += 1
    st.dataframe(df_leaderboard, height=400)

# Title section
st.markdown('<div class="main-title">🏆 Prode Inteligencia Artificial 2026 🏆</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Modelos de Deep Learning (Transformers) para predecir marcadores y simular el Mundial completo</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Main Tabs Setup
# -------------------------------------------------------------
tab1, tab_tourn, tab4 = st.tabs([
    "⚔️ Simular Partido Independiente",
    "🎮 Simular Mundial Completo (Prode)",
    "🔄 Cargar Resultados en Vivo"
])

# -------------------------------------------------------------
# TAB 1: SIMULADOR INTERACTIVO (FT-TRANSFORMER V2)
# -------------------------------------------------------------
with tab1:
    st.header("⚔️ Simular Partido Único")
    st.write("Selecciona dos países y simula su enfrentamiento en tiempo real.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuración del Encuentro")
        
        h_web_select = st.selectbox("Selección Local (Team A):", web_spanish_names, index=web_spanish_names.index("Argentina"))
        a_web_select = st.selectbox("Selección Visitante (Team B):", web_spanish_names, index=web_spanish_names.index("Francia"))
        
        h_eng = spanish_to_english_web[h_web_select]
        a_eng = spanish_to_english_web[a_web_select]
        
        # Retrieve default stats from team history automatically
        feat_a = team_features.get(h_eng, {'elo': 1500, 'form_pts': 1.0, 'recent_goals': 1.0, 'form10_pts': 1.0})
        feat_b = team_features.get(a_eng, {'elo': 1500, 'form_pts': 1.0, 'recent_goals': 1.0, 'form10_pts': 1.0})
        
        # Load stats natively (Zero sliders/clutter)
        h_elo = float(feat_a['elo'])
        a_elo = float(feat_b['elo'])
        h_form = float(feat_a['form_pts'])
        a_form = float(feat_b['form_pts'])
        h_goals = float(feat_a['recent_goals'])
        a_goals = float(feat_b['recent_goals'])
        
        # Standard tournament environment (FIFA World Cup - Neutral Ground)
        t_class_idx = tourn_map.get("FIFA World Cup", 0)
        venue_idx = 2
        neutral_val = 1
        
    with col2:
        st.subheader("🔮 Predicción en Tiempo Real")
        
        # Pack features into tensors
        cat_tensor = torch.tensor([[t_class_idx, neutral_val, venue_idx]], dtype=torch.long)
        cont_tensor = torch.tensor([[
            h_elo, a_elo, h_elo - a_elo,
            h_form, a_form,
            h_goals, a_goals,
            h_form, a_form # form 10 approximation
        ]], dtype=torch.float32)
        
        # Forward pass on FT-Transformer (v2)
        with torch.no_grad():
            logits = model(cat_tensor, cont_tensor)
            probs = F.softmax(logits, dim=-1)[0].numpy() # shape (36,)
            
        # Map 36 classes back to 3 outcomes (Home, Draw, Away) for v2
        p_home, p_draw, p_away = 0.0, 0.0, 0.0
        for idx in range(36):
            h_g = idx // 6
            a_g = idx % 6
            if h_g > a_g: p_home += probs[idx]
            elif h_g == a_g: p_draw += probs[idx]
            else: p_away += probs[idx]
            
        # Run D10Sformer v1 on the fly for comparison (without custom lineups selected, using default fallbacks)
        from data.tokenizer import MatchTokenizer, MatchDocument, RollingFeatures
        tokenizer_v1 = MatchTokenizer(vocab, max_seq_length=80)
        
        doc_v1 = MatchDocument(
            tournament="FIFA World Cup", team_a=h_eng, team_b=a_eng, venue="neutral", stage=None,
            lineup_a=None, lineup_b=None,
            features=RollingFeatures(
                home_elo=h_elo, away_elo=a_elo,
                home_form_pts=h_form, away_form_pts=a_form,
                home_recent_goals=h_goals, away_recent_goals=a_goals
            )
        )
        out_v1 = tokenizer_v1.tokenize(doc_v1)
        tok_v1_tensor = torch.tensor([out_v1.token_ids], dtype=torch.long)
        seg_v1_tensor = torch.tensor([out_v1.segment_ids], dtype=torch.long)
        with torch.no_grad():
            out_model_v1 = model_v1(tok_v1_tensor, seg_v1_tensor)
        res_v1_probs = F.softmax(out_model_v1["result_logits"], dim=-1)[0].numpy()
        p_h_v1, p_d_v1, p_a_v1 = res_v1_probs[0], res_v1_probs[1], res_v1_probs[2]
        score_v1_probs = F.softmax(out_model_v1["score_logits"], dim=-1)[0].numpy()
        
        # Calculate recommended scores for both models
        if prediction_mode == "Simulación Poisson (Realista y Goleador)":
            rec_h, rec_a = p_to_score(p_home, p_draw, p_away, h_elo, a_elo, h_goals, a_goals)
            rec_prob_str = "Modelo Calibrado (Poisson)"
            
            rec_h_v1, rec_a_v1 = p_to_score(p_h_v1, p_d_v1, p_a_v1, h_elo, a_elo, h_goals, a_goals)
        else:
            best_score_idx = np.argmax(probs)
            rec_h = best_score_idx // 6
            rec_a = best_score_idx % 6
            rec_prob_str = f"Probabilidad: {probs[best_score_idx]*100:.1f}%"
            
            best_score_v1_idx = np.argmax(score_v1_probs)
            rec_h_v1 = best_score_v1_idx // 6
            rec_a_v1 = best_score_v1_idx % 6
            
        # 1. Main Recommended Score Card (FT-Transformer v2)
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 1.5rem;">
            <div class="metric-label">Marcador Recomendado (IA Premium)</div>
            <div class="metric-value" style="font-size: 2.4rem; color: #2563EB;">{rec_h} - {rec_a}</div>
            <div style="font-weight: 700; color: #64748B; font-size: 0.85rem;">{rec_prob_str}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. Donut Chart for FT-Transformer v2
        labels_v2 = [f"Victoria {h_web_select}", "Empate", f"Victoria {a_web_select}"]
        values_v2 = [p_home, p_draw, p_away]
        colors_v2 = ['#10B981', '#F59E0B', '#EF4444']
        fig_v2 = go.Figure(data=[go.Pie(labels=labels_v2, values=values_v2, hole=.4, marker_colors=colors_v2)])
        fig_v2.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_v2, use_container_width=True)
        
        # 3. Subtle Reference to Classic Model
        st.info(f"🧠 **Referencia IA Clásica (v1):** Marcador `{rec_h_v1} - {rec_a_v1}` | Victoria {h_web_select}: `{p_h_v1*100:.1f}%` | Empate: `{p_d_v1*100:.1f}%` | Victoria {a_web_select}: `{p_a_v1*100:.1f}%`.")

# -------------------------------------------------------------
# TAB TOURN: SIMULADOR DE MUNDIAL COMPLETO
# -------------------------------------------------------------
with tab_tourn:
    st.header("🎮 Simular Mundial Completo")
    st.write("Simula el torneo para generar tus prodes con marcadores exactos de los 104 partidos.")
    
    col_sim_cfg, col_sim_act = st.columns([1, 2])
    with col_sim_cfg:
        st.subheader("Configuración")
        sim_predictor_choice = st.selectbox(
            "Modelo de Inteligencia Artificial:",
            ["🏆 IA Premium (Más Inteligente)", "🧠 IA Clásica (Rápida)"],
            index=0,
            key="sim_pred_choice_widget"
        )
        
        sim_goals_mode = st.selectbox(
            "Estilo de Goles (Marcadores):",
            ["🛡️ Marcadores Defensivos / Ajustados", "🔥 Marcadores Goleadores / Emocionantes"],
            index=1,
            key="sim_goals_widget"
        )
        
        st.markdown("---")
        st.markdown("**🎲 Opción 1: Generador de Prode (Azar)**")
        run_sim = st.button("🏁 Simular 1 Mundial Completo", type="primary", use_container_width=True)
        st.caption("Genera un fixture único con goles exactos y llaves de playoffs (con sorpresas del fútbol real).")
        
        st.markdown("---")
        st.markdown("**📊 Opción 2: Análisis Estadístico (Fiabilidad)**")
        mc_iters = st.slider("Iteraciones Monte Carlo:", min_value=50, max_value=1000, value=200, step=50, key="sim_mc_iters_widget")
        run_mc = st.button("📈 Correr Análisis Monte Carlo", type="secondary", use_container_width=True)
        st.caption("Consolida el verdadero ranking de favoritos promediando cientos de simulación.")
        
    with col_sim_act:
        st.subheader("📊 ¿Azar o Fiabilidad?")
        st.write(
            "🎲 **Opción 1:** Genera un torneo único con goles y playoffs, ideal para obtener combinaciones rápidas de resultados para tu prode.\n\n"
            "📈 **Opción 2:** Simula el torneo cientos de veces en paralelo y consolida los favoritos para darte probabilidades científicas."
        )
        if "sim_result" in st.session_state:
            st.success(f"Mostrando simulación anterior (Semilla: {st.session_state['sim_seed']}, Modelo: {st.session_state['sim_predictor']})")
        elif "mc_df" in st.session_state:
            st.success(f"Mostrando análisis Monte Carlo anterior ({st.session_state['mc_iters']} iteraciones, Modelo: {st.session_state['mc_predictor']})")
            
    if run_sim:
        with st.spinner("Simulando fase de grupos (72 partidos), calculando goles y resolviendo playoffs (32 partidos)..."):
            import random
            from simulation.simulator import GroupStanding, select_best_thirds, assign_thirds_to_slots, sample_match_result, sample_knockout_winner, sample_goals, TournamentResult, STAGES, REACHED_AT_LEAST
            from simulation.bracket import ROUND_OF_32, ROUND_OF_16, QUARTERFINALS, SEMIFINALS, THIRD_PLACE, FINAL, ALL_KNOCKOUT_MATCHES, round_of_match
            
            seed = random.randint(1, 100000)
            rng = np.random.default_rng(seed)
            
            # 1. Predictor Callable
            def current_predictor(a_eng, b_eng, venue="neutral"):
                feat_a = team_features.get(a_eng, {'elo': 1500, 'form_pts': 1.0, 'recent_goals': 1.0})
                feat_b = team_features.get(b_eng, {'elo': 1500, 'form_pts': 1.0, 'recent_goals': 1.0})
                
                if sim_predictor_choice == "🏆 IA Premium (Más Inteligente)":
                    cat_tensor = torch.tensor([[tourn_map.get("FIFA World Cup", 0), 1, 2]], dtype=torch.long)
                    cont_tensor = torch.tensor([[
                        feat_a['elo'], feat_b['elo'], feat_a['elo'] - feat_b['elo'],
                        feat_a['form_pts'], feat_b['form_pts'],
                        feat_a['recent_goals'], feat_b['recent_goals'],
                        feat_a['form_pts'], feat_b['form_pts']
                    ]], dtype=torch.float32)
                    with torch.no_grad():
                        logits = model(cat_tensor, cont_tensor)
                        probs = F.softmax(logits, dim=-1)[0].numpy()
                    p_home, p_draw, p_away = 0.0, 0.0, 0.0
                    for idx in range(36):
                        h_g = idx // 6
                        a_g = idx % 6
                        if h_g > a_g: p_home += probs[idx]
                        elif h_g == a_g: p_draw += probs[idx]
                        else: p_away += probs[idx]
                    return np.array([p_home, p_draw, p_away])
                else:
                    from data.tokenizer import MatchTokenizer, MatchDocument, RollingFeatures
                    tokenizer_v1 = MatchTokenizer(vocab, max_seq_length=80)
                    doc_v1 = MatchDocument(
                        tournament="FIFA World Cup", team_a=a_eng, team_b=b_eng, venue="neutral", stage=None,
                        lineup_a=None, lineup_b=None,
                        features=RollingFeatures(
                            home_elo=feat_a['elo'], away_elo=feat_b['elo'],
                            home_form_pts=feat_a['form_pts'], away_form_pts=feat_b['form_pts'],
                            home_recent_goals=feat_a['recent_goals'], away_recent_goals=feat_b['recent_goals']
                        )
                    )
                    out_v1 = tokenizer_v1.tokenize(doc_v1)
                    tok_v1_tensor = torch.tensor([out_v1.token_ids], dtype=torch.long)
                    seg_v1_tensor = torch.tensor([out_v1.segment_ids], dtype=torch.long)
                    with torch.no_grad():
                        out_model_v1 = model_v1(tok_v1_tensor, seg_v1_tensor)
                    res_v1_probs = F.softmax(out_model_v1["result_logits"], dim=-1)[0].numpy()
                    return np.array([res_v1_probs[0], res_v1_probs[1], res_v1_probs[2]])
            
            # 2. Simulate Group Stage stochastically and record exact scores
            sim_group_scores = defaultdict(list)
            group_standings = {}
            
            for grp, teams in WC2026_GROUPS.items():
                standings = {t: GroupStanding(team=t) for t in teams}
                for i in range(len(teams)):
                    for j in range(i + 1, len(teams)):
                        a, b = teams[i], teams[j]
                        probs = current_predictor(a, b)
                        p_h, p_d, p_a = float(probs[0]), float(probs[1]), float(probs[2])
                        
                        feat_a_match = team_features.get(a, {'elo': 1500.0, 'form_pts': 1.0, 'recent_goals': 1.0})
                        feat_b_match = team_features.get(b, {'elo': 1500.0, 'form_pts': 1.0, 'recent_goals': 1.0})
                        
                        # Internal Monte Carlo: 100 parallel runs per match
                        n_samples = 100
                        match_samples = []
                        for _ in range(n_samples):
                            s_outcome = sample_match_result(p_h, p_d, p_a, rng)
                            if sim_goals_mode == "🔥 Marcadores Goleadores / Emocionantes":
                                s_gh, s_ga = p_to_score(p_h, p_d, p_a, feat_a_match['elo'], feat_b_match['elo'], feat_a_match['recent_goals'], feat_b_match['recent_goals'])
                            else:
                                s_gh, s_ga = sample_goals(p_h, p_d, p_a, rng=rng)
                            # Enforce
                            if s_outcome == "home_win" and s_gh <= s_ga:
                                s_gh = s_ga + 1
                            elif s_outcome == "away_win" and s_ga <= s_gh:
                                s_ga = s_gh + 1
                            elif s_outcome == "draw" and s_gh != s_ga:
                                s_gh = s_ga = max(s_gh, s_ga)
                            match_samples.append((s_outcome, s_gh, s_ga))
                            
                        # Majority vote on outcome
                        outcomes = [s[0] for s in match_samples]
                        outcome = max(set(outcomes), key=outcomes.count)
                        
                        # Average of goals matching the majority outcome
                        matching_scores = [s for s in match_samples if s[0] == outcome]
                        ga_h = int(round(sum(s[1] for s in matching_scores) / len(matching_scores)))
                        ga_a = int(round(sum(s[2] for s in matching_scores) / len(matching_scores)))
                        
                        # Final sanity enforcement
                        if outcome == "home_win" and ga_h <= ga_a:
                            ga_h = ga_a + 1
                        elif outcome == "away_win" and ga_a <= ga_h:
                            ga_a = ga_h + 1
                        elif outcome == "draw" and ga_h != ga_a:
                            ga_h = ga_a = max(ga_h, ga_a)
                            
                        # Record exact match score
                        sim_group_scores[grp].append({
                            "home": a, "away": b,
                            "home_score": ga_h, "away_score": ga_a
                        })
                        
                        # Update standings
                        standings[a].goals_for += ga_h
                        standings[a].goals_against += ga_a
                        standings[b].goals_for += ga_a
                        standings[b].goals_against += ga_h
                        
                        if outcome == "home_win":
                            standings[a].points += 3
                        elif outcome == "away_win":
                            standings[b].points += 3
                        else:
                            standings[a].points += 1
                            standings[b].points += 1
                
                # Sort group standings according to official FIFA tiebreaker
                group_standings[grp] = sorted(standings.values(), key=lambda s: s.sort_key)
                
            # 3. Build context for playoffs
            ctx = {}
            progressions = {}
            for grp, standings in group_standings.items():
                ctx[f"1st_{grp}"] = standings[0].team
                ctx[f"2nd_{grp}"] = standings[1].team
                for s in standings[3:]:
                    progressions[s.team] = "group"
                for s in standings[:2]:
                    progressions[s.team] = "round_of_32"
                    
            # 4. Select best thirds and assign slots
            best_thirds = select_best_thirds(group_standings, n=8)
            third_assignment = assign_thirds_to_slots(best_thirds, ROUND_OF_32, rng)
            ctx.update(third_assignment)
            for slot, team in third_assignment.items():
                progressions[team] = "round_of_32"
            picked_thirds = set(third_assignment.values())
            for grp, standings in group_standings.items():
                if len(standings) >= 3 and standings[2].team not in picked_thirds:
                    progressions[standings[2].team] = "group"
                    
            # 5. Simulate Playoff Knockouts stochastically and record exact scores
            knockout_winners = {}
            knockout_losers = {}
            sim_knockout_scores = {}
            
            from simulation.simulator import resolve_slot, round_of_match
            for match in ALL_KNOCKOUT_MATCHES:
                team_a = resolve_slot(match.slot_a, ctx)
                team_b = resolve_slot(match.slot_b, ctx)
                
                probs = current_predictor(team_a, team_b)
                p_h, p_d, p_a = float(probs[0]), float(probs[1]), float(probs[2])
                
                feat_a_match = team_features.get(team_a, {'elo': 1500.0, 'form_pts': 1.0, 'recent_goals': 1.0})
                feat_b_match = team_features.get(team_b, {'elo': 1500.0, 'form_pts': 1.0, 'recent_goals': 1.0})
                
                # Internal Monte Carlo: 100 parallel runs per playoff match
                n_samples = 100
                match_samples = []
                for _ in range(n_samples):
                    s_outcome = sample_knockout_winner(p_h, p_d, p_a, rng)
                    if sim_goals_mode == "🔥 Marcadores Goleadores / Emocionantes":
                        s_gh, s_ga = p_to_score(p_h, p_d, p_a, feat_a_match['elo'], feat_b_match['elo'], feat_a_match['recent_goals'], feat_b_match['recent_goals'])
                    else:
                        s_gh, s_ga = sample_goals(p_h, p_d, p_a, rng=rng)
                    # Enforce
                    if s_outcome == "home_win" and s_gh <= s_ga:
                        s_gh = s_ga + 1
                    elif s_outcome != "home_win" and s_ga <= s_gh:
                        s_ga = s_gh + 1
                    match_samples.append((s_outcome, s_gh, s_ga))
                    
                # Majority vote on outcome
                outcomes = [s[0] for s in match_samples]
                outcome = max(set(outcomes), key=outcomes.count)
                
                # Average of goals matching the majority outcome
                matching_scores = [s for s in match_samples if s[0] == outcome]
                ga_h = int(round(sum(s[1] for s in matching_scores) / len(matching_scores)))
                ga_a = int(round(sum(s[2] for s in matching_scores) / len(matching_scores)))
                
                if outcome == "home_win":
                    winner, loser = team_a, team_b
                    if ga_h <= ga_a:
                        ga_h = ga_a + 1
                else:
                    winner, loser = team_b, team_a
                    if ga_a <= ga_h:
                        ga_a = ga_h + 1
                        
                knockout_winners[match.match_id] = winner
                knockout_losers[match.match_id] = loser
                ctx[f"winner_match_{match.match_id}"] = winner
                ctx[f"loser_match_{match.match_id}"] = loser
                
                # Record exact score of knockout match
                sim_knockout_scores[match.match_id] = {
                    "home": team_a, "away": team_b,
                    "home_score": ga_h, "away_score": ga_a
                }
                
                # Furthest stage tracking
                round_name = round_of_match(match.match_id)
                for t in (team_a, team_b):
                    current = progressions.get(t, "group")
                    if STAGES.index(round_name) > STAGES.index(current):
                        progressions[t] = round_name
                        
            # Determine Champion
            champion = knockout_winners[FINAL.match_id]
            runner_up = knockout_losers[FINAL.match_id]
            third_place = knockout_winners[THIRD_PLACE.match_id]
            progressions[champion] = "champion"
            if STAGES.index(progressions.get(runner_up, "group")) < STAGES.index("final"):
                progressions[runner_up] = "final"
                
            sim_result = TournamentResult(
                group_standings=group_standings,
                knockout_winners=knockout_winners,
                knockout_losers=knockout_losers,
                progressions=progressions,
                champion=champion,
                runner_up=runner_up,
                third_place=third_place
            )
            
            # Save all in state
            st.session_state["sim_result"] = sim_result
            st.session_state["sim_group_scores"] = sim_group_scores
            st.session_state["sim_knockout_scores"] = sim_knockout_scores
            st.session_state["sim_seed"] = seed
            st.session_state["sim_predictor"] = sim_predictor_choice
            
            # Resolve Spanish champion name for the toast
            eng_to_spanish_toast = {v: k for k, v in SPANISH_TO_ENGLISH.items()}
            eng_to_spanish_toast["Netherlands"] = "Países Bajos"
            eng_to_spanish_toast["Tunisia"] = "Túnez"
            eng_to_spanish_toast["South Korea"] = "República de Corea"
            eng_to_spanish_toast["Czech Republic"] = "República Checa"
            eng_to_spanish_toast["South Africa"] = "Sudáfrica"
            champ_spa_toast = eng_to_spanish_toast.get(champion, champion)
            
            st.toast(f"🏆 ¡{champ_spa_toast} se ha consagrado Campeón del Mundo!", icon="🏆")
            st.success(f"¡Mundial simulado con éxito! Semilla estocástica: {seed}")
            
    if run_mc:
        with st.spinner(f"Corriendo {mc_iters} simulaciones Monte Carlo completas del Mundial..."):
            import random
            from simulation.simulator import monte_carlo
            
            seed = random.randint(1, 100000)
            rng = np.random.default_rng(seed)
            
            # Predictor Callable
            def current_predictor(a_eng, b_eng, venue="neutral"):
                feat_a = team_features.get(a_eng, {'elo': 1500, 'form_pts': 1.0, 'recent_goals': 1.0})
                feat_b = team_features.get(b_eng, {'elo': 1500, 'form_pts': 1.0, 'recent_goals': 1.0})
                
                if sim_predictor_choice == "🏆 IA Premium (Más Inteligente)":
                    cat_tensor = torch.tensor([[tourn_map.get("FIFA World Cup", 0), 1, 2]], dtype=torch.long)
                    cont_tensor = torch.tensor([[
                        feat_a['elo'], feat_b['elo'], feat_a['elo'] - feat_b['elo'],
                        feat_a['form_pts'], feat_b['form_pts'],
                        feat_a['recent_goals'], feat_b['recent_goals'],
                        feat_a['form_pts'], feat_b['form_pts']
                    ]], dtype=torch.float32)
                    with torch.no_grad():
                        logits = model(cat_tensor, cont_tensor)
                        probs = F.softmax(logits, dim=-1)[0].numpy()
                    p_home, p_draw, p_away = 0.0, 0.0, 0.0
                    for idx in range(36):
                        h_g = idx // 6
                        a_g = idx % 6
                        if h_g > a_g: p_home += probs[idx]
                        elif h_g == a_g: p_draw += probs[idx]
                        else: p_away += probs[idx]
                    return np.array([p_home, p_draw, p_away])
                else:
                    from data.tokenizer import MatchTokenizer, MatchDocument, RollingFeatures
                    tokenizer_v1 = MatchTokenizer(vocab, max_seq_length=80)
                    doc_v1 = MatchDocument(
                        tournament="FIFA World Cup", team_a=a_eng, team_b=b_eng, venue="neutral", stage=None,
                        lineup_a=None, lineup_b=None,
                        features=RollingFeatures(
                            home_elo=feat_a['elo'], away_elo=feat_b['elo'],
                            home_form_pts=feat_a['form_pts'], away_form_pts=feat_b['form_pts'],
                            home_recent_goals=feat_a['recent_goals'], away_recent_goals=feat_b['recent_goals']
                        )
                    )
                    out_v1 = tokenizer_v1.tokenize(doc_v1)
                    tok_v1_tensor = torch.tensor([out_v1.token_ids], dtype=torch.long)
                    seg_v1_tensor = torch.tensor([out_v1.segment_ids], dtype=torch.long)
                    with torch.no_grad():
                        out_model_v1 = model_v1(tok_v1_tensor, seg_v1_tensor)
                    res_v1_probs = F.softmax(out_model_v1["result_logits"], dim=-1)[0].numpy()
                    return np.array([res_v1_probs[0], res_v1_probs[1], res_v1_probs[2]])
                    
            from simulation.simulator import FixedResults
            fr = FixedResults()
            for m in loaded_matches:
                if "match_id" in m:
                    fr.add_knockout_result(m["match_id"], spanish_to_english_web.get(m["winner"], m["winner"]))
                else:
                    match_group = None
                    for grp, teams in WC2026_GROUPS.items():
                        if m["home"] in teams and m["away"] in teams:
                            match_group = grp
                            break
                    if match_group:
                        fr.add_group_match(
                            match_group,
                            spanish_to_english_web.get(m["home"], m["home"]),
                            spanish_to_english_web.get(m["away"], m["away"]),
                            m["home_score"],
                            m["away_score"]
                        )
            mc_result = monte_carlo(current_predictor, n_iters=mc_iters, seed=seed, fixed_results=fr)
            df_mc = mc_result.to_dataframe()
            
            st.session_state["mc_df"] = df_mc
            st.session_state["mc_iters"] = mc_iters
            st.session_state["mc_predictor"] = sim_predictor_choice
            st.session_state["mc_seed"] = seed
            
            # Clear single simulation results when running Monte Carlo
            if "sim_result" in st.session_state:
                del st.session_state["sim_result"]
                
            st.success(f"¡Análisis Monte Carlo de {mc_iters} simulaciones finalizado con éxito!")
            
    if "sim_result" in st.session_state and "sim_group_scores" in st.session_state and "sim_knockout_scores" in st.session_state:
        sim_result = st.session_state["sim_result"]
        sim_group_scores = st.session_state["sim_group_scores"]
        sim_knockout_scores = st.session_state["sim_knockout_scores"]
        
        # English to Spanish name dictionary
        eng_to_spanish = {v: k for k, v in SPANISH_TO_ENGLISH.items()}
        eng_to_spanish["Netherlands"] = "Países Bajos"
        eng_to_spanish["Tunisia"] = "Túnez"
        eng_to_spanish["South Korea"] = "República de Corea"
        eng_to_spanish["Czech Republic"] = "República Checa"
        eng_to_spanish["South Africa"] = "Sudáfrica"
        
        champ = sim_result.champion
        runner = sim_result.runner_up
        third = sim_result.third_place
        
        champ_spa = eng_to_spanish.get(champ, champ)
        runner_spa = eng_to_spanish.get(runner, runner)
        third_spa = eng_to_spanish.get(third, third)
        
        st.markdown(f"""
        <div style="background-color: #FDF2E9; border: 3px solid #F59E0B; border-radius: 15px; padding: 2rem; text-align: center; margin-bottom: 2rem;">
            <h1 style="color: #D97706; margin-bottom: 0.5rem; font-size: 3rem;">🏆 CAMPEÓN DEL MUNDO 🏆</h1>
            <h2 style="color: #1E3A8A; font-size: 4rem; font-weight: 900; letter-spacing: 2px;">{champ_spa.upper()}</h2>
            <div style="display: flex; justify-content: center; gap: 3rem; margin-top: 1.5rem; font-size: 1.4rem; font-weight: 700; color: #4B5563;">
                <div>🥈 Subcampeón: <span style="color: #1F2937;">{runner_spa}</span></div>
                <div>🥉 Tercer Puesto: <span style="color: #1F2937;">{third_spa}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Side-by-side columns: left for group standings, right for the Playoff Tree Diagram
        col_left_res, col_right_res = st.columns([2, 3])
        
        with col_left_res:
            st.markdown("### 📊 Posiciones de la Fase de Grupos")
            st.write("Selecciona cualquier grupo para ver la tabla de posiciones consolidada y desplegar los marcadores exactos:")
            
            # Render groups 4 by 4 inside tabs to keep it clean and ultra-visual
            g_tab_a_d, g_tab_e_h, g_tab_i_l = st.tabs(["Grupos A - D", "Grupos E - H", "Grupos I - L"])
            
            with g_tab_a_d:
                col_a, col_b, col_c, col_d = st.columns(4)
                cols = [col_a, col_b, col_c, col_d]
                for idx, grp in enumerate(["Group_A", "Group_B", "Group_C", "Group_D"]):
                    standings = sim_result.group_standings[grp]
                    records = []
                    for rank, s in enumerate(standings):
                        records.append({
                            "Pos": rank + 1,
                            "País": eng_to_spanish.get(s.team, s.team),
                            "Pts": s.points,
                            "DG": s.goal_diff
                        })
                    cols[idx].markdown(f"##### Grupo {grp[-1]}")
                    cols[idx].dataframe(pd.DataFrame(records), hide_index=True)
                    
                    # Show group matches inside expander
                    with cols[idx].expander("👁️ Ver Marcadores"):
                        for gm in sim_group_scores[grp]:
                            h_s = eng_to_spanish.get(gm["home"], gm["home"])
                            a_s = eng_to_spanish.get(gm["away"], gm["away"])
                            st.write(f"{h_s} **{gm['home_score']} - {gm['away_score']}** {a_s}")
                    
            with g_tab_e_h:
                col_e, col_f, col_g, col_h = st.columns(4)
                cols = [col_e, col_f, col_g, col_h]
                for idx, grp in enumerate(["Group_E", "Group_F", "Group_G", "Group_H"]):
                    standings = sim_result.group_standings[grp]
                    records = []
                    for rank, s in enumerate(standings):
                        records.append({
                            "Pos": rank + 1,
                            "País": eng_to_spanish.get(s.team, s.team),
                            "Pts": s.points,
                            "DG": s.goal_diff
                        })
                    cols[idx].markdown(f"##### Grupo {grp[-1]}")
                    cols[idx].dataframe(pd.DataFrame(records), hide_index=True)
                    
                    # Show group matches inside expander
                    with cols[idx].expander("👁️ Ver Marcadores"):
                        for gm in sim_group_scores[grp]:
                            h_s = eng_to_spanish.get(gm["home"], gm["home"])
                            a_s = eng_to_spanish.get(gm["away"], gm["away"])
                            st.write(f"{h_s} **{gm['home_score']} - {gm['away_score']}** {a_s}")
                    
            with g_tab_i_l:
                col_i, col_j, col_k, col_l = st.columns(4)
                cols = [col_i, col_j, col_k, col_l]
                for idx, grp in enumerate(["Group_I", "Group_J", "Group_K", "Group_L"]):
                    standings = sim_result.group_standings[grp]
                    records = []
                    for rank, s in enumerate(standings):
                        records.append({
                            "Pos": rank + 1,
                            "País": eng_to_spanish.get(s.team, s.team),
                            "Pts": s.points,
                            "DG": s.goal_diff
                        })
                    cols[idx].markdown(f"##### Grupo {grp[-1]}")
                    cols[idx].dataframe(pd.DataFrame(records), hide_index=True)
                    
                    # Show group matches inside expander
                    with cols[idx].expander("👁️ Ver Marcadores"):
                        for gm in sim_group_scores[grp]:
                            h_s = eng_to_spanish.get(gm["home"], gm["home"])
                            a_s = eng_to_spanish.get(gm["away"], gm["away"])
                            st.write(f"{h_s} **{gm['home_score']} - {gm['away_score']}** {a_s}")
                    
        # Expander for Playoffs Tree Diagram in right column
        with col_right_res:
            st.markdown("### ⚔️ Cuadro de Playoffs (Árbol Visual)")
            
            # Sub-helper to render a small match card in HTML/CSS with exact scores
            def get_bracket_match_card(mid, title_label):
                score_info = sim_knockout_scores[mid]
                h_spa = eng_to_spanish.get(score_info["home"], score_info["home"])
                a_spa = eng_to_spanish.get(score_info["away"], score_info["away"])
                
                h_score = score_info["home_score"]
                a_score = score_info["away_score"]
                
                # Check who is the winner to apply bold green styling
                is_home_winner = h_score > a_score
                
                home_class = "winner" if is_home_winner else "loser"
                away_class = "loser" if is_home_winner else "winner"
                
                # Emoji indicators
                home_emoji = "✅" if is_home_winner else "❌"
                away_emoji = "❌" if is_home_winner else "✅"
                
                return f"""
                <div class="bracket-match">
                    <div class="match-header">{title_label}</div>
                    <div class="bracket-team {home_class}">
                        <span>{home_emoji} {h_spa}</span>
                        <span>{h_score}</span>
                    </div>
                    <div class="bracket-team {away_class}">
                        <span>{away_emoji} {a_spa}</span>
                        <span>{a_score}</span>
                    </div>
                </div>
                """
            
            # CSS Styles
            css_styles = """
            <style>
                .bracket-container {
                    display: flex;
                    flex-direction: row;
                    justify-content: space-between;
                    background-color: transparent; /* Seamless page integration */
                    padding: 0.5rem;
                    border-radius: 0;
                    overflow-x: auto;
                    gap: 1.2rem;
                    border: none; /* Minimalist flat look */
                }
                .round {
                    display: flex;
                    flex-direction: column;
                    justify-content: space-around;
                    width: 19%;
                    min-width: 180px;
                }
                .round-title {
                    text-align: center;
                    color: #1E3A8A; /* Primary theme color */
                    font-weight: 800;
                    font-size: 0.8rem;
                    margin-bottom: 0.8rem;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    border-bottom: 2px solid #E2E8F0;
                    padding-bottom: 0.4rem;
                }
                .bracket-match {
                    background: #FFFFFF; /* Modern white card */
                    border: 1px solid #E2E8F0; /* Soft border */
                    border-radius: 8px;
                    padding: 0.5rem;
                    margin: 0.3rem 0;
                    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05); /* Soft minimalist shadow */
                    transition: all 0.2s ease-in-out;
                }
                .bracket-match:hover {
                    border-color: #3B82F6; /* Vibrant blue hover border */
                    background-color: #F8FAFC;
                    transform: translateY(-1px); /* Gentle slide up */
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
                }
                .match-header {
                    font-size: 0.65rem;
                    color: #94A3B8;
                    margin-bottom: 0.2rem;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                .bracket-team {
                    display: flex;
                    justify-content: space-between;
                    padding: 0.1rem 0;
                    font-size: 0.78rem;
                    font-weight: 600;
                    color: #1E293B; /* Charcoal dark text */
                }
                .bracket-team.winner {
                    color: #059669; /* Rich green for winner */
                    font-weight: 700;
                }
                .bracket-team.loser {
                    color: #CBD5E1; /* Light faded grey for loser */
                    text-decoration: line-through;
                    font-weight: normal;
                }
            </style>
            """
            
            # Build Bracket columns
            bracket_html = css_styles + '<div class="bracket-container">'
            
            # 1. Round of 32 column
            bracket_html += '<div class="round"><div class="round-title">16avos de Final</div>'
            for mid in [74, 77, 73, 75, 76, 78, 79, 80, 83, 84, 81, 82, 86, 88, 85, 87]:
                bracket_html += get_bracket_match_card(mid, f"Partido {mid}")
            bracket_html += '</div>'
            
            # 2. Round of 16 column
            bracket_html += '<div class="round"><div class="round-title">Octavos de Final</div>'
            for mid in [89, 90, 91, 92, 93, 94, 95, 96]:
                bracket_html += get_bracket_match_card(mid, f"Partido {mid}")
            bracket_html += '</div>'
            
            # 3. Quarterfinals column
            bracket_html += '<div class="round"><div class="round-title">Cuartos de Final</div>'
            for mid in [97, 98, 99, 100]:
                bracket_html += get_bracket_match_card(mid, f"Partido {mid}")
            bracket_html += '</div>'
            
            # 4. Semifinals column
            bracket_html += '<div class="round"><div class="round-title">Semifinales</div>'
            for mid in [101, 102]:
                bracket_html += get_bracket_match_card(mid, f"Partido {mid}")
            bracket_html += '</div>'
            
            # 5. Finals column
            bracket_html += '<div class="round"><div class="round-title">Finales</div>'
            bracket_html += get_bracket_match_card(104, "Gran Final 🥇")
            bracket_html += get_bracket_match_card(103, "Tercer Puesto 🥉")
            bracket_html += '</div>'
            
            bracket_html += '</div>'
            
            st.components.v1.html(bracket_html, height=850, scrolling=True)

    if "mc_df" in st.session_state:
        df_mc = st.session_state["mc_df"]
        mc_iters = st.session_state["mc_iters"]
        mc_pred = st.session_state["mc_predictor"]
        
        # English to Spanish name dictionary
        eng_to_spanish = {v: k for k, v in SPANISH_TO_ENGLISH.items()}
        eng_to_spanish["Netherlands"] = "Países Bajos"
        eng_to_spanish["Tunisia"] = "Túnez"
        eng_to_spanish["South Korea"] = "República de Corea"
        eng_to_spanish["Czech Republic"] = "República Checa"
        eng_to_spanish["South Africa"] = "Sudáfrica"
        
        st.markdown(f"### 📊 Resultados del Análisis Monte Carlo ({mc_iters} simulaciones)")
        st.write(f"Aquí tienes el análisis consolidado libre de ruido estadístico calculado mediante la simulación de {mc_iters} Mundiales completos usando el modelo **{mc_pred}**:")
        
        # Split Monte Carlo into 2 columns side-by-side
        col_mc_chart, col_mc_table = st.columns(2)
        
        with col_mc_chart:
                # Plot top 10 favorites
            df_top10 = df_mc.head(10).copy()
            df_top10["País"] = df_top10["team"].map(lambda t: eng_to_spanish.get(t, t))
        
            fig_mc = px.bar(
                df_top10,
                x="P_champion",
                y="País",
                orientation='h',
                text_auto='.1%',
                labels={"P_champion": "Probabilidad de Campeonar", "País": "Selección"},
                color="P_champion",
                color_continuous_scale="Viridis",
                title=f"Top 10 Favoritos para Ganar el Mundial 2026 ({mc_pred})"
            )
            fig_mc.update_layout(
                yaxis={'categoryorder':'total ascending'},
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_mc, use_container_width=True)
        
        with col_mc_table:
            # Complete table of probabilities
            st.markdown("#### 📋 Matriz Completa de Probabilidades por Etapa (48 Selecciones)")
            st.write("Explora las probabilidades detalladas de todas las selecciones de avanzar a cada etapa del torneo:")
        
            df_mc_display = df_mc.copy()
            df_mc_display["Selección"] = df_mc_display["team"].map(lambda t: eng_to_spanish.get(t, t))
        
            # Format percentages beautifully
            df_mc_display["Pasa Grupos"] = df_mc_display["P_group_advance"].map(lambda p: f"{p*100:.1f}%")
            df_mc_display["Octavos"] = df_mc_display["P_round_of_16"].map(lambda p: f"{p*100:.1f}%")
            df_mc_display["Cuartos"] = df_mc_display["P_quarters"].map(lambda p: f"{p*100:.1f}%")
            df_mc_display["Semis"] = df_mc_display["P_semis"].map(lambda p: f"{p*100:.1f}%")
            df_mc_display["Final"] = df_mc_display["P_final"].map(lambda p: f"{p*100:.1f}%")
            df_mc_display["Campeón"] = df_mc_display["P_champion"].map(lambda p: f"{p*100:.1f}%")
        
            st.dataframe(
                df_mc_display[["Selección", "Pasa Grupos", "Octavos", "Cuartos", "Semis", "Final", "Campeón"]],
                width='stretch'
            )

# -------------------------------------------------------------
# TAB 4: ACTUALIZACIÓN EN VIVO (LIVE LOGGER)
# -------------------------------------------------------------
with tab4:
    st.header("🔄 Cargar Resultados en Vivo")
    st.write("Registra los marcadores reales del Mundial para recalcular el poder de cada selección.")
    
    # Render all loaded matches at the top inside a collapsible expander
    if loaded_matches:
        with st.expander("📋 Ver Historial de Resultados Cargados (Click para desplegar)", expanded=False):
            st.write("Historial de partidos jugados cargados por los usuarios. Sus resultados persisten globalmente.")
            
            scores_records = []
            for idx, m in enumerate(loaded_matches):
                scores_records.append({
                    "N°": idx + 1,
                    "Local": m["home"],
                    "Marcador": f"{m['home_score']} - {m['away_score']}",
                    "Visitante": m["away"]
                })
            st.dataframe(pd.DataFrame(scores_records), hide_index=True, width='stretch')
        st.markdown("---")
        
    # Fetch available matches dynamically based on current tournament progress
    available_to_load = get_available_matches_to_load(team_features, loaded_matches)
    
    if not available_to_load:
        st.success("🏆 ¡Mundial Completado! Se han cargado los 104 partidos del torneo con éxito global.")
    else:
        selected_match_dict = st.selectbox(
            "Selecciona el Partido a Cargar:",
            available_to_load,
            format_func=lambda x: x["label"],
            key="selected_match_logger"
        )
        
        h_select = selected_match_dict["home"]
        a_select = selected_match_dict["away"]
        
        st.markdown(f"⚽ Partido Seleccionado: **{h_select}** vs **{a_select}**")
        
        # Pre-fill goals automatically if the match has been loaded previously
        default_gh = 0
        default_ga = 0
        if "home_score" in selected_match_dict:
            default_gh = int(selected_match_dict["home_score"])
            default_ga = int(selected_match_dict["away_score"])
            
        col_gh, col_gv = st.columns(2)
        with col_gh:
            g_h_input = st.number_input(f"Goles de {h_select}:", min_value=0, max_value=20, value=default_gh, step=1, key="logger_g_h")
        with col_gv:
            g_a_input = st.number_input(f"Goles de {a_select}:", min_value=0, max_value=20, value=default_ga, step=1, key="logger_g_a")
            
        if st.button("💾 Cargar y Recalcular ELO Oficial"):
            h_eng = spanish_to_english_web[h_select]
            a_eng = spanish_to_english_web[a_select]
            
            # Fetch current stats
            fa = team_features[h_eng]
            fb = team_features[a_eng]
            
            elo_a = fa['elo']
            elo_b = fb['elo']
            
            # ELO Formula
            dr = elo_a - elo_b
            expected_a = 1 / (1 + 10 ** (-dr / 400))
            expected_b = 1.0 - expected_a
            
            actual_a = 1.0 if g_h_input > g_a_input else (0.5 if g_h_input == g_a_input else 0.0)
            actual_b = 1.0 - actual_a
            
            # Goal difference multiplier
            gd = abs(g_h_input - g_a_input)
            if gd <= 1:
                G = 1.0
            elif gd == 2:
                G = 1.5
            else:
                G = 1.75 + (gd - 3) / 8.0
                
            K = 40.0 # High multiplier for World Cup matches
            
            new_elo_a = elo_a + K * G * (actual_a - expected_a)
            new_elo_b = elo_b + K * G * (actual_b - expected_b)
            
            # Update team_features
            team_features[h_eng]['elo'] = new_elo_a
            team_features[a_eng]['elo'] = new_elo_b
            
            # Update history / form (approximate list-updating)
            if 'history' not in team_features[h_eng]:
                team_features[h_eng]['history'] = [(float(fa['form_pts']), float(fa['recent_goals']))] * 5
            if 'history' not in team_features[a_eng]:
                team_features[a_eng]['history'] = [(float(fb['form_pts']), float(fb['recent_goals']))] * 5
                
            pts_a = 3 if g_h_input > g_a_input else (1 if g_h_input == g_a_input else 0)
            pts_b = 3 if g_a_input > g_h_input else (1 if g_h_input == g_a_input else 0)
            
            team_features[h_eng]['history'].append((float(pts_a), float(g_h_input)))
            team_features[a_eng]['history'].append((float(pts_b), float(g_a_input)))
            
            # Recalculate averages
            team_features[h_eng]['form_pts'] = sum(x[0] for x in team_features[h_eng]['history'][-5:]) / 5.0
            team_features[h_eng]['recent_goals'] = sum(x[1] for x in team_features[h_eng]['history'][-5:]) / 5.0
            
            team_features[a_eng]['form_pts'] = sum(x[0] for x in team_features[a_eng]['history'][-5:]) / 5.0
            team_features[a_eng]['recent_goals'] = sum(x[1] for x in team_features[a_eng]['history'][-5:]) / 5.0
            
            # Filter out previous record of this match to enable dynamic overwrite (avoiding duplicates!)
            if selected_match_dict["type"] == "knockout":
                loaded_matches = [m for m in loaded_matches if m.get("match_id") != selected_match_dict["match_id"]]
            else:
                target_pair = frozenset({h_select, a_select})
                loaded_matches = [m for m in loaded_matches if frozenset({m["home"], m["away"]}) != target_pair]
                
            # Append newly updated/created match log
            new_match_log = {
                "home": h_select,
                "away": a_select,
                "home_score": int(g_h_input),
                "away_score": int(g_a_input)
            }
            if selected_match_dict["type"] == "knockout":
                new_match_log["match_id"] = selected_match_dict["match_id"]
            loaded_matches.append(new_match_log)
            
            # Save to live state file
            live_dir = Path('live')
            live_dir.mkdir(exist_ok=True)
            with open(live_dir / 'streamlit_live_state.pkl', 'wb') as f:
                pickle.dump({
                    'team_features': team_features,
                    'WC_TEAMS': WC_TEAMS,
                    'loaded_matches': loaded_matches
                }, f)
                
            # Clear Cache & reload
            st.cache_data.clear()
            st.success(f"¡Resultados guardados con éxito! ELO actualizado: {h_select} {int(elo_a)} ➜ {int(new_elo_a)} | {a_select} {int(elo_b)} ➜ {int(new_elo_b)}. Panel recargado.")

