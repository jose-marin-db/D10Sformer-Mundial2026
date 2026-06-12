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

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: 900;
        color: #10B981;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
        font-weight: 600;
        text-transform: uppercase;
    }
    .stTabs [data-basetabs="tablist"] {
        justify-content: center;
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
        return state['team_features'], state['WC_TEAMS']

    DATA_INTERIM = Path('interim')
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
            
    return team_features, WC_TEAMS

# Load cached items
try:
    model, vocab, tourn_map = load_transformer_model()
    model_v1 = load_d10sformer_v1(vocab)
    team_features, WC_TEAMS = load_historical_data_and_features()
except Exception as e:
    st.error(f"Error loading models or dataset. Make sure you trained the model first: {e}")
    st.stop()

def p_to_score(p_home, p_draw, p_away, base_total=2.8):
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
    goals_a = round(rate_a)
    
    # Enforce favorite has more goals
    if p_home > p_away and goals_h <= goals_a:
        goals_h = goals_a + 1
    elif p_away > p_home and goals_a <= goals_h:
        goals_a = goals_h + 1
        
    # Inject variance for major favorites
    if p_home > 0.70 and goals_h < 2:
        goals_h = 2
    if p_home > 0.85 and goals_h < 3:
        goals_h = 3
        
    if p_away > 0.70 and goals_a < 2:
        goals_a = 2
    if p_away > 0.85 and goals_a < 3:
        goals_a = 3
        
    return goals_h, goals_a

# Sidebar Settings
st.sidebar.header("⚙️ Configuración del Predictor")
prediction_mode = st.sidebar.radio(
    "Modelo de Goles (Marcadores):",
    ["Directo del Transformer (Conservador)", "Simulación Poisson (Realista y Goleador)"],
    index=1 # Default to exciting Poisson simulation
)
base_total_goals = st.sidebar.slider(
    "Promedio de Goles por Partido:",
    min_value=1.5, max_value=4.5, value=2.8, step=0.1
)

# Title section
st.markdown('<div class="main-title">⚽ D10Sformer v2 ⚽</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Feature Tokenizer Transformer (FT-Transformer) — Tabular Deep Learning Dashboard</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Main Tabs Setup
# -------------------------------------------------------------
tab1, tab2, tab_tourn, tab3, tab4, tab5 = st.tabs([
    "🏆 Predicciones de Fechas (Prode)",
    "⚔️ Simulador de Partido Único",
    "🎮 Simulador del Mundial Completo",
    "📈 Rankings de Fuerza (ELO)",
    "🔄 Actualización de Resultados Reales",
    "🧠 Simulador de Alineaciones (v1)"
])

# -------------------------------------------------------------
# TAB 1: PREDICCIONES FASE 1
# -------------------------------------------------------------
with tab1:
    st.header("Predicciones Oficiales de la Fase 1 del Mundial 2026")
    st.write("Visualiza los marcadores exactos calculados por el cabezal multi-tarea del **D10Sformer** original sobre los 24 partidos inaugurales:")
    
    # Load JSON output
    try:
        with open("d10sformer_fase1_predictions_output.json") as f:
            predictions_fase1 = json.load(f)
            
        # Structure into DataFrame for clean display
        records = []
        for p in predictions_fase1:
            if prediction_mode == "Simulación Poisson (Realista y Goleador)":
                g_h, g_a = p_to_score(p["p_home"], p["p_draw"], p["p_away"], base_total=base_total_goals)
                rec_score = f"{g_h} - {g_a}"
                conf_score = "Calibrado (Poisson)"
            else:
                rec_score = f"{p['score_h']} - {p['score_a']}"
                conf_score = f"{p['p_score']*100:.1f}%"
                
            records.append({
                "Grupo": p["group"],
                "Local (Home)": p["home_web"],
                "Visitante (Away)": p["away_web"],
                "Gana Local": f"{p['p_home']*100:.1f}%",
                "Empate": f"{p['p_draw']*100:.1f}%",
                "Gana Visita": f"{p['p_away']*100:.1f}%",
                "Marcador Recomendado": rec_score,
                "Confianza / Tipo": conf_score
            })
        df_pred = pd.DataFrame(records)
        
        st.dataframe(
            df_pred,
            width='stretch',
            column_config={
                "Grupo": st.column_config.TextColumn(width="medium"),
                "Local (Home)": st.column_config.TextColumn(width="medium"),
                "Visitante (Away)": st.column_config.TextColumn(width="medium"),
                "Marcador Recomendado": st.column_config.TextColumn(width="medium")
            },
            hide_index=True
        )
    except FileNotFoundError:
        st.warning("No se encontró el archivo de predicciones de Fase 1. Asegúrate de haber ejecutado 'predict_d10sformer_matchday1.py'.")

# -------------------------------------------------------------
# TAB 2: SIMULADOR INTERACTIVO (FT-TRANSFORMER V2)
# -------------------------------------------------------------
with tab2:
    st.header("Simulador de Partidos con FT-Transformer v2")
    st.write("Selecciona cualquier emparejamiento, ajusta las variables continuas de alta resolución y observa cómo el Transformer recalcula la distribución de goles en tiempo real:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuración del Encuentro")
        
        # Team selections (map Spanish name on page to English name in model)
        spanish_to_english_web = SPANISH_TO_ENGLISH.copy()
        # Add names missing from standard dict
        spanish_to_english_web["Paises Bajos"] = "Netherlands"
        spanish_to_english_web["Tunez"] = "Tunisia"
        spanish_to_english_web["Iraq"] = "Iraq"
        spanish_to_english_web["República de Corea"] = "South Korea"
        
        web_spanish_names = sorted(list(spanish_to_english_web.keys()))
        
        h_web_select = st.selectbox("Selección Local (Team A):", web_spanish_names, index=web_spanish_names.index("Argentina"))
        a_web_select = st.selectbox("Selección Visitante (Team B):", web_spanish_names, index=web_spanish_names.index("Francia"))
        
        h_eng = spanish_to_english_web[h_web_select]
        a_eng = spanish_to_english_web[a_web_select]
        
        # Retrieve default stats from team history
        feat_a = team_features.get(h_eng, {'elo': 1500, 'form_pts': 1.0, 'recent_goals': 1.0, 'form10_pts': 1.0})
        feat_b = team_features.get(a_eng, {'elo': 1500, 'form_pts': 1.0, 'recent_goals': 1.0, 'form10_pts': 1.0})
        
        # Override stats with sliders inside expander
        with st.expander("🛠️ Modo Dios / Ajuste Manual de Stats"):
            # Team A Sliders
            st.markdown(f"**📊 {h_web_select}:**")
            h_elo = st.slider("ELO Rating A:", min_value=1000.0, max_value=2500.0, value=float(feat_a['elo']), step=10.0, key=f"h_elo_{h_eng}")
            h_form = st.slider("Forma Pts A (Últimos 5):", min_value=0.0, max_value=3.0, value=float(feat_a['form_pts']), step=0.1, key=f"h_form_{h_eng}")
            h_goals = st.slider("Goles Recientes A (Promedio):", min_value=0.0, max_value=5.0, value=float(feat_a['recent_goals']), step=0.1, key=f"h_goals_{h_eng}")
            
            # Team B Sliders
            st.markdown(f"**📊 {a_web_select}:**")
            a_elo = st.slider("ELO Rating B:", min_value=1000.0, max_value=2500.0, value=float(feat_b['elo']), step=10.0, key=f"a_elo_{a_eng}")
            a_form = st.slider("Forma Pts B (Últimos 5):", min_value=0.0, max_value=3.0, value=float(feat_b['form_pts']), step=0.1, key=f"a_form_{a_eng}")
            a_goals = st.slider("Goles Recientes B (Promedio):", min_value=0.0, max_value=5.0, value=float(feat_b['recent_goals']), step=0.1, key=f"a_goals_{a_eng}")
        
        # Categoricals override
        st.markdown("**Configuración del Entorno:**")
        tournament_sel = st.selectbox("Clase de Torneo:", list(tourn_map.keys()), index=list(tourn_map.keys()).index("FIFA World Cup") if "FIFA World Cup" in tourn_map else 0)
        venue_sel = st.selectbox("Localía del Equipo A (Venue):", ["Home", "Away", "Neutral"], index=2) # default to Neutral
        
        # Map selected values
        t_class_idx = tourn_map[tournament_sel]
        venue_idx = 0 if venue_sel == "Home" else (1 if venue_sel == "Away" else 2)
        neutral_val = 1 if venue_sel == "Neutral" else 0
        
    with col2:
        st.subheader("🔮 Predicciones de Inferencia en Tiempo Real")
        
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
        
        # Split output into side-by-side columns
        col_comp_v2, col_comp_v1 = st.columns(2)
        
        with col_comp_v2:
            st.markdown("<h4 style='text-align: center; color: #1E3A8A;'>🏆 FT-Transformer (v2)</h4>", unsafe_allow_html=True)
            if prediction_mode == "Simulación Poisson (Realista y Goleador)":
                rec_h, rec_a = p_to_score(p_home, p_draw, p_away, base_total=base_total_goals)
                rec_prob_str = "Modelo Calibrado (Poisson)"
            else:
                best_score_idx = np.argmax(probs)
                rec_h = best_score_idx // 6
                rec_a = best_score_idx % 6
                rec_prob_str = f"Probabilidad: {probs[best_score_idx]*100:.1f}%"
                
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 1.5rem;">
                <div class="metric-label">Marcador Recomendado (v2)</div>
                <div class="metric-value" style="font-size: 2.2rem; color: #3B82F6;">{rec_h} - {rec_a}</div>
                <div style="font-weight: 700; color: #6B7280; font-size: 0.8rem;">{rec_prob_str}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Donut Chart v2
            labels_v2 = [f"Ganador A", "Empate", f"Ganador B"]
            values_v2 = [p_home, p_draw, p_away]
            colors_v2 = ['#10B981', '#F59E0B', '#EF4444']
            fig_v2 = go.Figure(data=[go.Pie(labels=labels_v2, values=values_v2, hole=.4, marker_colors=colors_v2)])
            fig_v2.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig_v2, use_container_width=True)
            
        with col_comp_v1:
            st.markdown("<h4 style='text-align: center; color: #4B5563;'>🧠 D10Sformer (v1)</h4>", unsafe_allow_html=True)
            if prediction_mode == "Simulación Poisson (Realista y Goleador)":
                rec_h_v1, rec_a_v1 = p_to_score(p_h_v1, p_d_v1, p_a_v1, base_total=base_total_goals)
                rec_prob_v1_str = "Modelo Calibrado (Poisson)"
            else:
                best_score_v1_idx = np.argmax(score_v1_probs)
                rec_h_v1 = best_score_v1_idx // 6
                rec_a_v1 = best_score_v1_idx % 6
                rec_prob_v1_str = f"Probabilidad: {score_v1_probs[best_score_v1_idx]*100:.1f}%"
                
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 1.5rem;">
                <div class="metric-label">Marcador Recomendado (v1)</div>
                <div class="metric-value" style="font-size: 2.2rem; color: #6B7280;">{rec_h_v1} - {rec_a_v1}</div>
                <div style="font-weight: 700; color: #6B7280; font-size: 0.8rem;">{rec_prob_v1_str}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Donut Chart v1
            labels_v1 = [f"Ganador A", "Empate", f"Ganador B"]
            values_v1 = [p_h_v1, p_d_v1, p_a_v1]
            colors_v1 = ['#10B981', '#F59E0B', '#EF4444']
            fig_v1 = go.Figure(data=[go.Pie(labels=labels_v1, values=values_v1, hole=.4, marker_colors=colors_v1)])
            fig_v1.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig_v1, use_container_width=True)

# -------------------------------------------------------------
# TAB TOURN: SIMULADOR DE MUNDIAL COMPLETO
# -------------------------------------------------------------
with tab_tourn:
    st.header("🎮 Simulador de Torneo Completo (Monte Carlo)")
    st.write("¡Simula un Mundial completo en tiempo real! Corre el modelo estocástico de principio a fin, clasifica a los mejores terceros, resuelve las llaves de playoffs y corona al campeón de forma dinámica con marcadores exactos de los 104 partidos para armar tus prodes.")
    
    col_sim_cfg, col_sim_act = st.columns([1, 2])
    with col_sim_cfg:
        st.subheader("Configuración")
        sim_predictor_choice = st.selectbox(
            "Modelo Predictor del Prode:",
            ["FT-Transformer (v2)", "D10Sformer (v1)"],
            index=0,
            key="sim_pred_choice_widget"
        )
        
        sim_goals_mode = st.selectbox(
            "Criterio de Goles / Marcadores:",
            ["Directo del Transformer (Conservador)", "Simulación Poisson (Realista/Goleador)"],
            index=1,
            key="sim_goals_widget"
        )
        
        st.markdown("---")
        st.markdown("**🎲 Opción 1: Generador de Prode (Timeline de Azar)**")
        run_sim = st.button("🏁 Simular 1 Mundial Completo", type="primary", use_container_width=True)
        st.caption("Corre un único Mundial stochásticamente. Excelente para obtener una combinación divertida de marcadores exactos y llaves de playoffs, con un alto nivel de sorpresas (como en la vida real).")
        
        st.markdown("---")
        st.markdown("**📊 Opción 2: Análisis Estadístico (Fiabilidad)**")
        mc_iters = st.slider("Iteraciones Monte Carlo:", min_value=50, max_value=1000, value=200, step=50, key="sim_mc_iters_widget")
        run_mc = st.button("📈 Correr Análisis Monte Carlo", type="secondary", use_container_width=True)
        st.caption("Simula cientos de Mundiales en segundos de forma stochástica y consolida los resultados para darnos el verdadero ranking científico de favoritos sin el ruido del azar.")
        
    with col_sim_act:
        st.subheader("🔍 Estado de los Datos y Fiabilidad")
        st.write(
            "⚽ **¿Por qué varía el Campeón cada vez que simulo un solo Mundial?**\n"
            "Porque el fútbol es intrínsecamente impredecible. El modelo predice **probabilidades**, no verdades absolutas. "
            "Cuando simulas un Mundial único, cada partido se define por azar ponderado. Esto puede generar sorpresas increíbles (como en la vida real, donde Grecia ganó la Eurocopa o Arabia Saudita le ganó a Argentina).\n\n"
            "📊 **Para obtener predicciones FIABLES (Análisis Monte Carlo):**\n"
            "Utiliza la **Opción 2** para simular el Mundial 200 o 500 veces de corrido. Al promediar los resultados, "
            "el ruido estadístico desaparece y emerge el **verdadero y robusto favoritismo matemático** de cada país."
        )
        if "sim_result" in st.session_state:
            st.success(f"Mostrando simulación anterior (Semilla: {st.session_state['sim_seed']}, Modelo: {st.session_state['sim_predictor']})")
        elif "mc_df" in st.session_state:
            st.success(f"Mostrando análisis Monte Carlo anterior ({st.session_state['mc_iters']} iteraciones, Modelo: {st.session_state['mc_predictor']})")
            
    if run_sim:
        with st.spinner("Simulando fase de grupos (72 partidos), calculando goles y resolviendo playoffs (32 partidos)..."):
            import random
            from simulation.simulator import GroupStanding, select_best_thirds, assign_thirds_to_slots, sample_match_result, sample_knockout_winner, sample_goals, TournamentResult
            from simulation.bracket import ROUND_OF_32, ROUND_OF_16, QUARTERFINALS, SEMIFINALS, THIRD_PLACE, FINAL, ALL_KNOCKOUT_MATCHES, round_of_match, STAGES, REACHED_AT_LEAST
            
            seed = random.randint(1, 100000)
            rng = np.random.default_rng(seed)
            
            # 1. Predictor Callable
            def current_predictor(a_eng, b_eng):
                feat_a = team_features.get(a_eng, {'elo': 1500, 'form_pts': 1.0, 'recent_goals': 1.0})
                feat_b = team_features.get(b_eng, {'elo': 1500, 'form_pts': 1.0, 'recent_goals': 1.0})
                
                if sim_predictor_choice == "FT-Transformer (v2)":
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
                        outcome = sample_match_result(p_h, p_d, p_a, rng)
                        
                        # Dynamic goal generation
                        if sim_goals_mode == "Simulación Poisson (Realista/Goleador)":
                            ga_h, ga_a = p_to_score(p_h, p_d, p_a, base_total=base_total_goals)
                        else:
                            ga_h, ga_a = sample_goals(p_h, p_d, p_a, rng=rng)
                            
                        # Enforce outcome consistency
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
                outcome = sample_knockout_winner(p_h, p_d, p_a, rng)
                
                # Dynamic goals generation
                if sim_goals_mode == "Simulación Poisson (Realista/Goleador)":
                    ga_h, ga_a = p_to_score(p_h, p_d, p_a, base_total=base_total_goals)
                else:
                    ga_h, ga_a = sample_goals(p_h, p_d, p_a, rng=rng)
                    
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
            
            st.balloons()
            st.snow()
            st.success(f"¡Mundial simulado con éxito! Semilla estocástica: {seed}")
            
    if run_mc:
        with st.spinner(f"Corriendo {mc_iters} simulaciones Monte Carlo completas del Mundial..."):
            import random
            from simulation.simulator import monte_carlo
            
            seed = random.randint(1, 100000)
            rng = np.random.default_rng(seed)
            
            # Predictor Callable
            def current_predictor(a_eng, b_eng):
                feat_a = team_features.get(a_eng, {'elo': 1500, 'form_pts': 1.0, 'recent_goals': 1.0})
                feat_b = team_features.get(b_eng, {'elo': 1500, 'form_pts': 1.0, 'recent_goals': 1.0})
                
                if sim_predictor_choice == "FT-Transformer (v2)":
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
                    
            mc_result = monte_carlo(current_predictor, n_iters=mc_iters, seed=seed)
            df_mc = mc_result.to_dataframe()
            
            st.session_state["mc_df"] = df_mc
            st.session_state["mc_iters"] = mc_iters
            st.session_state["mc_predictor"] = sim_predictor_choice
            st.session_state["mc_seed"] = seed
            
            # Clear single simulation results when running Monte Carlo
            if "sim_result" in st.session_state:
                del st.session_state["sim_result"]
                
            st.success(f"¡Análisis Monte Carlo de {mc_iters} simulaciones finalizado con éxito!")
            
    if "sim_result" in st.session_state:
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
        
        # Expander for Group Standings AND match scores
        with st.expander("📊 Ver Posiciones y Marcadores de la Fase de Grupos (A - L)"):
            st.write("¡Revisa las posiciones finales de cada grupo junto con los **marcadores exactos de los 72 partidos** de la primera etapa para armar tu prode!")
            
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
                    
        # Expander for Playoffs Tree Diagram
        from simulation.bracket import ROUND_OF_32, ROUND_OF_16, QUARTERFINALS, SEMIFINALS, THIRD_PLACE, FINAL
        
        with st.expander("⚔️ Ver Cuadro y Llaves del Mundial (Formato Árbol)", expanded=True):
            st.markdown("### 🏆 Diagrama de Eliminación Directa del Mundial 2026")
            st.write("Sigue el flujo de las llaves oficiales de la FIFA de izquierda a derecha. Cada partido destaca al ganador clasificado y muestra los **marcadores exactos** del enfrentamiento:")
            
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
                    background-color: #0F172A; /* Modern dark Slate background */
                    padding: 1.5rem;
                    border-radius: 12px;
                    overflow-x: auto;
                    gap: 1.5rem;
                    border: 1px solid #1E293B;
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
                    color: #38BDF8;
                    font-weight: 800;
                    font-size: 0.85rem;
                    margin-bottom: 1rem;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    border-bottom: 2px solid #334155;
                    padding-bottom: 0.5rem;
                }
                .bracket-match {
                    background: #1E293B;
                    border: 1px solid #334155;
                    border-radius: 8px;
                    padding: 0.6rem;
                    margin: 0.4rem 0;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                    transition: transform 0.2s;
                }
                .bracket-match:hover {
                    border-color: #38BDF8;
                    transform: scale(1.02);
                }
                .match-header {
                    font-size: 0.7rem;
                    color: #64748B;
                    margin-bottom: 0.3rem;
                    font-weight: bold;
                    text-transform: uppercase;
                }
                .bracket-team {
                    display: flex;
                    justify-content: space-between;
                    padding: 0.15rem 0;
                    font-size: 0.8rem;
                    font-weight: 600;
                    color: #E2E8F0;
                }
                .bracket-team.winner {
                    color: #10B981; /* Green for winner */
                    font-weight: bold;
                }
                .bracket-team.loser {
                    color: #64748B; /* Darker slate for loser */
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
# TAB 3: RANKING DE FUERZA E INSIGHTS (XAI)
# -------------------------------------------------------------
with tab3:
    st.header("Líderes ELO del Mundial 2026")
    st.write("Líderes absolutos calculados mediante la integración histórica de partidos desde 1872:")
    
    # Sort teams by ELO
    lead_records = []
    for t, feat in team_features.items():
        # Get Spanish name
        spa_name = SPANISH_TO_ENGLISH.get(t, t)
        lead_records.append({
            "Selección": spa_name,
            "ELO Rating": int(feat["elo"]),
            "Forma Reciente (Goles/Partidos)": f"{feat['recent_goals']:.2f}",
            "Puntos Forma (Últimos 5)": f"{feat['form_pts']:.2f}"
        })
    df_leaderboard = pd.DataFrame(lead_records).sort_values("ELO Rating", ascending=False).reset_index(drop=True)
    df_leaderboard.index += 1
    
    st.dataframe(df_leaderboard, width='stretch')

# -------------------------------------------------------------
# TAB 4: ACTUALIZACIÓN EN VIVO (LIVE LOGGER)
# -------------------------------------------------------------
with tab4:
    st.header("🔄 Cargar Resultados Oficiales en Vivo")
    st.write("Registra los resultados reales del Mundial 2026. El sistema recalculará automáticamente el ELO, forma y promedio de goles de ambos equipos, actualizando todo el panel de forma interactiva:")
    
    col_l, col_v = st.columns(2)
    with col_l:
        h_select = st.selectbox("Selección Local (A):", web_spanish_names, key="live_h", index=web_spanish_names.index("México"))
    with col_v:
        a_select = st.selectbox("Selección Visitante (B):", web_spanish_names, key="live_a", index=web_spanish_names.index("Sudáfrica"))
        
    col_gh, col_gv = st.columns(2)
    with col_gh:
        g_h_input = st.number_input(f"Goles de {h_select}:", min_value=0, max_value=20, value=0, step=1)
    with col_gv:
        g_a_input = st.number_input(f"Goles de {a_select}:", min_value=0, max_value=20, value=0, step=1)
        
    if st.button("💾 Cargar y Recalcular ELO Oficial"):
        if h_select == a_select:
            st.error("Por favor, selecciona dos selecciones distintas.")
        else:
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
            
            # Save to live state file
            live_dir = Path('live')
            live_dir.mkdir(exist_ok=True)
            with open(live_dir / 'streamlit_live_state.pkl', 'wb') as f:
                pickle.dump({'team_features': team_features, 'WC_TEAMS': WC_TEAMS}, f)
                
            # Clear Cache & reload
            st.cache_data.clear()
            st.success(f"¡Resultados guardados con éxito! ELO actualizado: {h_select} {int(elo_a)} ➜ {int(new_elo_a)} | {a_select} {int(elo_b)} ➜ {int(new_elo_b)}. Panel recargado.")

# -------------------------------------------------------------
# TAB 5: SIMULADOR DE ALINEACIONES (v1)
# -------------------------------------------------------------
@st.cache_data
def extract_team_rosters():
    # Build team rosters backward
    with open('processed/corpus/pretrain.pkl', 'rb') as f:
        train_docs = pickle.load(f)
        
    rosters = defaultdict(list)
    seen_players = defaultdict(set)
    
    for m in reversed(train_docs):
        # Lineup A
        if m.lineup_a:
            for p in m.lineup_a:
                pid_str = str(p.player_id)
                if pid_str not in seen_players[m.team_a]:
                    p_info = vocab.player_info.get(pid_str)
                    name = p_info.name if p_info else f"Jugador {pid_str}"
                    rosters[m.team_a].append({
                        'id': pid_str,
                        'name': name,
                        'position': p.position
                    })
                    seen_players[m.team_a].add(pid_str)
        # Lineup B
        if m.lineup_b:
            for p in m.lineup_b:
                pid_str = str(p.player_id)
                if pid_str not in seen_players[m.team_b]:
                    p_info = vocab.player_info.get(pid_str)
                    name = p_info.name if p_info else f"Jugador {pid_str}"
                    rosters[m.team_b].append({
                        'id': pid_str,
                        'name': name,
                        'position': p.position
                    })
                    seen_players[m.team_b].add(pid_str)
                    
    # Sort rosters by name
    for team in rosters:
        rosters[team] = sorted(rosters[team], key=lambda x: x['name'])
        
    return rosters

with tab5:
    st.header("🧠 Simulador de Alineaciones (D10Sformer v1)")
    st.write("Aprovecha los embeddings semánticos profundos de los jugadores entrenados en el D10Sformer v1. Selecciona las alineaciones titulares exactas para ver cómo varía la probabilidad de marcador:")
    
    rosters = extract_team_rosters()
    
    col_la, col_lb = st.columns(2)
    
    with col_la:
        h_v1_select = st.selectbox("Selección Local (A):", web_spanish_names, key="v1_h", index=web_spanish_names.index("Argentina"))
        h_v1_eng = spanish_to_english_web[h_v1_select]
        roster_a = rosters.get(h_v1_eng, [])
        
        st.markdown(f"**Alineación Titular {h_v1_select}:**")
        player_options_a = [f"[{p['position']}] {p['name']} (ID: {p['id']})" for p in roster_a]
        selected_player_strings_a = st.multiselect(
            "Selecciona los jugadores titulares de A (por defecto se asumen fallbacks):",
            player_options_a,
            default=player_options_a[:11] if len(player_options_a) >= 11 else player_options_a,
            key=f"v1_lineup_{h_v1_eng}"
        )
        
    with col_lb:
        a_v1_select = st.selectbox("Selección Visitante (B):", web_spanish_names, key="v1_b", index=web_spanish_names.index("Francia"))
        a_v1_eng = spanish_to_english_web[a_v1_select]
        roster_b = rosters.get(a_v1_eng, [])
        
        st.markdown(f"**Alineación Titular {a_v1_select}:**")
        player_options_b = [f"[{p['position']}] {p['name']} (ID: {p['id']})" for p in roster_b]
        selected_player_strings_b = st.multiselect(
            "Selecciona los jugadores titulares de B (por defecto se asumen fallbacks):",
            player_options_b,
            default=player_options_b[:11] if len(player_options_b) >= 11 else player_options_b,
            key=f"v1_lineup_{a_v1_eng}"
        )
        
    # Helper to parse multi-select back to players
    def get_selected_players(selected_strings, roster):
        selected_pids = []
        for s in selected_strings:
            part = s.split("ID: ")[1][:-1]
            selected_pids.append(part)
        return [p for p in roster if p['id'] in selected_pids]
        
    sel_players_a = get_selected_players(selected_player_strings_a, roster_a)
    sel_players_b = get_selected_players(selected_player_strings_b, roster_b)
    
    # Process through Tokenizer and D10Sformer v1
    from data.tokenizer import MatchTokenizer, MatchDocument, PlayerRef, RollingFeatures
    tokenizer_v1 = MatchTokenizer(vocab, max_seq_length=80)
    
    fa_v1 = team_features.get(h_v1_eng, {'elo': 1500, 'form_pts': 1.0, 'recent_goals': 1.0})
    fb_v1 = team_features.get(a_v1_eng, {'elo': 1500, 'form_pts': 1.0, 'recent_goals': 1.0})
    
    # Construct lineups PlayerRef lists
    lineup_refs_a = [PlayerRef(p['id'], p['position']) for p in sel_players_a] if sel_players_a else None
    lineup_refs_b = [PlayerRef(p['id'], p['position']) for p in sel_players_b] if sel_players_b else None
    
    doc_v1 = MatchDocument(
        tournament="FIFA World Cup",
        team_a=h_v1_eng,
        team_b=a_v1_eng,
        venue="neutral",
        stage=None,
        lineup_a=lineup_refs_a,
        lineup_b=lineup_refs_b,
        features=RollingFeatures(
            home_elo=fa_v1['elo'],
            away_elo=fb_v1['elo'],
            home_form_pts=fa_v1['form_pts'],
            away_form_pts=fb_v1['form_pts'],
            home_recent_goals=fa_v1['recent_goals'],
            away_recent_goals=fb_v1['recent_goals']
        )
    )
    
    out_v1 = tokenizer_v1.tokenize(doc_v1)
    
    tok_v1_tensor = torch.tensor([out_v1.token_ids], dtype=torch.long)
    seg_v1_tensor = torch.tensor([out_v1.segment_ids], dtype=torch.long)
    
    with torch.no_grad():
        out_model_v1 = model_v1(tok_v1_tensor, seg_v1_tensor)
        
    res_v1_probs = F.softmax(out_model_v1["result_logits"], dim=-1)[0].numpy()
    score_v1_probs = F.softmax(out_model_v1["score_logits"], dim=-1)[0].numpy()
    
    p_h_v1, p_d_v1, p_a_v1 = res_v1_probs[0], res_v1_probs[1], res_v1_probs[2]
    
    # Render Output Layout
    col_out1, col_out2 = st.columns(2)
    with col_out1:
        labels_v1 = [f"Victoria {h_v1_select}", "Empate", f"Victoria {a_v1_select}"]
        values_v1 = [p_h_v1, p_d_v1, p_a_v1]
        colors_v1 = ['#10B981', '#F59E0B', '#EF4444']
        
        fig_v1 = go.Figure(data=[go.Pie(labels=labels_v1, values=values_v1, hole=.4, marker_colors=colors_v1)])
        fig_v1.update_layout(
            title_text="Distribución del Resultado (Alineación Seleccionada)",
            annotations=[dict(text='Resultado', x=0.5, y=0.5, font_size=16, showarrow=False)],
            margin=dict(l=20, r=20, t=40, b=20),
            height=300
        )
        st.plotly_chart(fig_v1, use_container_width=True)
        
    with col_out2:
        if prediction_mode == "Simulación Poisson (Realista y Goleador)":
            v1_h, v1_a = p_to_score(p_h_v1, p_d_v1, p_a_v1, base_total=base_total_goals)
            v1_prob_str = "Modelo Calibrado (Poisson)"
        else:
            best_v1_idx = np.argmax(score_v1_probs)
            v1_h = best_v1_idx // 6
            v1_a = best_v1_idx % 6
            v1_prob_str = f"Probabilidad: {score_v1_probs[best_v1_idx]*100:.1f}%"
        
        st.markdown(f"""
        <div style="display: flex; flex-direction: column; justify-content: center; height: 100%;">
            <div class="metric-card" style="margin-top: 2rem;">
                <div class="metric-label">Marcador Recomendado (v1 con Alineaciones)</div>
                <div class="metric-value">{v1_h} - {v1_a}</div>
                <div style="font-weight: 700; color: #6B7280; font-size: 1.2rem;">{v1_prob_str}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
