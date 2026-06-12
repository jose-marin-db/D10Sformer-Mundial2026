# D10Sformer

> Temporal Relational Transformer for football match prediction.
> Final project for **MIA305 — NLP** (Universidad de San Andrés, 2026).

---

## Hipótesis central

El resultado de un partido depende más de **relaciones contextuales temporales** entre jugadores que de la identidad histórica del equipo. Por eso modelamos el fútbol como un problema de **embeddings contextuales** — la misma idea que motivó el salto desde Word2Vec hacia BERT en NLP.

> Messi 2012 ≠ Messi 2025
> Argentina 2014 ≠ Argentina 2022

El modelo aprende `estado(jugador, tiempo, contexto)` en lugar de `identidad_estatica(jugador)`.

---

## Arquitectura

**Transformer Encoder** (BERT-like, no autorregresivo) sobre secuencias de eventos futbolísticos tokenizados. Entrenado con:

- Clasificación multi-clase del resultado (W/D/L)
- Regresión auxiliar de goles
- Masked Language Modeling auxiliar (regularización semántica)
- Masking estocástico de features → robustez ante información parcial

Inferencia del Mundial 2026 vía **Monte Carlo rollout** sobre el bracket completo.

---

## Estructura del proyecto

```
d10sformer/
├── data/                # Datos (no commiteados)
│   ├── raw/             # StatsBomb Open Data
│   ├── interim/         # Parsed
│   └── processed/       # Tokenized
├── notebooks/           # Análisis y experimentos
├── src/
│   ├── data/            # Loaders, feature engineering, tokenizer
│   ├── models/          # Embeddings, transformer, baselines, heads
│   ├── training/        # Dataset, masking, train loop
│   ├── eval/            # Metrics (Brier, ECE), calibration
│   └── simulation/      # Tournament Monte Carlo
├── tests/               # Pytest
├── configs/             # YAML configs
├── reports/             # Paper final, figuras
└── scripts/             # Scripts ejecutables
```

---

## Setup

### En Colab Pro (recomendado)

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/d10sformer
!pip install -r requirements.txt
```

### En local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Datos

Usamos **StatsBomb Open Data**, que cubre Mundiales 2018, 2022 y ligas top. Para clonarla:

```bash
# Dentro de d10sformer/data/raw/
git clone https://github.com/statsbomb/open-data.git statsbomb
```

(No se commitea — está en `.gitignore`.)

---

## Roadmap

Ver [`../plan_de_implementacion.md`](../plan_de_implementacion.md) para el plan completo en 8 fases.

| Fase | Estado |
|---|---|
| 0. Setup + EDA | 🚧 en curso |
| 1. Baselines tabulares | pending |
| 2. Tokenización | pending |
| 3. Transformer | pending |
| 4. Entrenamiento | pending |
| 5. Eval + calibración | pending |
| 6. Simulación Mundial | pending |
| 7. Pipeline live | pending |

---

## Referencias clave

- Vaswani et al. (2017). *Attention is All You Need.*
- Devlin et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers.*
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
- Jurafsky & Martin. *Speech and Language Processing* (3rd ed.).
- Eisenstein. *Introduction to Natural Language Processing.*

---

## Autoría

Proyecto final MIA305 — Universidad de San Andrés (2026).
