# Fase 0 — Hallazgos del EDA y Decisiones de Diseño

**Fecha:** 16 de mayo de 2026
**Notebook:** `notebooks/00_eda.ipynb`
**Fuente:** StatsBomb Open Data clonado el 16/05/2026 (~6.6 GB)

---

## Resumen ejecutivo

El dataset es **suficiente en tamaño y cobertura** para entrenar el Transformer, pero el EDA reveló dos problemas estructurales que **cambian decisiones del diseño original**:

1. **Long tail de jugadores:** el 91% aparece en una sola competición → no podemos asignar un token dedicado a cada jugador.
2. **Granularidad excesiva de eventos:** ~3,500 eventos por partido → necesitamos filtrar agresivamente.

Las decisiones tomadas para mitigar ambos problemas están documentadas abajo.

---

## 1. Métricas clave

| Métrica | Valor | Observación |
|---|---|---|
| Total partidos | 3,464 | Suficiente |
| Rango temporal | 1958–2025 | Filtrar a ≥ 2010 para entrenamiento |
| Competiciones únicas | 21 | WC 2018/22, Euro 2020/24, CA 2024 ✅ |
| Equipos únicos | 308 | Manejable como tokens dedicados |
| Jugadores únicos (200-match sample) | 3,986 | Long tail severo |
| Distribución HOME / DRAW / AWAY | 45.2% / 23.0% / 31.8% | Balanceada |
| Eventos por partido (muestra) | 3,549 | Filtrar |
| Tipos de eventos distintos | 30 | Conservar ~10–15 |

## 2. Cobertura de competiciones relevantes

✅ **Internacionales modernas (críticas):**
- FIFA World Cup 2022, 2018
- UEFA Euro 2024, 2020
- Copa America 2024
- African Cup of Nations 2023

✅ **Ligas / clubes con cobertura buena:**
- Champions League 2008–2019 (varias temporadas)
- La Liga 2004–2021

⚠️ **Cobertura limitada (potencial dolor):**
- Premier League: solo 2003/2004 y 2015/2016
- Bundesliga: solo 2023/24 y 2015/16
- Serie A: solo 2015/16 y 1986/87

⚠️ **Históricas (probablemente sin eventos completos):**
- World Cups antiguos (1958, 1962, 1970, 1974, 1986, 1990) — usar con cuidado, los eventos pueden estar parcialmente completados.

## 3. Distribución de overlap jugador-competiciones

Resultado del sample de 200 partidos:

```
1 competición:  3,626 jugadores (91.0%)
2 competiciones:   310 (7.8%)
3 competiciones:    47 (1.2%)
4 competiciones:     3 (0.1%)
```

**Implicación:** la mayoría de embeddings de jugador van a estar mal aprendidos si los tratamos como tokens independientes.

---

## 4. Decisiones de diseño derivadas

### A. Vocabulario jerárquico de jugadores

En vez de un token por jugador, **dos niveles**:

- **Jugadores frecuentes** (≥ 10 partidos en el dataset): token dedicado `PLAYER_<id>`.
- **Long tail / nuevos**: token compuesto `POS_<position>_TIER_<frequency_tier>` (ej. `POS_FW_TIER_2`).

Esto permite manejar jugadores que **nunca vimos** durante entrenamiento (caso típico de un debutante en WC 2026): caen automáticamente en su token de fallback positional.

Threshold inicial: 10 partidos. A ajustar en Fase 2 según la distribución empírica final.

### B. Filtrado de eventos

**Conservar:**
- `Shot` (incluye Goal vía `shot.outcome.name == "Goal"`)
- `Substitution`
- `Foul Won`, `Foul Committed` (con derivación de tarjetas vía `card.name`)
- `Penalty Won`, `Own Goal Against`

**Descartar (al menos en v1):**
- `Pass`, `Ball Receipt`, `Carry`, `Pressure`, `Ball Recovery`
- `Camera On/Off`, `Goal Keeper` (movimientos rutinarios)
- `Duel`, `Block`, `Dribble`, `Clearance`, `Miscontrol`, `Interception`

Esto reduce la secuencia de ~3,500 a ~50–80 tokens por partido.

### C. Filtrado temporal del dataset

Usar solo partidos con `match_date >= 2010-01-01`. Razones:
- Los partidos pre-2010 tienen cobertura de eventos posiblemente incompleta.
- El fútbol moderno (post-2010) tiene patrones tácticos distintos al del s.XX → mezclar épocas introduce ruido.

Estimado: ~3,200 partidos sobreviven el filtro.

### D. Composición del training set para Fase 1 (baselines)

Para los baselines tabulares vamos a usar **TODOS los partidos post-2010**, no solo los internacionales. La intuición: features como ELO y forma reciente se aprenden mejor con más datos, y los baselines tabulares no sufren del problema de embeddings de jugador.

Para el Transformer (Fase 3+), podemos considerar **dos versiones del dataset**:
- **Full**: todos los partidos post-2010 (~3,200)
- **Internationals-only**: solo partidos de selección (~250-400)

Y comparar performance — hipótesis a testear es que el dataset full enseña mejor las representaciones aunque el target final son partidos internacionales.

---

## 5. Riesgos identificados para fases futuras

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| Embeddings de jugadores raros mal aprendidos | Alta | Fallback positional (Decisión A) |
| Mundial 2026 → jugadores desconocidos | Alta | Fallback + rolling features individuales |
| Modelo overfittea a Champions League | Media | Stratified eval por competición |
| Eventos pre-2010 mal etiquetados | Media | Filtro temporal (Decisión C) |
| Sequence length >512 si pasamos contexto | Media | Limitar a 3 partidos recientes en contexto |

---

## 6. Decisiones que quedan ABIERTAS para fases siguientes

1. ¿Bucketización numérica de minutos: cada 5min, 10min, o 15min?
2. ¿Cuántos partidos de contexto reciente pasar como "Mecanismo B"? (default actual: 3)
3. ¿Threshold exacto para jugadores frecuentes? (default actual: 10)
4. ¿Modelo Poisson de baseline o solo XGBoost/LightGBM?

Resolveremos estas durante Fase 1 y Fase 2.

---

## 7. Stack de datos definitivo (post-Fase 0.5)

Tras discusión con la dirección del proyecto, se relajó el requisito de granularidad: **no necesitamos eventos pase-por-pase**. Esto abre la puerta a fuentes complementarias y resuelve parcialmente el sesgo de cobertura geográfica.

**Stack final:**

| Fuente | Granularidad | Cobertura | Uso |
|---|---|---|---|
| **StatsBomb Open Data** | Eventos detallados + lineups | Sesgada a Europa + WC + Euro + Copa Am 2024 + Africa Cup 2023 | Transformer (eventos filtrados: goles, tarjetas, subs) |
| **`martj42/international_results`** | Resultado + score + torneo | **TODAS las selecciones, desde 1872** | Baselines + features rolling + ELO |
| **ELO computado in-house** | Rating temporal por equipo | Todas las selecciones | Feature crítica para baselines |

**Decisión metodológica clave:** ELO se computa **desde cero en código** (notebook `00b_data_augmentation.ipynb`), no se scrapea de eloratings.net. Razones:
- Reproducibilidad completa
- Control de hiperparámetros (K-factor, ventaja local, multiplicador por GD)
- Permite extraer ELO de cualquier equipo en cualquier fecha del pasado, evitando leakage temporal

**Datos NO incluidos (decisión consciente):**
- Brasileirão, Liga MX, MLS reciente, Saudi Pro League, J-League, etc. → requeriría APIs pagas (API-Football, Sportradar). Aceptable para v1 — la mitigación de ELO + resultados internacionales cubre el gap más crítico.

**Decisión metodológica para el Transformer (Fase 3+):**

Estrategia de **transfer learning estilo BERT** (Devlin et al. 2018):

1. **Pretraining (Fase 3-4):** objetivo Masked Language Modeling sobre TODO el dataset de StatsBomb — partidos de selecciones + partidos de clubes. Justificación: aprender representaciones ricas de jugadores, equipos y eventos requiere muchas co-ocurrencias; los partidos de clubes (Champions, La Liga, etc.) son data esencial para que el embedding de Messi aprenda quién es Messi.

2. **Fine-tuning (Fase 4 final):** objetivo de predicción de resultado SOLO sobre partidos de selecciones. Esto garantiza que la distribución de inferencia (Mundial 2026) coincide con la de fine-tuning, mitigando el distribution shift.

Esta separación replica exactamente el paradigma BERT (preentrenamiento no-supervisado en corpus heterogéneo + fine-tuning supervisado en task específica) y es el approach metodológicamente más defendible.

**Resultados de Fase 0.5 (run del 19/05/2026):**
- 49,257 partidos internacionales procesados con ELO
- 336 selecciones únicas con rating computado
- Top ELO: España (2209), Argentina (2177), Francia (2128), Inglaterra (2070), Brasil (2050)
- Cobertura WC 2026: 50/55 selecciones candidatas con datos StatsBomb; 5 sin (Austria, Honduras, Saudi Arabia, Ghana, Qatar) → dependerán solo de features tabulares.

## 8. Próximos pasos inmediatos

**Fase 1: Baselines tabulares.** Construir feature engineering (ELO derivado de Fase 0.5, forma reciente, h2h, días de descanso, localía, etc.) y entrenar Logistic Regression + XGBoost + LightGBM (+ opcionalmente Poisson). Métrica principal: log-loss en validación temporal (post-2023). Esto garantiza un sistema operativo para predecir el Mundial 2026 independientemente de cómo evolucione el Transformer.

---

*Documento generado al cierre de Fase 0.*
*Actualizado 16/05/2026 con stack de datos definitivo (Fase 0.5).*
