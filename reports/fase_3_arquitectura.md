# Fase 3 — Arquitectura del D10Sformer

## Resumen ejecutivo

Implementamos la arquitectura completa del Transformer encoder bidireccional
(estilo BERT) que será pre-entrenado en Fase 4. La elección es **encoder-only**
(no decoder autoregresivo) porque el objetivo final no es generar texto sino
producir representaciones bidireccionales del partido que alimenten cabezales
de clasificación calibrados.

## Módulos implementados

```
src/models/
├── embeddings.py        — token + position + segment (BERT-style)
├── attention.py         — Multi-Head Self-Attention from scratch
├── transformer.py       — Pre-LN encoder block + stack
├── heads.py             — MLMHead, ResultHead, ScoreHead
└── d10sformer.py        — Wrapper end-to-end + D10SformerConfig
```

## Decisiones de diseño justificadas

### 1. Encoder-only (no decoder)
**Por qué:** la tarea es predicción (clasificación), no generación. Necesitamos
representaciones del [CLS] que vean *todo* el partido bidireccionalmente. Un
decoder autoregresivo (estilo GPT) restringe cada token a mirar solo hacia
atrás — innecesario y subóptimo aquí.

Referencia: Devlin et al. (2018) §3.

### 2. Pre-LN (no Post-LN)
**Por qué:** el Transformer original (Vaswani et al., 2017) aplica
LayerNorm *después* del residual:

```
x' = LN(x + sublayer(x))
```

Esto requiere un esquema de *warmup* delicado del learning rate para no
divergir. Xiong et al. (2020) probaron formalmente que la variante Pre-LN:

```
x' = x + sublayer(LN(x))
```

permite gradientes mucho más estables porque el camino residual no atraviesa
una LayerNorm. Esto **elimina la necesidad de warmup ajustado a mano** y
acelera la convergencia. Es el estándar en GPT-2, BLOOM, LLaMA, Falcon, etc.

### 3. Atención implementada desde cero (no `nn.MultiheadAttention`)
**Por qué académico:** el programa de MIA305 exige entender la "fontanería"
matemática de Q/K/V y el escalado por 1/√d_k. Implementar desde cero permite:
- mostrar explícitamente las 4 proyecciones (W_q, W_k, W_v, W_o);
- justificar el factor de escala 1/√d_k (sin él, los productos escalares
  crecen con d_k y la softmax cae en zona de gradiente desvaneciente);
- exponer las matrices de atención para visualizarlas en Fase 5
  (interpretabilidad).

### 4. Embeddings compuestos
Cada token recibe la **suma** de tres embeddings aprendidos:

```
E_final(t_i) = E_tok(t_i) + E_pos(i) + E_seg(s_i)
```

El segment embedding es clave en nuestro dominio: distingue tokens de la
sección `[LINEUP_A]` vs. `[LINEUP_B]` vs. `[EVENTS]`. Sin él, el modelo
no podría saber a qué equipo "pertenece" un jugador (los tokens
PLAYER_<id> son los mismos sin importar dónde aparecen).

Posición: **learned**, no sinusoidal. Razón: nuestra `max_seq_length=512`
y no necesitamos extrapolar a longitudes mayores; los learned positions
empíricamente igualan o superan a las sinusoidales en este régimen.

### 5. Weight tying entre embedding y MLM head
Compartir `decoder.weight = token_embedding.weight` ahorra
`vocab_size × d_model ≈ 4521 × 256 ≈ 1.16M` parámetros — ~17% del total.
Es la receta de Press & Wolf (2017) adoptada por BERT y GPT-2.

### 6. Tres cabezales multitarea
1. **MLMHead** — pre-training masked-language modeling.
2. **ResultHead** — 3 clases (home/draw/away) sobre [CLS] (calibrado).
3. **ScoreHead** — 36 clases sobre vocabulario `SCORE_*_*`.

**¿Por qué clasificar el score en lugar de regresarlo?** Goles son
discretos y pequeños (0..5+); la clasificación captura naturalmente la
distribución conjunta y produce probabilidades calibrables. La regresión
con MSE haría que 1-0 y 0-1 estén equidistantes de 0-0, ignorando que
implican resultados opuestos.

## Presupuesto de parámetros

Con la configuración base (`d_model=256, num_layers=6, num_heads=8,
d_ff=1024, vocab≈4521, max_len=512`):

| Componente | Parámetros aproximados |
|---|---|
| Embeddings (token+pos+seg+LN) | ~1.36M |
| 6 × Encoder block | ~4.74M |
| MLM head (decoder tied con embedding) | ~70K |
| Result head + Score head | ~140K |
| **TOTAL** | **~6.3M** |

Esto cabe holgado en T4 (16GB) incluso con batch 64 y secuencias de 512
tokens. La memoria pico estimada es de orden 1-2 GB durante el forward+
backward (activaciones), muy por debajo del límite.

## Tests

Suite de 5 archivos, ~35 tests:

- `tests/test_embeddings.py` (7 tests)
- `tests/test_attention.py` (7 tests)
- `tests/test_transformer.py` (6 tests)
- `tests/test_heads.py` (7 tests)
- `tests/test_d10sformer.py` (8 tests)

Cubren: shapes, residual path correctness, padding-mask propagation,
weight tying, backward pass, parámetros en el rango esperado.

**Validación local:** `py_compile` OK en los 5 módulos + 5 tests.
**Validación funcional:** se corren en el notebook `03_architecture.ipynb`
sección 8 (subprocess pytest), aprovechando que Colab tiene PyTorch
preinstalado.

## Riesgos identificados

1. **Inestabilidad en gradiente al inicio del pre-training (Fase 4).**
   Mitigación: usamos Pre-LN, init `N(0, 0.02)`, AdamW con weight decay
   solo en pesos (no biases ni LN), warmup lineal del 10% de los pasos.

2. **NaN en filas totalmente padded.** El módulo `attention.py` aplica
   `nan_to_num` después de la softmax para casos donde una query no
   tenga ningún key visible. Esto es preventivo: no debería ocurrir si
   los datos están bien formados.

3. **Tied weights y MLM scaling.** Cuando tied=True, el gradiente del
   decoder fluye también al embedding. Esto puede acelerar el aprendizaje
   pero también amplificar updates pequeños. Mantener `initializer_range
   = 0.02` mitiga.

## Cómo proceder

1. Vic corre `notebooks/03_architecture.ipynb` en Colab y reporta:
   - Total de parámetros + desglose
   - Tamaño del checkpoint inicial
   - Resultado de pytest (todos verdes)
   - Visualización de atenciones (deben verse difusas/aleatorias)

2. Confirmado eso, pasamos a **Fase 4**: construcción del dataset de
   pre-training (objetivos: MLM + masking estocástico de features), loop
   de entrenamiento con AMP, logging en wandb, fine-tuning en
   selecciones, métricas vs. baselines (log-loss, Brier, ECE).
