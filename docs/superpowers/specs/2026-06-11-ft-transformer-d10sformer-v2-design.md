# Technical Specification: FT-Transformer (D10Sformer v2)

**Author:** Expert LLM & Statistics Developer  
**Date:** June 11, 2026  
**Status:** Approved / Design Phase  
**Target Architecture:** Feature Tokenizer Transformer (FT-Transformer)  
**Task:** Tabular Football Prediction (Joint Score & Outcome)

---

## 1. Background & Retrospective of v1

The original `D10Sformer v1` was a sequence-to-sequence style model (style BERT) trained from scratch over discrete tokens. While conceptually interesting, it fell short of the simple tabular baseline (Logistic Regression) in terms of cross-entropy calibration (`log_loss = 0.88` vs `0.86`, and `ECE = 0.040` vs `0.024`). 

### Why did v1 underperform?
1. **The Discretization Bottleneck:** Numerical values (such as ELO ratings, forms, and goals) were coarsened into discrete bins (e.g., `ELO_BUCKET_2000`). This threw away high-precision relative differences, which hold the strongest predictive signal in football analytics.
2. **Positional Shift and Token Out-of-Bounds:** During our retrospective, we discovered that `stage="group"` introduced a `STAGE_GROUP` token that appeared exactly **zero** times in the training and validation corpus (where all matches had `stage=None`). This shifted sequence positions, broke positional embeddings, and completely inverted predictions.
3. **Competing heads:** The model used separate, independent classification heads for outcome (3 classes) and score (36 classes). This allowed the model to output contradictory predictions and diluted the gradients during multi-task learning.

---

## 2. Core Architecture: FT-Transformer (D10Sformer v2)

The `D10Sformer v2` migrates to a **Feature Tokenizer Transformer** architecture specifically engineered for high-precision hybrid tabular data. It removes positional embeddings (enforcing permutation-invariance of features) and implements continuous-feature linear projections.

```
                      ┌────────────────────────────────────────┐
                      │          [CLS] Output (128,)           │
                      └──────────────────┬─────────────────────┘
                                         ▼
                      ┌────────────────────────────────────────┐
                      │            Score Head MLP              │
                      │   Linear(128, 128) -> Tanh ->          │
                      │   Linear(128, 36) -> Softmax           │
                      └──────────────────┬─────────────────────┘
                                         ▼
                     Probabilidades de Marcadores Exactos (36,)
                        [P(0-0), P(0-1), ..., P(5-5)]
                                         │
               ┌─────────────────────────┴─────────────────────────┐
               ▼                                                   ▼
      [Marcador más probable]                             [Result Probability (3,)]
          argmax(36,)                                     Derivado sumando clases:
       Ej: index 7 -> 1-1                               P(Home) = Sum(SCORE_h_a) donde h > a
                                                        P(Draw) = Sum(SCORE_h_a) donde h == a
                                                        P(Away) = Sum(SCORE_h_a) donde h < a
```

### A. The Feature Tokenizer (Input Layer)
We project every individual feature (categorical and continuous) to a shared embedding dimension `d_model = 128`.

1. **Categorical Features (Lookup Embeddings):**
   * `tournament_class` (e.g., world_cup, friendly): `nn.Embedding(num_classes, 128)`
   * `neutral` (0 or 1): `nn.Embedding(2, 128)`
   * `venue` (home, away, neutral): `nn.Embedding(3, 128)`

2. **Continuous Features (Linear Projections):**
   For each of our 9 continuous variables, we apply a dedicated, learnable linear projection:
   $$\mathbf{e}_j = x_j \cdot \mathbf{w}_j + \mathbf{b}_j$$
   Where $\mathbf{w}_j \in \mathbb{R}^{128}$ and $\mathbf{b}_j \in \mathbb{R}^{128}$ are trainable parameters. This preserves continuous precision without bucketization.
   
   The 9 continuous features are:
   * `home_elo` (Team A ELO)
   * `away_elo` (Team B ELO)
   * `elo_diff` (home_elo - away_elo)
   * `home_form5_pts` (Form pts Team A)
   * `away_form5_pts` (Form pts Team B)
   * `home_form5_gf` (Recent goals Team A)
   * `away_form5_gf` (Recent goals Team B)
   * `home_form10_pts` (Extended form Team A)
   * `away_form10_pts` (Extended form Team B)

3. **[CLS] Token:**
   We prepand a trainable, randomly initialized $\mathbf{e}_{[\text{CLS}]} \in \mathbb{R}^{128}$ vector to act as the global interaction pool.

The final sequence length is exactly **13 tokens**:
$$\mathbf{X}_{\text{in}} = \big[ \mathbf{e}_{[\text{CLS}]}, \mathbf{e}_{\text{tourn\_class}}, \mathbf{e}_{\text{neutral}}, \mathbf{e}_{\text{venue}}, \mathbf{e}_{\text{home\_elo}}, \mathbf{e}_{\text{away\_elo}}, \mathbf{e}_{\text{elo\_diff}}, \dots \big]$$

---

### B. The Transformer Encoder Stack
* `d_model`: 128
* `num_layers`: 3 (regularized for low-data regime)
* `num_heads`: 4
* `d_ff`: 256
* `dropout` / `attention_dropout`: 0.15
* `activation`: GELU
* **No Positional Embeddings:** Since the order of features is arbitrary, we omit positional embeddings. This ensures permutation invariance, prevents spatial overfitting, and forces self-attention to focus purely on semantic cross-feature correlations.

---

### C. Joint Output Head (Consistency Guarantee)
To guarantee 100% logical consistency between match outcomes and exact scoreline predictions, `D10Sformer v2` implements a **single unified Score Head** of 36 classes (representing joint outcomes from `0-0` to `5-5`).

1. **Score Head Architecture:**
   * Reads from the `[CLS]` token output at layer 3: $\mathbf{h}_{[\text{CLS}]} \in \mathbb{R}^{128}$.
   * Passes through a dedicated projection: `nn.Linear(128, 128)` -> `nn.Tanh()` -> `nn.Dropout(0.15)` -> `nn.Linear(128, 36)`.
   * Class output order is mapped to `SCORE_h_a` where index $i$ translates to home goals $h = i \div 6$ and away goals $a = i \bmod 6$.

2. **Deterministic Outcome Reduction:**
   Instead of predicting home, draw, and away probabilities separately, we sum the corresponding score slice probabilities:
   * $$P(\text{Home Win}) = \sum P(\text{SCORE}_{h, a}) \quad \forall h > a$$
   * $$P(\text{Draw}) = \sum P(\text{SCORE}_{h, a}) \quad \forall h = a$$
   * $$P(\text{Away Win}) = \sum P(\text{SCORE}_{h, a}) \quad \forall h < a$$
   
   This guarantees absolute consistency, simplifies optimization, and leverages the joint probability of goals to output beautifully calibrated outcome probabilities.

---

## 3. Data Processing & Training Pipeline

1. **Dataset:** We will build a unified feature matrix directly from `processed/corpus/finetune_train.pkl`, `val.pkl`, and `test.pkl`.
2. **Loss Function:** Multi-class Cross-Entropy on the 36 score classes, weighted by inverse class frequencies in the training set to prevent collapse onto common scores like `1-0` or `1-1`.
3. **Training Setup:**
   * Epochs: 15 (with cosine learning rate decay and warmup).
   * Learning Rate: 1e-4.
   * Batch size: 64.
   * Fully compatible and lightweight for instantaneous training on Apple Silicon M4 (<15 seconds).

---

## 4. Evaluation & Comparison Metrics
To prove our hypothesis that a Transformer can outperform probabilistic baselines, we will evaluate the model on the test dataset (`test.pkl`) and compare it head-to-head with:
*   **Logistic Regression** (`log_loss`, `Brier`, `ECE`, `Accuracy`)
*   **XGBoost** (`log_loss`, `Brier`, `ECE`, `Accuracy`)
*   **D10Sformer v1** (to quantify our architectural delta!)

---

## 5. Self-Review Check
* **Placeholders:** None.
* **Internal Consistency:** No conflicting dimensions; sequence length is exactly 13; classification mappings are fully defined.
* **Scope:** Compact, high-performance, and lightweight. Perfect for isolated, 100% autonomous implementation.
