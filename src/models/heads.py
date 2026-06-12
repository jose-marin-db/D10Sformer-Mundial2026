"""
Multi-task heads for D10Sformer.

Three heads attached on top of the Transformer encoder:

1. **MLM head** — predicts the original token at [MASK] positions.
   Used during pre-training only. Tied (optionally) with the input token
   embedding to save parameters (BERT recipe).

2. **Result head** — 3-class classification (home_win / draw / away_win),
   reading the [CLS] hidden state.

3. **Score head** — 36-class classification over the SCORE_*_* vocabulary
   (clamped 0..5 per side, see vocabulary.py), again from [CLS].

Why classify the score (36 classes) instead of regressing two integers?
- Goals are discrete and small (0..5+), so a classifier captures the joint
  distribution naturally and produces calibratable probabilities.
- Regression with MSE makes the loss insensitive to the unimodal nature of
  scoring (e.g., 1-0 vs 0-1 are equidistant from 0-0 under MSE but they
  imply opposite results).

All heads return logits — the loss is computed externally (CrossEntropyLoss).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLMHead(nn.Module):
    """Masked Language Modeling head — BERT-style.

    Transformation pipeline (same as `BertOnlyMLMHead` in HuggingFace):
        dense → GELU → LayerNorm → decoder (Linear to vocab)

    The decoder weight is optionally tied to the input token embedding.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        layer_norm_eps: float = 1e-12,
        tied_embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = nn.Linear(d_model, vocab_size, bias=True)

        if tied_embedding is not None:
            # Weight tying: decoder.weight shares storage with embedding.weight
            self.decoder.weight = tied_embedding.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: (B, T, d_model) → logits (B, T, vocab_size)."""
        x = self.dense(hidden_states)
        x = self.act(x)
        x = self.layer_norm(x)
        return self.decoder(x)


class ClassificationHead(nn.Module):
    """Generic pooled-CLS classification head.

    Pipeline: take h[:, 0] (the [CLS] vector) → dense → tanh → dropout → out.

    This mirrors BERT's pooled output. We deliberately use `tanh` (BERT) rather
    than GELU here because the pooled vector is meant to be a fixed-size
    representation of the whole sequence, and tanh keeps it bounded.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: (B, T, d_model) → logits (B, num_classes).

        Reads the first position (assumed to be [CLS]).
        """
        cls = hidden_states[:, 0, :]      # (B, d_model)
        x = self.dense(cls)
        x = self.act(x)
        x = self.dropout(x)
        return self.classifier(x)


class ResultHead(ClassificationHead):
    """3-class head: home_win / draw / away_win."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__(d_model=d_model, num_classes=3, dropout=dropout)


class ScoreHead(ClassificationHead):
    """36-class head over the SCORE_*_* vocabulary (0..5 per side)."""

    def __init__(self, d_model: int, num_score_classes: int = 36, dropout: float = 0.1):
        super().__init__(d_model=d_model, num_classes=num_score_classes, dropout=dropout)
