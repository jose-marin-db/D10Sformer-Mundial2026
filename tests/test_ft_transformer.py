"""Tests for src/models/ft_transformer.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.ft_transformer import FeatureTokenizer, FTD10Sformer, FTD10SformerConfig


def test_feature_tokenizer_output_shape() -> None:
    B = 4
    tokenizer = FeatureTokenizer(d_model=64, num_tournament_classes=10)
    cat_feats = torch.randint(0, 2, (B, 3))
    cont_feats = torch.randn(B, 9)
    out = tokenizer(cat_feats, cont_feats)
    assert out.shape == (B, 12, 64)


def test_ft_d10sformer_forward_and_loss_flow() -> None:
    config = FTD10SformerConfig(
        vocab_size=10,
        d_model=32,
        num_layers=2,
        num_heads=2,
        d_ff=64,
        num_tournament_classes=5,
        dropout=0.0,
    )
    model = FTD10Sformer(config)
    B = 2
    cat_feats = torch.randint(0, 2, (B, 3))
    cont_feats = torch.randn(B, 9)
    logits = model(cat_feats, cont_feats)
    assert logits.shape == (B, 36)
