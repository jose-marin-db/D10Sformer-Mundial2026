"""
End-to-end D10Sformer model.

Wires together:
    MatchEmbedding  →  TransformerEncoder  →  [MLMHead, ResultHead, ScoreHead]

The forward pass returns a dict of all logits; loss computation lives in the
training loop (so loss weighting can be configured per-experiment).

Parameter budget at the default config (d=256, layers=6, heads=8, ff=1024,
vocab≈4,500, max_len=512):

    Embeddings (token+pos+seg):       ~4,500*256 + 512*256 + 8*256 ≈ 1.30M
    Per encoder block (attn+ffn+ln):  4*d^2 + 2*d*d_ff + ~ ≈ 0.79M
    6 blocks                          ≈ 4.74M
    MLM head (tied with token emb):   d^2 + 2*d ≈ 0.07M (+ tied decoder weight)
    Result + Score heads:             ~0.14M
    -----------------------------------------------------------------
    TOTAL trainable                   ≈ 6-7M parameters (very lean — perfect
                                        for T4 training in <2h).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .embeddings import MatchEmbedding, DEFAULT_NUM_SEGMENTS
from .transformer import TransformerEncoder
from .heads import MLMHead, ResultHead, ScoreHead


@dataclass
class D10SformerConfig:
    """Hyper-parameters for the D10Sformer. Mirrors configs/base_config.yaml."""

    vocab_size: int
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 1024
    max_seq_length: int = 512
    num_segments: int = DEFAULT_NUM_SEGMENTS
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    pad_token_id: int = 0

    # Score head — defaults to 36 (6*6 combinations clamped 0..5 each side)
    num_score_classes: int = 36

    # Tie MLM decoder weight with token embedding (BERT recipe, saves ~vocab*d params)
    tie_mlm_weights: bool = True


class D10Sformer(nn.Module):
    """The full D10Sformer: a BERT-like encoder with three task heads."""

    def __init__(self, config: D10SformerConfig):
        super().__init__()
        self.config = config

        self.embeddings = MatchEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_length=config.max_seq_length,
            num_segments=config.num_segments,
            pad_token_id=config.pad_token_id,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
            initializer_range=config.initializer_range,
        )

        self.encoder = TransformerEncoder(
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            layer_norm_eps=config.layer_norm_eps,
        )

        tied_emb = self.embeddings.token_embedding if config.tie_mlm_weights else None
        self.mlm_head = MLMHead(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            layer_norm_eps=config.layer_norm_eps,
            tied_embedding=tied_emb,
        )
        self.result_head = ResultHead(d_model=config.d_model, dropout=config.dropout)
        self.score_head = ScoreHead(
            d_model=config.d_model,
            num_score_classes=config.num_score_classes,
            dropout=config.dropout,
        )

        self._init_linear_weights()

    def _init_linear_weights(self) -> None:
        """Truncated-normal init for all Linears, zero for biases."""
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ---------- main forward ----------

    def forward(
        self,
        token_ids: torch.LongTensor,                  # (B, T)
        segment_ids: torch.LongTensor,                # (B, T)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T) — 1=keep, 0=pad
        return_hidden: bool = False,
    ) -> dict:
        """Run a full forward pass and return all logits.

        Returns dict with keys:
            "result_logits": (B, 3)
            "score_logits":  (B, num_score_classes)
            "mlm_logits":    (B, T, vocab_size)
            "hidden":        (B, T, d_model)  — only if return_hidden=True
        """
        if attention_mask is None:
            # If not given, assume nothing is padded
            attention_mask = torch.ones_like(token_ids)

        emb = self.embeddings(token_ids, segment_ids)
        hidden = self.encoder(emb, attention_mask=attention_mask)

        out = {
            "result_logits": self.result_head(hidden),
            "score_logits": self.score_head(hidden),
            "mlm_logits": self.mlm_head(hidden),
        }
        if return_hidden:
            out["hidden"] = hidden
        return out

    # ---------- introspection ----------

    def num_parameters(self, only_trainable: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if not only_trainable or p.requires_grad
        )

    def parameter_breakdown(self) -> dict[str, int]:
        """Per-module parameter counts (handy for the paper)."""
        def n(mod): return sum(p.numel() for p in mod.parameters())
        return {
            "embeddings": n(self.embeddings),
            "encoder": n(self.encoder),
            "mlm_head": n(self.mlm_head),
            "result_head": n(self.result_head),
            "score_head": n(self.score_head),
            "TOTAL": self.num_parameters(),
        }
