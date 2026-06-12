"""
Multi-Head Self-Attention — implemented from scratch (not nn.MultiheadAttention)
for didactic transparency.

Reference: Vaswani et al., 2017 "Attention Is All You Need", §3.2.
Notation follows Jurafsky & Martin (3rd ed.), Chapter 11.

Scaled dot-product attention:

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

Where Q, K, V are linear projections of the input X:

    Q = X W_Q,  K = X W_K,  V = X W_V

The 1/sqrt(d_k) scaling is *not* cosmetic: without it, for large d_k the dot
products grow in magnitude and push softmax into regions of vanishing
gradient. This is why the model would not train without it.

Multi-Head: rather than a single attention with d_model features, we run h
parallel attentions of d_model/h features each, then concatenate. This lets
different heads attend to different "subspaces" of the representation.

Complexity: O(T^2 * d_model) per layer — *quadratic in sequence length*.
For T=512, d=256, this is ~67M ops per head per layer; with 8 heads and 6
layers that's ~3.2B ops per forward pass per example. T4 handles this fine.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention with explicit Q, K, V projections.

    Args:
        d_model: hidden dimension (must be divisible by num_heads).
        num_heads: number of attention heads.
        dropout: dropout applied to the attention probabilities.
        bias: whether the Q/K/V/output projections have bias terms.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads     # dimension per head
        self.scale = 1.0 / math.sqrt(self.d_k)

        # Separate Q, K, V projections (instead of one fused 3*d_model linear)
        # for didactic clarity in the paper.
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) → (B, h, T, d_k)."""
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, h, T, d_k) → (B, T, d_model)."""
        bsz, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(bsz, seq_len, self.d_model)

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, d_model)
        attention_mask: torch.Tensor | None = None,  # (B, T) bool/int — 1=keep, 0=pad
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Returns context tensor of same shape as input.

        attention_mask convention: 1 for real tokens, 0 for [PAD]. Internally
        we convert to additive mask (-inf for padded positions).
        """
        bsz, seq_len, _ = x.shape

        # 1. Project to Q, K, V
        q = self.W_q(x)   # (B, T, d_model)
        k = self.W_k(x)
        v = self.W_v(x)

        # 2. Split into heads
        q = self._split_heads(q)  # (B, h, T, d_k)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # 3. Scaled dot-product: scores = Q K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # scores: (B, h, T, T)

        # 4. Apply padding mask if given
        if attention_mask is not None:
            # mask shape: (B, T) → (B, 1, 1, T) so it broadcasts to (B, h, T, T)
            mask = attention_mask.to(dtype=torch.bool)
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(~mask, float("-inf"))

        # 5. Softmax → attention probabilities
        attn = F.softmax(scores, dim=-1)
        # When an entire row is masked (all -inf), softmax yields NaN. Replace
        # with zeros so the downstream mat-mul produces zero (those positions
        # are padded anyway and will be ignored by the loss).
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)

        # 6. Weighted sum over V
        context = torch.matmul(attn, v)   # (B, h, T, d_k)

        # 7. Merge heads + output projection
        context = self._merge_heads(context)  # (B, T, d_model)
        output = self.W_o(context)

        if return_weights:
            return output, attn
        return output
