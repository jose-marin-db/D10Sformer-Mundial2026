"""
Transformer Encoder block + full encoder stack.

Design choices:

1. **Pre-LN (Pre-Norm) variant**, not the original Post-LN (Vaswani et al.,
   2017). Pre-LN is more stable: gradients flow directly through the residual
   path, and warmup is less critical. This matches modern practice
   (Xiong et al., 2020 "On Layer Normalization in the Transformer Architecture").

       x' = x + Dropout( MHSA( LN(x) ) )
       y  = x' + Dropout( FFN( LN(x') ) )

2. **GELU activation** in the FFN (BERT default, Hendrycks & Gimpel 2016),
   not ReLU. GELU is smoother and consistently better in language models.

3. **No causal mask** — this is an *encoder*, bidirectional attention is
   essential for the masked-language modeling objective and for capturing
   relations between players in a lineup (every player should see every
   other player, regardless of position in the sequence).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear → GELU → Linear.

    Standard FFN(x) = max(0, x W_1 + b_1) W_2 + b_2 from Vaswani et al. (2017),
    but with GELU instead of ReLU.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class TransformerEncoderBlock(nn.Module):
    """A single Pre-LN Transformer encoder block.

    Args:
        d_model: hidden dimension.
        num_heads: number of attention heads.
        d_ff: hidden dimension of the FFN (typically 4 * d_model).
        dropout: residual / FFN dropout.
        attention_dropout: dropout inside the attention probabilities.
        layer_norm_eps: epsilon for the LayerNorm modules.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, dropout=attention_dropout
        )
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-LN attention sublayer
        h = self.ln1(x)
        h = self.attn(h, attention_mask=attention_mask)
        x = x + self.dropout1(h)

        # Pre-LN FFN sublayer
        h = self.ln2(x)
        h = self.ffn(h)
        x = x + self.dropout2(h)
        return x


class TransformerEncoder(nn.Module):
    """Stack of N TransformerEncoderBlocks + final LayerNorm.

    The final LayerNorm is standard in Pre-LN architectures (otherwise the
    output of the last residual is not normalised before downstream heads).
    """

    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                attention_dropout=attention_dropout,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return self.final_norm(x)
