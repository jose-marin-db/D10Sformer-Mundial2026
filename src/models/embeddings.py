"""
Composed embeddings for D10Sformer.

Following BERT (Devlin et al., 2018, §3.1), the input representation of each
token is the sum of three learned embeddings:

    E_final(t_i) = E_token(t_i) + E_position(i) + E_segment(s_i)

We use *learned* positional embeddings (BERT-style), not sinusoidal (Vaswani
et al., 2017), because (a) our sequence length is bounded (max 512), and (b)
learned positions consistently match or beat sinusoidal in low-to-medium
sequence regimes.

The segment embedding tells the model which structural section of the match
each token belongs to (meta / features / lineup_a / ... / sep). The segment
ids are produced by the MatchTokenizer (see src/data/tokenizer.py for the
SEG_* constants).

After summation we apply LayerNorm and Dropout, again following BERT.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Must match the number of distinct SEG_* values in src/data/tokenizer.py
# (SEG_META=0 ... SEG_SEP=7). We add a small buffer for future extensions.
DEFAULT_NUM_SEGMENTS = 8


class MatchEmbedding(nn.Module):
    """Token + Position + Segment embedding, with LayerNorm + Dropout.

    Args:
        vocab_size: size of the FootballVocab (e.g., ~4,500).
        d_model: hidden dimension of the Transformer (e.g., 256).
        max_seq_length: maximum sequence length the model will ever see.
        num_segments: number of distinct segment ids (default 8).
        pad_token_id: id of [PAD] in the vocab — embedding is zero-initialised
            and skipped in gradients via padding_idx.
        dropout: dropout applied to the summed embedding.
        layer_norm_eps: epsilon for the LayerNorm.
        initializer_range: std of the truncated-normal weight init (BERT default 0.02).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        max_seq_length: int = 512,
        num_segments: int = DEFAULT_NUM_SEGMENTS,
        pad_token_id: int = 0,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.num_segments = num_segments
        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        self.segment_embedding = nn.Embedding(num_segments, d_model)

        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        # Pre-compute the position ids buffer (non-trainable)
        self.register_buffer(
            "position_ids",
            torch.arange(max_seq_length).unsqueeze(0),  # (1, max_seq_length)
            persistent=False,
        )

        self._init_weights(initializer_range)

    def _init_weights(self, initializer_range: float) -> None:
        """Truncated-normal init for embeddings (BERT recipe)."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.segment_embedding.weight, mean=0.0, std=initializer_range)
        # Zero out the [PAD] token embedding
        with torch.no_grad():
            self.token_embedding.weight[self.pad_token_id].zero_()

    def forward(
        self,
        token_ids: torch.LongTensor,    # (B, T)
        segment_ids: torch.LongTensor,  # (B, T)
    ) -> torch.Tensor:
        """Returns embedded tensor of shape (B, T, d_model)."""
        bsz, seq_len = token_ids.shape
        if seq_len > self.max_seq_length:
            raise ValueError(
                f"Sequence length {seq_len} > max_seq_length {self.max_seq_length}"
            )

        # Slice the pre-computed position ids
        pos_ids = self.position_ids[:, :seq_len]  # (1, T)

        tok_emb = self.token_embedding(token_ids)      # (B, T, d)
        pos_emb = self.position_embedding(pos_ids)     # (1, T, d) — broadcasts
        seg_emb = self.segment_embedding(segment_ids)  # (B, T, d)

        emb = tok_emb + pos_emb + seg_emb
        emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb
