"""
Collator for D10Sformer: pads variable-length batches and applies BERT-style
MLM masking on the fly.

MLM recipe (Devlin et al., 2018, §3.1):
    Sample 15% of input tokens uniformly. For each sampled token:
      - 80% replace with [MASK]
      - 10% replace with a random vocabulary token
      - 10% leave unchanged
    The model is trained to predict the *original* token at all sampled
    positions; positions NOT sampled have label = -100 (ignored by
    nn.CrossEntropyLoss).

We forbid masking on **structural / special tokens** (CLS, SEP, PAD, MASK,
UNK, [LINEUP_A], [FEATURES_START], etc.) — masking those is meaningless and
hurts learning. We also forbid the [PREDICT_RESULT] sentinel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .dataset import MatchSample
from .vocabulary import FootballVocab, SPECIAL_TOKENS


@dataclass
class CollatedBatch:
    """A batched, padded, MLM-masked training example ready for the model."""
    token_ids: torch.LongTensor       # (B, T) — with [MASK]/random replacements
    segment_ids: torch.LongTensor     # (B, T)
    attention_mask: torch.LongTensor  # (B, T) — 1 for real tokens, 0 for pad
    mlm_labels: torch.LongTensor      # (B, T) — original token where masked, -100 elsewhere
    result_labels: torch.LongTensor   # (B,) — original result id, -100 if absent
    score_labels: torch.LongTensor    # (B,) — original score id, -100 if absent

    def to(self, device) -> "CollatedBatch":
        return CollatedBatch(
            token_ids=self.token_ids.to(device),
            segment_ids=self.segment_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            mlm_labels=self.mlm_labels.to(device),
            result_labels=self.result_labels.to(device),
            score_labels=self.score_labels.to(device),
        )


class MLMCollator:
    """Collator that pads + applies MLM masking.

    Args:
        vocab: FootballVocab (for ids of [MASK], [PAD], and for the special
            tokens set we must NOT mask).
        mlm_probability: total fraction of tokens to sample for masking
            (default 0.15, BERT).
        mask_prob: of the sampled tokens, fraction replaced with [MASK]
            (default 0.80).
        random_prob: of the sampled tokens, fraction replaced with a random
            vocab token (default 0.10). The remaining (1 - mask_prob - random_prob)
            are left unchanged.
        seed: optional seed for reproducibility (None → use global generator).
    """

    def __init__(
        self,
        vocab: FootballVocab,
        mlm_probability: float = 0.15,
        mask_prob: float = 0.80,
        random_prob: float = 0.10,
        seed: int | None = None,
    ):
        if mask_prob + random_prob > 1.0:
            raise ValueError("mask_prob + random_prob must be <= 1.0")
        self.vocab = vocab
        self.mlm_probability = mlm_probability
        self.mask_prob = mask_prob
        self.random_prob = random_prob
        self.vocab_size = len(vocab)

        # Cache ids of unmaskable tokens (all specials + bracket-style structural)
        unmaskable = set()
        for tok in vocab.token_to_id:
            if tok in SPECIAL_TOKENS or tok.startswith("[") or tok == "[PREDICT_RESULT]":
                unmaskable.add(vocab.token_to_id[tok])
        self.unmaskable_ids = unmaskable
        self.pad_id = vocab.token_to_id["[PAD]"]
        self.mask_id = vocab.token_to_id["[MASK]"]
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    # ----- main entry -----

    def __call__(self, batch: list[MatchSample]) -> CollatedBatch:
        bsz = len(batch)
        max_len = max(s.length for s in batch)

        token_ids = torch.full((bsz, max_len), self.pad_id, dtype=torch.long)
        segment_ids = torch.zeros((bsz, max_len), dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)

        for i, s in enumerate(batch):
            token_ids[i, :s.length] = torch.tensor(s.token_ids, dtype=torch.long)
            segment_ids[i, :s.length] = torch.tensor(s.segment_ids, dtype=torch.long)
            attention_mask[i, :s.length] = 1

        # Apply MLM masking — operates on a copy so we keep the originals as labels
        masked_token_ids, mlm_labels = self._apply_mlm_masking(token_ids, attention_mask)

        result_labels = torch.tensor(
            [s.target_result_id if s.target_result_id is not None else -100 for s in batch],
            dtype=torch.long,
        )
        score_labels = torch.tensor(
            [s.target_score_id if s.target_score_id is not None else -100 for s in batch],
            dtype=torch.long,
        )

        return CollatedBatch(
            token_ids=masked_token_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            mlm_labels=mlm_labels,
            result_labels=result_labels,
            score_labels=score_labels,
        )

    # ----- internals -----

    def _apply_mlm_masking(
        self,
        token_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        """Returns (masked_token_ids, mlm_labels).

        mlm_labels: -100 where loss is ignored, original token id otherwise.
        """
        labels = token_ids.clone()
        masked_ids = token_ids.clone()

        # Probability matrix uniformly = mlm_probability everywhere
        prob_matrix = torch.full(token_ids.shape, self.mlm_probability)

        # Don't mask padded positions
        prob_matrix = prob_matrix * attention_mask.float()

        # Don't mask unmaskable (special/structural) tokens
        unmaskable_mask = self._build_unmaskable_mask(token_ids)
        prob_matrix = prob_matrix * (~unmaskable_mask).float()

        # Bernoulli sample
        masked_indices = torch.bernoulli(prob_matrix, generator=self.generator).bool()

        # Labels: -100 for non-masked
        labels[~masked_indices] = -100

        # Of the masked, decide replacement
        rand = torch.rand(token_ids.shape, generator=self.generator)
        # 80% [MASK]
        replace_with_mask = masked_indices & (rand < self.mask_prob)
        masked_ids[replace_with_mask] = self.mask_id

        # 10% random token (next mask_prob..mask_prob+random_prob range)
        replace_with_random = (
            masked_indices
            & (rand >= self.mask_prob)
            & (rand < self.mask_prob + self.random_prob)
        )
        random_tokens = torch.randint(
            low=0, high=self.vocab_size, size=token_ids.shape,
            generator=self.generator, dtype=torch.long,
        )
        masked_ids[replace_with_random] = random_tokens[replace_with_random]

        # Remaining 10% left unchanged (no operation needed)
        return masked_ids, labels

    def _build_unmaskable_mask(self, token_ids: torch.LongTensor) -> torch.BoolTensor:
        """True where the token is in `self.unmaskable_ids`."""
        unmask = torch.zeros_like(token_ids, dtype=torch.bool)
        for tid in self.unmaskable_ids:
            unmask |= (token_ids == tid)
        return unmask
