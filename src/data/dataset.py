"""
PyTorch Dataset wrapping `MatchDocument`s and a `MatchTokenizer`.

Design:
- The Dataset owns the *list of MatchDocuments* (in-memory, since ~50k matches
  is trivially small) plus the tokenizer.
- `__getitem__` returns a Python dict (NOT yet padded, NOT yet masked). The
  MLMCollator does padding + MLM masking at batch construction time.

We pass through MatchDocuments (not parsed dicts) so subclasses can inject
custom pre-processing (e.g., dynamic mask of an entire section, swap teams,
etc.) without rewriting the tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from torch.utils.data import Dataset

from .tokenizer import MatchTokenizer, MatchDocument, TokenizationOutput


@dataclass
class MatchSample:
    """A single training example, BEFORE batching/padding/masking."""
    token_ids: list[int]
    segment_ids: list[int]
    target_result_id: int | None
    target_score_id: int | None
    length: int

    @classmethod
    def from_tokenization(cls, out: TokenizationOutput) -> "MatchSample":
        return cls(
            token_ids=out.token_ids,
            segment_ids=out.segment_ids,
            target_result_id=out.target_result_id,
            target_score_id=out.target_score_id,
            length=len(out.token_ids),
        )


class MatchDataset(Dataset):
    """In-memory dataset over `MatchDocument`s.

    Args:
        matches: a sequence of MatchDocument objects.
        tokenizer: a fitted MatchTokenizer.
        drop_no_target: if True, drop matches without a result (cannot be used
            for the result/score heads; still useful for pure MLM pre-training
            though — set False then).
    """

    def __init__(
        self,
        matches: Sequence[MatchDocument],
        tokenizer: MatchTokenizer,
        drop_no_target: bool = False,
    ):
        if drop_no_target:
            matches = [m for m in matches if m.result is not None]
        self.matches = list(matches)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.matches)

    def __getitem__(self, idx: int) -> MatchSample:
        match = self.matches[idx]
        out = self.tokenizer.tokenize(match)
        return MatchSample.from_tokenization(out)

    # ----- introspection -----

    def length_stats(self, sample_size: int | None = None) -> dict:
        """Compute length distribution. Useful for choosing max_seq_length / batch size."""
        import random
        idxs = range(len(self))
        if sample_size and sample_size < len(self):
            idxs = random.sample(list(idxs), sample_size)
        lengths = [len(self.tokenizer.tokenize(self.matches[i]).token_ids) for i in idxs]
        import statistics
        return {
            "n": len(lengths),
            "min": min(lengths),
            "max": max(lengths),
            "mean": statistics.mean(lengths),
            "median": statistics.median(lengths),
            "p90": sorted(lengths)[int(0.9 * len(lengths))],
            "p99": sorted(lengths)[int(0.99 * len(lengths))],
        }
