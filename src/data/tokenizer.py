"""
Match → token sequence conversion for the D10Sformer.

Converts a `MatchDocument` (which can be richly populated for StatsBomb matches
or sparse for pure international-results entries) into a sequence of token ids
that can be fed to the Transformer.

Schema of a tokenized match (sections in this fixed order):

    [CLS]
    TOURNAMENT_<...>  STAGE_<...>  VENUE_<HOME|AWAY|NEUTRAL>
    TEAM_<A>  TEAM_<B>
    [FEATURES_START]
        ELO_BUCKET_<A>  ELO_BUCKET_<B>
        FORM_<A_LEVEL>  FORM_<B_LEVEL>
        GOALS_<A_LEVEL>  GOALS_<B_LEVEL>
    [FEATURES_END]
    [LINEUP_A]  PLAYER_<...>  ... (or [MASK] if unknown)
    [BENCH_A]   PLAYER_<...>  ... (or [MASK])
    [LINEUP_B]  ...
    [BENCH_B]   ...
    [EVENTS_START]
        [TEAM_A_TURN] MIN_<bucket> EVENT_<type> PLAYER_<...>
        ...
    [EVENTS_END]
    [SEP]
    [PREDICT_RESULT]

Sections that have no data are either:
    - replaced by a single [MASK] (lineups / benches) — Decision 3 = C
    - omitted entirely (features / events) — saves sequence budget
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .vocabulary import (
    FootballVocab,
    bucketize_minute,
    bucketize_form,
    bucketize_goals,
    bucketize_elo,
    team_token,
    tournament_token,
    stage_token,
)


# ---------------------------------------------------------------------------
# Input data structures
# ---------------------------------------------------------------------------

@dataclass
class PlayerRef:
    """A reference to a player in a lineup or event."""
    player_id: int | str
    position: str = "UNKNOWN"   # GK / DF / MF / FW / UNKNOWN


@dataclass
class MatchEvent:
    """An in-match event (goal, card, substitution, etc.)."""
    minute: int                       # 0..120+, will be bucketized
    team: str                         # 'a' or 'b' — which team is the actor
    event_type: str                   # one of EVENT_<*> from vocabulary
    player_id: int | str | None = None
    player_position: str = "UNKNOWN"


@dataclass
class RollingFeatures:
    """Pre-computed rolling features for both teams (bucketized at tokenization time).

    All fields are optional — pass None to skip.
    """
    home_elo: float | None = None
    away_elo: float | None = None
    home_form_pts: float | None = None        # avg points per match
    away_form_pts: float | None = None
    home_recent_goals: float | None = None
    away_recent_goals: float | None = None


@dataclass
class MatchDocument:
    """Universal match representation. Fields can be missing depending on data source."""

    # Always required
    tournament: str
    team_a: str
    team_b: str
    venue: str                    # 'home' | 'away' | 'neutral' (perspective of team_a)

    # Optional
    stage: str | None = None      # 'group' | 'r16' | 'qf' | 'sf' | 'final' | 'regular_season' | None
    features: RollingFeatures | None = None
    lineup_a: list[PlayerRef] | None = None
    bench_a: list[PlayerRef] | None = None
    lineup_b: list[PlayerRef] | None = None
    bench_b: list[PlayerRef] | None = None
    events: list[MatchEvent] | None = None

    # Target (for training)
    result: str | None = None     # 'home_win' | 'draw' | 'away_win'
    home_score: int | None = None
    away_score: int | None = None


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

@dataclass
class TokenizationOutput:
    """Result of tokenizing a match."""
    token_ids: list[int]
    token_strings: list[str]          # parallel to token_ids — useful for debugging
    segment_ids: list[int]            # which segment each token belongs to (for embeddings)
    target_result_id: int | None = None
    target_score_id: int | None = None
    truncated: bool = False


# Segment IDs for the embedding layer (Fase 3 will use these)
SEG_META = 0          # CLS + tournament + stage + venue + teams
SEG_FEATURES = 1
SEG_LINEUP_A = 2
SEG_BENCH_A = 3
SEG_LINEUP_B = 4
SEG_BENCH_B = 5
SEG_EVENTS = 6
SEG_SEP = 7


class MatchTokenizer:
    """Converts MatchDocument → token id sequence using a fixed FootballVocab."""

    def __init__(
        self,
        vocab: FootballVocab,
        max_seq_length: int = 512,
        mask_lineup_a: bool = False,
        mask_bench_a: bool = False,
        mask_lineup_b: bool = False,
        mask_bench_b: bool = False,
        mask_features: bool = False,
        mask_events: bool = False,
    ):
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.mask_lineup_a = mask_lineup_a
        self.mask_bench_a = mask_bench_a
        self.mask_lineup_b = mask_lineup_b
        self.mask_bench_b = mask_bench_b
        self.mask_features = mask_features
        self.mask_events = mask_events

    # ----- main entry point -----

    def tokenize(self, match: MatchDocument) -> TokenizationOutput:
        tokens: list[str] = []
        segments: list[int] = []

        def add(token: str, seg: int) -> None:
            tokens.append(token)
            segments.append(seg)

        # ---- META section ----
        add("[CLS]", SEG_META)
        add(tournament_token(match.tournament), SEG_META)
        if match.stage:
            add(stage_token(match.stage), SEG_META)
        add(f"VENUE_{match.venue.upper()}", SEG_META)
        add(team_token(match.team_a), SEG_META)
        add(team_token(match.team_b), SEG_META)

        # ---- FEATURES section (optional, omit if absent and not forcing mask) ----
        if self.mask_features:
            add("[FEATURES_START]", SEG_FEATURES)
            add("[MASK]", SEG_FEATURES)
            add("[FEATURES_END]", SEG_FEATURES)
        elif match.features is not None:
            self._emit_features(match.features, add)

        # ---- LINEUP A + BENCH A ----
        add("[LINEUP_A]", SEG_LINEUP_A)
        if self.mask_lineup_a or match.lineup_a is None:
            add("[MASK]", SEG_LINEUP_A)
        else:
            for p in match.lineup_a:
                add(self.vocab.player_token_string(p.player_id, p.position), SEG_LINEUP_A)

        add("[BENCH_A]", SEG_BENCH_A)
        if self.mask_bench_a or match.bench_a is None:
            add("[MASK]", SEG_BENCH_A)
        else:
            for p in match.bench_a:
                add(self.vocab.player_token_string(p.player_id, p.position), SEG_BENCH_A)

        # ---- LINEUP B + BENCH B ----
        add("[LINEUP_B]", SEG_LINEUP_B)
        if self.mask_lineup_b or match.lineup_b is None:
            add("[MASK]", SEG_LINEUP_B)
        else:
            for p in match.lineup_b:
                add(self.vocab.player_token_string(p.player_id, p.position), SEG_LINEUP_B)

        add("[BENCH_B]", SEG_BENCH_B)
        if self.mask_bench_b or match.bench_b is None:
            add("[MASK]", SEG_BENCH_B)
        else:
            for p in match.bench_b:
                add(self.vocab.player_token_string(p.player_id, p.position), SEG_BENCH_B)

        # ---- EVENTS section (optional, omit if absent and not forcing mask) ----
        if self.mask_events:
            add("[EVENTS_START]", SEG_EVENTS)
            add("[MASK]", SEG_EVENTS)
            add("[EVENTS_END]", SEG_EVENTS)
        elif match.events:
            add("[EVENTS_START]", SEG_EVENTS)
            for ev in match.events:
                self._emit_event(ev, add)
            add("[EVENTS_END]", SEG_EVENTS)

        # ---- TRAILER ----
        add("[SEP]", SEG_SEP)
        add("[PREDICT_RESULT]", SEG_SEP)

        # ---- Encode and truncate ----
        truncated = False
        if len(tokens) > self.max_seq_length:
            # Keep [CLS] + as much as fits + force [SEP] [PREDICT_RESULT] at the end
            keep = self.max_seq_length - 2
            tokens = [tokens[0]] + tokens[1:keep] + ["[SEP]", "[PREDICT_RESULT]"]
            segments = [segments[0]] + segments[1:keep] + [SEG_SEP, SEG_SEP]
            truncated = True

        token_ids = [self.vocab.encode(t) for t in tokens]

        # ---- Targets (if available) ----
        target_result_id = None
        if match.result:
            res_tok = f"RESULT_{match.result.upper()}"
            target_result_id = self.vocab.encode(res_tok)

        target_score_id = None
        if match.home_score is not None and match.away_score is not None:
            hs = min(max(match.home_score, 0), 5)
            as_ = min(max(match.away_score, 0), 5)
            target_score_id = self.vocab.encode(f"SCORE_{hs}_{as_}")

        return TokenizationOutput(
            token_ids=token_ids,
            token_strings=tokens,
            segment_ids=segments,
            target_result_id=target_result_id,
            target_score_id=target_score_id,
            truncated=truncated,
        )

    # ----- helpers -----

    def _emit_features(self, feats: RollingFeatures, add) -> None:
        add("[FEATURES_START]", SEG_FEATURES)
        if feats.home_elo is not None:
            add(bucketize_elo(feats.home_elo), SEG_FEATURES)
        if feats.away_elo is not None:
            add(bucketize_elo(feats.away_elo), SEG_FEATURES)
        if feats.home_form_pts is not None:
            add(bucketize_form(feats.home_form_pts), SEG_FEATURES)
        if feats.away_form_pts is not None:
            add(bucketize_form(feats.away_form_pts), SEG_FEATURES)
        if feats.home_recent_goals is not None:
            add(bucketize_goals(feats.home_recent_goals), SEG_FEATURES)
        if feats.away_recent_goals is not None:
            add(bucketize_goals(feats.away_recent_goals), SEG_FEATURES)
        add("[FEATURES_END]", SEG_FEATURES)

    def _emit_event(self, ev: MatchEvent, add) -> None:
        turn_tok = "[TEAM_A_TURN]" if ev.team.lower() == "a" else "[TEAM_B_TURN]"
        add(turn_tok, SEG_EVENTS)
        add(bucketize_minute(ev.minute), SEG_EVENTS)
        add(f"EVENT_{ev.event_type.upper()}", SEG_EVENTS)
        if ev.player_id is not None:
            add(self.vocab.player_token_string(ev.player_id, ev.player_position), SEG_EVENTS)

    # ----- inverse for debugging -----

    def decode_sequence(self, token_ids: Iterable[int]) -> list[str]:
        return [self.vocab.decode(i) for i in token_ids]
