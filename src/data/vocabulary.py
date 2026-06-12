"""
Football vocabulary for the D10Sformer Transformer.

Design principles:
    1. Hierarchical player encoding: frequent players (>=K matches) get a dedicated
       token; rare players fall back to a positional+tier token. This solves the
       "long tail" problem identified in Fase 0 (91% of players appeared in only
       1 competition) and gracefully handles unseen players at WC 2026.

    2. Bucketization of continuous quantities (time, ELO, form, recent goals)
       is part of the vocabulary itself — every bucket is a distinct token.

    3. The vocabulary is built ONCE from the training data and serialized.
       Inference uses the frozen vocabulary; out-of-vocabulary entities map to
       [UNK] or their positional fallback.

    4. Special tokens follow BERT conventions where possible: [CLS], [SEP], [MASK],
       [PAD], [UNK].

Vocabulary size (estimated):
    Specials                ~ 20
    Tournaments + stages    ~ 40
    Time buckets (15 min)   ~  9
    Venues                  ~  3
    Teams                   ~ 500
    Players (dedicated)     ~ 2,500
    Player fallback         ~ 20
    Events                  ~ 12
    Numeric buckets         ~ 80
    Result                  ~ 35
    ----------------------------
    TOTAL                   ~ 3,200
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Special tokens (mirror BERT conventions)
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = [
    "[PAD]",            # padding to fixed length
    "[UNK]",            # out-of-vocabulary
    "[CLS]",            # classification head reads from here
    "[SEP]",            # section separator
    "[MASK]",           # for MLM pretraining + missing-lineup masking
    "[MATCH_START]",
    "[MATCH_END]",
    "[LINEUP_A]",       # marker before team A lineup
    "[LINEUP_B]",
    "[BENCH_A]",
    "[BENCH_B]",
    "[CONTEXT_START]",  # marker for prior-matches context block
    "[CONTEXT_END]",
    "[FEATURES_START]", # marker for rolling-feature block
    "[FEATURES_END]",
    "[EVENTS_START]",
    "[EVENTS_END]",
    "[TEAM_A_TURN]",    # event belongs to team A
    "[TEAM_B_TURN]",
    "[PREDICT_RESULT]", # tells the model to output a prediction
]


# ---------------------------------------------------------------------------
# Bucketization schemes — fixed, encoded in vocabulary
# ---------------------------------------------------------------------------

# Decision 1 (Fase 2): 15-minute buckets
MINUTE_BUCKETS = [
    "MIN_0_15", "MIN_16_30", "MIN_31_45",
    "MIN_HALF_TIME",
    "MIN_46_60", "MIN_61_75", "MIN_76_90",
    "MIN_EXTRA_TIME",
    "MIN_PENALTIES",
]

def bucketize_minute(minute: int | None) -> str:
    """Map a raw minute to its 15-minute bucket token."""
    if minute is None:
        return "MIN_HALF_TIME"  # shouldn't happen but safe default
    if minute < 0:
        return "MIN_0_15"
    if minute <= 15:
        return "MIN_0_15"
    if minute <= 30:
        return "MIN_16_30"
    if minute <= 45:
        return "MIN_31_45"
    if minute <= 60:
        return "MIN_46_60"
    if minute <= 75:
        return "MIN_61_75"
    if minute <= 90:
        return "MIN_76_90"
    if minute <= 120:
        return "MIN_EXTRA_TIME"
    return "MIN_PENALTIES"


# Form (recent points percentage, 0.0 - 1.0)
FORM_BUCKETS = ["FORM_VERY_LOW", "FORM_LOW", "FORM_MID", "FORM_HIGH", "FORM_VERY_HIGH"]

def bucketize_form(form_pts_avg: float | None) -> str:
    """form_pts_avg in [0, 3] (avg points per match in window)."""
    if form_pts_avg is None:
        return "FORM_MID"
    if form_pts_avg < 0.6:
        return "FORM_VERY_LOW"
    if form_pts_avg < 1.2:
        return "FORM_LOW"
    if form_pts_avg < 1.8:
        return "FORM_MID"
    if form_pts_avg < 2.4:
        return "FORM_HIGH"
    return "FORM_VERY_HIGH"


# Recent goals scored (avg per match in window)
GOALS_BUCKETS = ["GOALS_VERY_LOW", "GOALS_LOW", "GOALS_MID", "GOALS_HIGH", "GOALS_VERY_HIGH"]

def bucketize_goals(goals_avg: float | None) -> str:
    if goals_avg is None:
        return "GOALS_MID"
    if goals_avg < 0.5:
        return "GOALS_VERY_LOW"
    if goals_avg < 1.0:
        return "GOALS_LOW"
    if goals_avg < 1.75:
        return "GOALS_MID"
    if goals_avg < 2.5:
        return "GOALS_HIGH"
    return "GOALS_VERY_HIGH"


# ELO buckets (every 100 points from 1100 to 2300)
ELO_BUCKET_EDGES = list(range(1100, 2401, 100))   # 1100, 1200, ..., 2400

def bucketize_elo(elo: float | None) -> str:
    if elo is None:
        return "ELO_BUCKET_1500"
    # Snap to nearest 100
    lower = max(ELO_BUCKET_EDGES[0], min(ELO_BUCKET_EDGES[-1], int(elo // 100) * 100))
    return f"ELO_BUCKET_{lower}"

ELO_BUCKETS = [f"ELO_BUCKET_{e}" for e in ELO_BUCKET_EDGES]


# Venues
VENUE_TOKENS = ["VENUE_HOME", "VENUE_AWAY", "VENUE_NEUTRAL"]


# Event types we keep (per Decision B of Fase 0)
EVENT_TOKENS = [
    "EVENT_GOAL",
    "EVENT_OWN_GOAL",
    "EVENT_PENALTY_GOAL",
    "EVENT_PENALTY_MISS",
    "EVENT_YELLOW_CARD",
    "EVENT_RED_CARD",
    "EVENT_SECOND_YELLOW",
    "EVENT_SUBSTITUTION_IN",
    "EVENT_SUBSTITUTION_OUT",
]


# Result tokens (target for fine-tuning)
RESULT_TOKENS = [
    "RESULT_HOME_WIN",
    "RESULT_DRAW",
    "RESULT_AWAY_WIN",
]

# Score buckets (the actual score — useful for auxiliary regression-as-classification)
# We bucket scores >5 as "5+" to limit vocab
SCORE_TOKENS = [f"SCORE_{a}_{b}"
                for a in range(0, 6) for b in range(0, 6)]


# Positions for player fallback (Decision 2)
POSITIONS = ["GK", "DF", "MF", "FW", "UNKNOWN"]
PLAYER_TIERS = [1, 2, 3, 4, 5]   # tier 1 = top, tier 5 = unknown

def player_fallback_token(position: str, tier: int) -> str:
    pos = position if position in POSITIONS else "UNKNOWN"
    tier = tier if tier in PLAYER_TIERS else 5
    return f"POS_{pos}_TIER_{tier}"

PLAYER_FALLBACK_TOKENS = [player_fallback_token(p, t)
                          for p in POSITIONS for t in PLAYER_TIERS]


# ---------------------------------------------------------------------------
# Helpers for entity → token strings
# ---------------------------------------------------------------------------

_INVALID_CHARS = re.compile(r"[^A-Z0-9_]+")

def slugify(name: str) -> str:
    """Convert 'Argentina' / 'CF Real Madrid' to 'ARGENTINA' / 'CF_REAL_MADRID'."""
    s = name.upper().replace(" ", "_").replace("-", "_")
    s = _INVALID_CHARS.sub("", s)
    return s

def team_token(team_name: str) -> str:
    return f"TEAM_{slugify(team_name)}"

def tournament_token(tournament_name: str) -> str:
    return f"TOURNAMENT_{slugify(tournament_name)}"

def stage_token(stage_name: str) -> str:
    return f"STAGE_{slugify(stage_name)}"

def player_token(player_id: int | str) -> str:
    """Token for a 'specific' player (has dedicated embedding)."""
    return f"PLAYER_{player_id}"


# ---------------------------------------------------------------------------
# Player frequency analysis (for hierarchical encoding)
# ---------------------------------------------------------------------------

@dataclass
class PlayerInfo:
    """Auxiliary info needed for the fallback when a player is rare."""
    player_id: int | str
    name: str = ""
    position: str = "UNKNOWN"   # one of POSITIONS
    tier: int = 5               # 1=top, 5=unknown


def assign_player_tier(n_matches: int) -> int:
    """Heuristic mapping from number of appearances to tier.

    Tier 1: 200+ matches (elite, played top leagues + national team frequently)
    Tier 2: 100-199
    Tier 3: 40-99
    Tier 4: 10-39
    Tier 5: < 10  (the fallback "unknown" bucket)
    """
    if n_matches >= 200:
        return 1
    if n_matches >= 100:
        return 2
    if n_matches >= 40:
        return 3
    if n_matches >= 10:
        return 4
    return 5


# ---------------------------------------------------------------------------
# Vocabulary class
# ---------------------------------------------------------------------------

@dataclass
class FootballVocab:
    """The full football vocabulary, with bidirectional lookup and hierarchical
    player encoding.

    Construction is done via `FootballVocab.build_from_data(...)` factory.
    Use `save()` / `load()` for persistence.
    """

    # Core mapping: token string -> integer id
    token_to_id: dict[str, int] = field(default_factory=dict)

    # Player metadata for fallback: player_id -> PlayerInfo
    # If a player_id is NOT in player_info, treat it as unseen.
    # If a player_id IS in player_info AND its token is in token_to_id → dedicated.
    # If a player_id IS in player_info AND its token is NOT in token_to_id → fallback to player_info.tier.
    player_info: dict[str, PlayerInfo] = field(default_factory=dict)

    # Threshold used during construction
    k_player_threshold: int = 10

    # ----- core API -----

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def id_to_token(self) -> dict[int, str]:
        return {i: t for t, i in self.token_to_id.items()}

    def encode(self, token: str) -> int:
        """String token -> id. Falls back to [UNK] if not present."""
        return self.token_to_id.get(token, self.token_to_id["[UNK]"])

    def decode(self, token_id: int) -> str:
        """Id -> string token. Returns '[UNK]' for unknown ids."""
        for tok, tid in self.token_to_id.items():  # small enough; O(N) is fine
            if tid == token_id:
                return tok
        return "[UNK]"

    def has(self, token: str) -> bool:
        return token in self.token_to_id

    # ----- player encoding with hierarchical fallback -----

    def encode_player(
        self,
        player_id: int | str,
        position: str = "UNKNOWN",
    ) -> int:
        """Hierarchical player encoding:
        1. If player has dedicated token → use it.
        2. Else, use positional fallback based on player_info (if known)
           or position arg (if unseen player).
        """
        player_id_str = str(player_id)
        specific_tok = player_token(player_id_str)
        if specific_tok in self.token_to_id:
            return self.token_to_id[specific_tok]

        # Fallback path
        if player_id_str in self.player_info:
            info = self.player_info[player_id_str]
            fallback_tok = player_fallback_token(info.position, info.tier)
        else:
            # Unseen player → use given position, tier 5 (unknown)
            fallback_tok = player_fallback_token(position, 5)

        return self.token_to_id[fallback_tok]

    def player_token_string(
        self,
        player_id: int | str,
        position: str = "UNKNOWN",
    ) -> str:
        """Same as encode_player but returns the token STRING (useful for debugging)."""
        player_id_str = str(player_id)
        if player_token(player_id_str) in self.token_to_id:
            return player_token(player_id_str)
        if player_id_str in self.player_info:
            info = self.player_info[player_id_str]
            return player_fallback_token(info.position, info.tier)
        return player_fallback_token(position, 5)

    # ----- introspection -----

    def stats(self) -> dict:
        """Vocabulary breakdown by category."""
        cats = defaultdict(int)
        for tok in self.token_to_id:
            if tok.startswith("[") and tok.endswith("]"):
                cats["special"] += 1
            elif tok.startswith("TEAM_"):
                cats["team"] += 1
            elif tok.startswith("PLAYER_"):
                cats["player_specific"] += 1
            elif tok.startswith("POS_"):
                cats["player_fallback"] += 1
            elif tok.startswith("TOURNAMENT_"):
                cats["tournament"] += 1
            elif tok.startswith("STAGE_"):
                cats["stage"] += 1
            elif tok.startswith("MIN_"):
                cats["time_bucket"] += 1
            elif tok.startswith("VENUE_"):
                cats["venue"] += 1
            elif tok.startswith("EVENT_"):
                cats["event"] += 1
            elif tok.startswith("FORM_"):
                cats["form_bucket"] += 1
            elif tok.startswith("GOALS_"):
                cats["goals_bucket"] += 1
            elif tok.startswith("ELO_BUCKET_"):
                cats["elo_bucket"] += 1
            elif tok.startswith("RESULT_"):
                cats["result"] += 1
            elif tok.startswith("SCORE_"):
                cats["score"] += 1
            else:
                cats["other"] += 1
        cats["TOTAL"] = len(self.token_to_id)
        return dict(cats)

    # ----- persistence -----

    def save(self, path: str | Path) -> None:
        payload = {
            "k_player_threshold": self.k_player_threshold,
            "token_to_id": self.token_to_id,
            "player_info": {
                pid: {"player_id": info.player_id, "name": info.name,
                      "position": info.position, "tier": info.tier}
                for pid, info in self.player_info.items()
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "FootballVocab":
        payload = json.loads(Path(path).read_text())
        player_info = {
            pid: PlayerInfo(player_id=d["player_id"], name=d["name"],
                            position=d["position"], tier=d["tier"])
            for pid, d in payload["player_info"].items()
        }
        return cls(
            token_to_id=payload["token_to_id"],
            player_info=player_info,
            k_player_threshold=payload.get("k_player_threshold", 10),
        )

    # ----- factory -----

    @classmethod
    def build_from_data(
        cls,
        teams: Iterable[str],
        tournaments: Iterable[str],
        stages: Iterable[str],
        player_appearances: dict[str, int],         # player_id_str -> # matches
        player_positions: dict[str, str] | None = None,  # player_id_str -> position
        player_names: dict[str, str] | None = None,
        k_player_threshold: int = 10,
    ) -> "FootballVocab":
        """Build a vocabulary from observed entities in the training data.

        Parameters
        ----------
        teams : iterable of str
            All team names that appear in the dataset (national + club).
        tournaments : iterable of str
            All tournament names.
        stages : iterable of str
            All stage names (e.g., 'group', 'r16', 'qf', 'sf', 'final', 'regular_season').
        player_appearances : dict
            Mapping player_id_str -> total number of matches in which the player appears.
            Players with count >= k_player_threshold get a dedicated token; others get
            their fallback.
        player_positions : dict, optional
            Mapping player_id_str -> position string (GK / DF / MF / FW).
            Used only for the fallback assignment.
        player_names : dict, optional
            For debugging: player_id_str -> human-readable name.
        k_player_threshold : int
            Minimum appearances for a dedicated player token.
        """
        player_positions = player_positions or {}
        player_names = player_names or {}

        # Order of insertion controls token id assignment (specials first, then categorical
        # buckets, then teams, players). This is purely for cleanliness — model is invariant
        # to id permutations.

        tok_list: list[str] = []

        # 1. Specials
        tok_list.extend(SPECIAL_TOKENS)

        # 2. Fixed bucketization schemes
        tok_list.extend(MINUTE_BUCKETS)
        tok_list.extend(FORM_BUCKETS)
        tok_list.extend(GOALS_BUCKETS)
        tok_list.extend(ELO_BUCKETS)
        tok_list.extend(VENUE_TOKENS)
        tok_list.extend(EVENT_TOKENS)
        tok_list.extend(RESULT_TOKENS)
        tok_list.extend(SCORE_TOKENS)

        # 3. Tournaments and stages
        for t in sorted(set(tournaments)):
            tok_list.append(tournament_token(t))
        for s in sorted(set(stages)):
            tok_list.append(stage_token(s))

        # 4. Teams
        for t in sorted(set(teams)):
            tok_list.append(team_token(t))

        # 5. Player fallback (always present, even if not all positions appear)
        tok_list.extend(PLAYER_FALLBACK_TOKENS)

        # 6. Dedicated players (those above threshold)
        dedicated_players = sorted(
            (pid for pid, n in player_appearances.items() if n >= k_player_threshold),
            key=lambda p: -player_appearances[p],   # most-frequent first
        )
        for pid in dedicated_players:
            tok_list.append(player_token(pid))

        # Build dict (preserve order, deduplicate just in case)
        seen = set()
        token_to_id: dict[str, int] = {}
        for tok in tok_list:
            if tok in seen:
                continue
            token_to_id[tok] = len(token_to_id)
            seen.add(tok)

        # Build PlayerInfo for ALL players we know about (needed for fallback even if
        # the player has a dedicated token — useful to remember position for downstream).
        player_info: dict[str, PlayerInfo] = {}
        for pid, n in player_appearances.items():
            player_info[str(pid)] = PlayerInfo(
                player_id=pid,
                name=player_names.get(pid, ""),
                position=player_positions.get(pid, "UNKNOWN"),
                tier=assign_player_tier(n),
            )

        return cls(
            token_to_id=token_to_id,
            player_info=player_info,
            k_player_threshold=k_player_threshold,
        )
