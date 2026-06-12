"""Tests for src/data/vocabulary.py."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.vocabulary import (  # noqa: E402
    FootballVocab,
    SPECIAL_TOKENS,
    MINUTE_BUCKETS,
    FORM_BUCKETS,
    ELO_BUCKETS,
    PLAYER_FALLBACK_TOKENS,
    bucketize_minute,
    bucketize_form,
    bucketize_elo,
    player_token,
    team_token,
    tournament_token,
    assign_player_tier,
)


# ---------------------------------------------------------------------------
# Bucketization functions
# ---------------------------------------------------------------------------

def test_bucketize_minute_boundaries():
    assert bucketize_minute(0) == "MIN_0_15"
    assert bucketize_minute(15) == "MIN_0_15"
    assert bucketize_minute(16) == "MIN_16_30"
    assert bucketize_minute(45) == "MIN_31_45"
    assert bucketize_minute(46) == "MIN_46_60"
    assert bucketize_minute(90) == "MIN_76_90"
    assert bucketize_minute(95) == "MIN_EXTRA_TIME"
    assert bucketize_minute(125) == "MIN_PENALTIES"


def test_bucketize_form_levels():
    assert bucketize_form(0.0) == "FORM_VERY_LOW"
    assert bucketize_form(1.0) == "FORM_LOW"
    assert bucketize_form(1.5) == "FORM_MID"
    assert bucketize_form(2.0) == "FORM_HIGH"
    assert bucketize_form(3.0) == "FORM_VERY_HIGH"


def test_bucketize_elo_snaps_to_bucket():
    assert bucketize_elo(1850) == "ELO_BUCKET_1800"
    assert bucketize_elo(1899) == "ELO_BUCKET_1800"
    assert bucketize_elo(1900) == "ELO_BUCKET_1900"
    # Clamping
    assert bucketize_elo(500) == "ELO_BUCKET_1100"
    assert bucketize_elo(3000) == "ELO_BUCKET_2400"


def test_assign_player_tier():
    assert assign_player_tier(300) == 1
    assert assign_player_tier(150) == 2
    assert assign_player_tier(50) == 3
    assert assign_player_tier(20) == 4
    assert assign_player_tier(5) == 5
    assert assign_player_tier(0) == 5


# ---------------------------------------------------------------------------
# Vocabulary construction
# ---------------------------------------------------------------------------

def _sample_vocab() -> FootballVocab:
    """Small toy vocab for testing."""
    return FootballVocab.build_from_data(
        teams=["Argentina", "Brazil", "France", "Real Madrid"],
        tournaments=["FIFA World Cup", "Champions League"],
        stages=["group", "r16", "final", "regular_season"],
        player_appearances={
            "1001": 250,    # tier 1, dedicated
            "1002": 150,    # tier 2, dedicated
            "1003": 50,     # tier 3, dedicated
            "1004": 15,     # tier 4, dedicated (>=10 threshold)
            "1005": 5,      # tier 5, fallback only
            "1006": 1,      # tier 5, fallback only
        },
        player_positions={
            "1001": "FW", "1002": "MF", "1003": "GK", "1004": "DF",
            "1005": "FW", "1006": "MF",
        },
        k_player_threshold=10,
    )


def test_vocab_contains_all_specials():
    v = _sample_vocab()
    for tok in SPECIAL_TOKENS:
        assert tok in v.token_to_id, f"missing special token {tok}"


def test_vocab_contains_all_buckets():
    v = _sample_vocab()
    for tok in MINUTE_BUCKETS + FORM_BUCKETS + ELO_BUCKETS + PLAYER_FALLBACK_TOKENS:
        assert tok in v.token_to_id, f"missing bucket token {tok}"


def test_vocab_contains_teams():
    v = _sample_vocab()
    assert team_token("Argentina") in v.token_to_id
    assert team_token("Real Madrid") in v.token_to_id


def test_vocab_dedicated_players_above_threshold():
    v = _sample_vocab()
    assert player_token("1001") in v.token_to_id
    assert player_token("1002") in v.token_to_id
    assert player_token("1003") in v.token_to_id
    assert player_token("1004") in v.token_to_id  # 15 >= 10
    assert player_token("1005") not in v.token_to_id  # 5 < 10
    assert player_token("1006") not in v.token_to_id


def test_unique_ids():
    v = _sample_vocab()
    ids = list(v.token_to_id.values())
    assert len(ids) == len(set(ids)), "Duplicate ids in vocabulary"
    assert min(ids) == 0
    assert max(ids) == len(ids) - 1


# ---------------------------------------------------------------------------
# Player encoding with hierarchical fallback
# ---------------------------------------------------------------------------

def test_encode_player_dedicated():
    v = _sample_vocab()
    expected_id = v.token_to_id[player_token("1001")]
    assert v.encode_player("1001") == expected_id


def test_encode_player_fallback_known_player():
    """Player 1005 has 5 matches → no dedicated token → use fallback FW tier 5."""
    v = _sample_vocab()
    fallback_id = v.encode_player("1005")
    assert fallback_id == v.token_to_id["POS_FW_TIER_5"]


def test_encode_player_unseen_with_position():
    """Player 9999 never seen → use given position, tier 5."""
    v = _sample_vocab()
    unseen_id = v.encode_player("9999", position="DF")
    assert unseen_id == v.token_to_id["POS_DF_TIER_5"]


def test_encode_player_unseen_unknown_position():
    v = _sample_vocab()
    unseen_id = v.encode_player("9999")
    assert unseen_id == v.token_to_id["POS_UNKNOWN_TIER_5"]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip():
    v1 = _sample_vocab()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    try:
        v1.save(path)
        v2 = FootballVocab.load(path)
        assert v1.token_to_id == v2.token_to_id
        assert v1.k_player_threshold == v2.k_player_threshold
        # PlayerInfo equivalence
        for pid, info1 in v1.player_info.items():
            info2 = v2.player_info[pid]
            assert info1.position == info2.position
            assert info1.tier == info2.tier
    finally:
        Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Stats / introspection
# ---------------------------------------------------------------------------

def test_stats_breakdown():
    v = _sample_vocab()
    s = v.stats()
    assert s["TOTAL"] == len(v)
    assert s["special"] == len(SPECIAL_TOKENS)
    assert s["team"] == 4
    assert s["player_specific"] == 4   # only players with >=10
    assert s["tournament"] == 2
