"""Tests for src/data/tokenizer.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.vocabulary import FootballVocab  # noqa: E402
from data.tokenizer import (  # noqa: E402
    MatchTokenizer,
    MatchDocument,
    MatchEvent,
    PlayerRef,
    RollingFeatures,
    SEG_META,
    SEG_LINEUP_A,
    SEG_EVENTS,
)


def _build_test_vocab() -> FootballVocab:
    return FootballVocab.build_from_data(
        teams=["Argentina", "France", "Real Madrid", "Barcelona"],
        tournaments=["FIFA World Cup", "Champions League", "Friendly"],
        stages=["group", "final", "regular_season"],
        player_appearances={
            f"100{i}": 100 for i in range(1, 25)   # 24 dedicated players
        },
        player_positions={f"100{i}": "FW" for i in range(1, 25)},
        k_player_threshold=10,
    )


def _sample_lineup(prefix: int) -> list[PlayerRef]:
    """11 players starting at id 100<prefix> ... 100<prefix+10>."""
    return [PlayerRef(player_id=f"100{i+prefix}", position="FW") for i in range(11)]


# ---------------------------------------------------------------------------
# Basic tokenization
# ---------------------------------------------------------------------------

def test_minimal_match_is_tokenizable():
    """A match with only required fields should produce a valid token sequence."""
    vocab = _build_test_vocab()
    tok = MatchTokenizer(vocab)
    match = MatchDocument(
        tournament="Friendly",
        team_a="Argentina",
        team_b="France",
        venue="neutral",
    )
    out = tok.tokenize(match)
    # Must start with CLS, end with SEP + PREDICT_RESULT
    assert out.token_strings[0] == "[CLS]"
    assert out.token_strings[-1] == "[PREDICT_RESULT]"
    assert out.token_strings[-2] == "[SEP]"
    # Must contain venue and team tokens
    assert "VENUE_NEUTRAL" in out.token_strings
    assert "TEAM_ARGENTINA" in out.token_strings
    assert "TEAM_FRANCE" in out.token_strings
    # All ids must be valid (< vocab size)
    for tid in out.token_ids:
        assert 0 <= tid < len(vocab)


def test_lineup_a_is_mask_when_absent():
    """If lineup_a is None, the section becomes [LINEUP_A] [MASK]."""
    vocab = _build_test_vocab()
    tok = MatchTokenizer(vocab)
    match = MatchDocument(
        tournament="Friendly",
        team_a="Argentina",
        team_b="France",
        venue="home",
        lineup_a=None,
    )
    out = tok.tokenize(match)
    # Find [LINEUP_A], next token should be [MASK]
    idx = out.token_strings.index("[LINEUP_A]")
    assert out.token_strings[idx + 1] == "[MASK]"


def test_lineup_a_fully_tokenized_when_present():
    vocab = _build_test_vocab()
    tok = MatchTokenizer(vocab)
    lineup_a = _sample_lineup(1)
    match = MatchDocument(
        tournament="Friendly",
        team_a="Argentina",
        team_b="France",
        venue="home",
        lineup_a=lineup_a,
    )
    out = tok.tokenize(match)
    idx = out.token_strings.index("[LINEUP_A]")
    # Next 11 tokens should be player tokens
    for i in range(1, 12):
        assert out.token_strings[idx + i].startswith("PLAYER_") or out.token_strings[idx + i].startswith("POS_")


def test_explicit_mask_overrides_present_lineup():
    """If we force mask_lineup_a=True, the actual lineup is replaced."""
    vocab = _build_test_vocab()
    tok = MatchTokenizer(vocab, mask_lineup_a=True)
    match = MatchDocument(
        tournament="Friendly",
        team_a="Argentina",
        team_b="France",
        venue="home",
        lineup_a=_sample_lineup(1),
    )
    out = tok.tokenize(match)
    idx = out.token_strings.index("[LINEUP_A]")
    assert out.token_strings[idx + 1] == "[MASK]"


def test_features_section_omitted_when_none():
    """If features=None, the [FEATURES_*] section should not appear (saves budget)."""
    vocab = _build_test_vocab()
    tok = MatchTokenizer(vocab)
    match = MatchDocument(
        tournament="Friendly",
        team_a="Argentina",
        team_b="France",
        venue="home",
        features=None,
    )
    out = tok.tokenize(match)
    assert "[FEATURES_START]" not in out.token_strings
    assert "[FEATURES_END]" not in out.token_strings


def test_features_section_emitted_when_present():
    vocab = _build_test_vocab()
    tok = MatchTokenizer(vocab)
    feats = RollingFeatures(home_elo=2050, away_elo=1850, home_form_pts=2.0)
    match = MatchDocument(
        tournament="Friendly",
        team_a="Argentina",
        team_b="France",
        venue="home",
        features=feats,
    )
    out = tok.tokenize(match)
    assert "[FEATURES_START]" in out.token_strings
    assert "[FEATURES_END]" in out.token_strings
    assert "ELO_BUCKET_2000" in out.token_strings   # 2050 → 2000
    assert "ELO_BUCKET_1800" in out.token_strings   # 1850 → 1800
    assert "FORM_HIGH" in out.token_strings         # 2.0 → HIGH (< 2.4)


def test_events_section_with_player():
    vocab = _build_test_vocab()
    tok = MatchTokenizer(vocab)
    events = [
        MatchEvent(minute=23, team="a", event_type="goal", player_id="1001"),
        MatchEvent(minute=67, team="b", event_type="yellow_card", player_id="1015"),
    ]
    match = MatchDocument(
        tournament="Friendly",
        team_a="Argentina",
        team_b="France",
        venue="home",
        events=events,
    )
    out = tok.tokenize(match)
    assert "[EVENTS_START]" in out.token_strings
    assert "[EVENTS_END]" in out.token_strings
    assert "[TEAM_A_TURN]" in out.token_strings
    assert "[TEAM_B_TURN]" in out.token_strings
    assert "EVENT_GOAL" in out.token_strings
    assert "EVENT_YELLOW_CARD" in out.token_strings
    assert "MIN_16_30" in out.token_strings   # min 23
    assert "MIN_61_75" in out.token_strings   # min 67


def test_target_result_encoded():
    vocab = _build_test_vocab()
    tok = MatchTokenizer(vocab)
    match = MatchDocument(
        tournament="Friendly", team_a="Argentina", team_b="France", venue="home",
        result="home_win", home_score=2, away_score=1,
    )
    out = tok.tokenize(match)
    assert out.target_result_id == vocab.encode("RESULT_HOME_WIN")
    assert out.target_score_id == vocab.encode("SCORE_2_1")


def test_score_clamped_to_5():
    vocab = _build_test_vocab()
    tok = MatchTokenizer(vocab)
    match = MatchDocument(
        tournament="Friendly", team_a="A", team_b="B", venue="home",
        home_score=8, away_score=0, result="home_win",
    )
    out = tok.tokenize(match)
    assert out.target_score_id == vocab.encode("SCORE_5_0")


def test_segment_ids_consistent_length():
    vocab = _build_test_vocab()
    tok = MatchTokenizer(vocab)
    match = MatchDocument(
        tournament="Friendly", team_a="Argentina", team_b="France", venue="home",
        lineup_a=_sample_lineup(1),
        events=[MatchEvent(minute=10, team="a", event_type="goal", player_id="1001")],
    )
    out = tok.tokenize(match)
    assert len(out.segment_ids) == len(out.token_ids) == len(out.token_strings)


def test_truncation_keeps_cls_and_sep():
    """When a match exceeds max_seq_length, the [CLS] and [SEP][PREDICT_RESULT]
    structure must be preserved."""
    vocab = _build_test_vocab()
    tok = MatchTokenizer(vocab, max_seq_length=10)
    # Force a huge events list
    events = [MatchEvent(minute=i, team="a", event_type="goal", player_id="1001")
              for i in range(50)]
    match = MatchDocument(
        tournament="Friendly", team_a="Argentina", team_b="France", venue="home",
        events=events,
    )
    out = tok.tokenize(match)
    assert out.truncated
    assert len(out.token_ids) == 10
    assert out.token_strings[0] == "[CLS]"
    assert out.token_strings[-1] == "[PREDICT_RESULT]"
    assert out.token_strings[-2] == "[SEP]"


def test_decode_roundtrip():
    vocab = _build_test_vocab()
    tok = MatchTokenizer(vocab)
    match = MatchDocument(
        tournament="Friendly", team_a="Argentina", team_b="France", venue="home",
    )
    out = tok.tokenize(match)
    decoded = tok.decode_sequence(out.token_ids)
    assert decoded == out.token_strings
