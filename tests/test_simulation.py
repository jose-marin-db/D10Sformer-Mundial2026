"""Tests for src/simulation/."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simulation.bracket import (  # noqa: E402
    WC2026_GROUPS, WC2026_GROUPS_RAW, SPANISH_TO_ENGLISH,
    ALL_KNOCKOUT_MATCHES, ROUND_OF_32, parse_third_slot_eligible_groups,
    assert_bracket_consistency, to_english, round_of_match,
)
from simulation.simulator import (  # noqa: E402
    GroupStanding, simulate_group, simulate_tournament, monte_carlo,
    sample_match_result, sample_knockout_winner, select_best_thirds,
    PrecomputedPredictor, FixedResults,
)


# ---------------------------------------------------------------------------
# Bracket structure
# ---------------------------------------------------------------------------

def test_bracket_is_consistent():
    assert_bracket_consistency()


def test_48_teams_in_groups():
    assert sum(len(g) for g in WC2026_GROUPS.values()) == 48


def test_all_teams_have_english_translations():
    for grp, teams in WC2026_GROUPS_RAW.items():
        for t in teams:
            tr = to_english(t)
            assert tr != "" and isinstance(tr, str)


def test_no_team_appears_twice():
    flat = [t for g in WC2026_GROUPS.values() for t in g]
    assert len(flat) == len(set(flat))


def test_parse_third_slot():
    assert parse_third_slot_eligible_groups("3rd_Group_A_B_C_D_F") == \
        ["Group_A", "Group_B", "Group_C", "Group_D", "Group_F"]


def test_round_of_match():
    assert round_of_match(73) == "round_of_32"
    assert round_of_match(89) == "round_of_16"
    assert round_of_match(97) == "quarterfinals"
    assert round_of_match(101) == "semifinals"
    assert round_of_match(103) == "third_place"
    assert round_of_match(104) == "final"
    with pytest.raises(ValueError):
        round_of_match(999)


def test_total_knockout_matches():
    # 16 + 8 + 4 + 2 + 1 (3rd place) + 1 (final) = 32
    assert len(ALL_KNOCKOUT_MATCHES) == 32


# ---------------------------------------------------------------------------
# Sampling primitives
# ---------------------------------------------------------------------------

def test_sample_match_result_respects_probs():
    """If P(home)=1, all samples should be home_win."""
    rng = np.random.default_rng(0)
    outcomes = [sample_match_result(1.0, 0.0, 0.0, rng) for _ in range(100)]
    assert all(o == "home_win" for o in outcomes)


def test_sample_match_result_approximate_distribution():
    """Many samples should produce close-to-target distribution."""
    rng = np.random.default_rng(42)
    n = 10000
    outcomes = [sample_match_result(0.5, 0.25, 0.25, rng) for _ in range(n)]
    c = Counter(outcomes)
    assert abs(c["home_win"] / n - 0.5) < 0.02
    assert abs(c["draw"] / n - 0.25) < 0.02
    assert abs(c["away_win"] / n - 0.25) < 0.02


def test_knockout_winner_no_draws():
    rng = np.random.default_rng(0)
    outcomes = [sample_knockout_winner(0.5, 0.3, 0.2, rng) for _ in range(1000)]
    assert all(o in ("home_win", "away_win") for o in outcomes)


def test_knockout_winner_draw_split_evenly():
    """P(home)=0, P(draw)=1, P(away)=0 → expect 50/50 home/away."""
    rng = np.random.default_rng(123)
    n = 10000
    outcomes = [sample_knockout_winner(0.0, 1.0, 0.0, rng) for _ in range(n)]
    c = Counter(outcomes)
    assert abs(c["home_win"] / n - 0.5) < 0.02


# ---------------------------------------------------------------------------
# Group + best-thirds
# ---------------------------------------------------------------------------

def _dummy_predictor(team_a, team_b, venue="neutral"):
    """Uniform predictor."""
    return np.array([1/3, 1/3, 1/3])


def test_simulate_group_returns_4_standings():
    rng = np.random.default_rng(0)
    teams = WC2026_GROUPS["Group_A"]
    st = simulate_group("Group_A", teams, _dummy_predictor, rng)
    assert len(st) == 4
    assert all(isinstance(s, GroupStanding) for s in st)
    # Points are between 0 and 9 (3 matches per team)
    assert all(0 <= s.points <= 9 for s in st)
    # Total points distributed = total points awarded
    # Each game = 3 (win/loss) or 2 (draw); 6 games per group → 12-18 points total
    total = sum(s.points for s in st)
    assert 12 <= total <= 18


def test_select_best_thirds_returns_n():
    rng = np.random.default_rng(0)
    all_standings = {}
    for grp, teams in WC2026_GROUPS.items():
        all_standings[grp] = simulate_group(grp, teams, _dummy_predictor, rng)
    thirds = select_best_thirds(all_standings, n=8)
    assert len(thirds) == 8
    # All entries are (group, GroupStanding)
    for grp, s in thirds:
        assert grp in WC2026_GROUPS
        assert isinstance(s, GroupStanding)


# ---------------------------------------------------------------------------
# Full tournament + Monte Carlo
# ---------------------------------------------------------------------------

def test_simulate_tournament_produces_champion():
    rng = np.random.default_rng(7)
    result = simulate_tournament(_dummy_predictor, rng)
    assert result.champion is not None
    assert result.runner_up is not None
    assert result.third_place is not None
    assert result.champion in result.progressions
    assert result.progressions[result.champion] == "champion"


def test_monte_carlo_smoke():
    """50 sims with a uniform predictor should produce a champions distribution."""
    agg = monte_carlo(_dummy_predictor, n_iters=50, seed=42, progress=False)
    assert agg.n_iters == 50
    total_titles = sum(agg.champion_counts.values())
    assert total_titles == 50
    # Convert to DF and check the column structure
    df = agg.to_dataframe()
    expected_cols = {"team", "P_group_advance", "P_round_of_16", "P_quarters",
                     "P_semis", "P_final", "P_champion"}
    assert expected_cols.issubset(df.columns)
    # P_champion should sum to ~1 over all teams that won at least once
    assert abs(df["P_champion"].sum() - 1.0) < 1e-9
    # Monotonicity: P_quarters >= P_semis >= P_final >= P_champion (cumulative)
    for _, row in df.iterrows():
        assert row["P_quarters"] >= row["P_semis"] - 1e-9
        assert row["P_semis"]   >= row["P_final"] - 1e-9
        assert row["P_final"]   >= row["P_champion"] - 1e-9


def test_strong_predictor_wins_more():
    """If we make Argentina deterministically win every match, Argentina should be champion."""
    rng = np.random.default_rng(0)
    def argentina_always(a, b, venue="neutral"):
        if a == "Argentina": return np.array([1.0, 0.0, 0.0])
        if b == "Argentina": return np.array([0.0, 0.0, 1.0])
        return np.array([1/3, 1/3, 1/3])
    agg = monte_carlo(argentina_always, n_iters=50, seed=1, progress=False)
    assert agg.champion_counts.get("Argentina", 0) == 50


# ---------------------------------------------------------------------------
# Precomputed predictor + fixed results (Fase 7)
# ---------------------------------------------------------------------------

def test_precomputed_predictor_matches_raw():
    """Precomputed predictor must return the same probs as the underlying raw one."""
    teams_subset = ["Argentina", "Brazil", "France", "Spain"]
    def raw(a, b, venue="neutral"):
        # Different probs per ordered pair
        return np.array([0.5 if a < b else 0.3,
                          0.2 if a < b else 0.3,
                          0.3 if a < b else 0.4])
    pred = PrecomputedPredictor(raw, teams_subset, venue="neutral")
    for a in teams_subset:
        for b in teams_subset:
            if a == b:
                continue
            np.testing.assert_allclose(pred(a, b), raw(a, b))


def test_precomputed_predictor_faster(monkeypatch):
    """Sanity: precomputed should be much faster than raw on many calls (smoke check)."""
    teams = ["A", "B", "C", "D"]
    call_count = {"n": 0}
    def slow_raw(a, b, venue="neutral"):
        call_count["n"] += 1
        return np.array([1/3, 1/3, 1/3])
    pred = PrecomputedPredictor(slow_raw, teams)
    # The precomputed should have called raw exactly 4*3 = 12 times
    assert call_count["n"] == 12
    # Subsequent 100 calls should NOT increment call_count
    n_before = call_count["n"]
    for _ in range(100):
        pred("A", "B")
    assert call_count["n"] == n_before


def test_fixed_results_group_match_symmetric():
    fr = FixedResults()
    fr.add_group_match("Group_A", "Argentina", "Brazil", home_score=2, away_score=1)
    # Lookup in either order returns the same record
    m1 = fr.get_group_match("Group_A", "Argentina", "Brazil")
    m2 = fr.get_group_match("Group_A", "Brazil", "Argentina")
    assert m1 == m2
    assert m1["home"] == "Argentina"
    assert m1["home_score"] == 2
    assert m1["away_score"] == 1


def test_fixed_group_match_overrides_sampling():
    """If we fix all 6 matches of a group with crushing scores for team X,
    team X should always be top-1 regardless of the underlying predictor."""
    fr = FixedResults()
    grp = "Group_A"
    teams = WC2026_GROUPS[grp]
    favorite = teams[0]
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            if a == favorite:
                fr.add_group_match(grp, a, b, home_score=5, away_score=0)
            elif b == favorite:
                fr.add_group_match(grp, a, b, home_score=0, away_score=5)
            else:
                fr.add_group_match(grp, a, b, home_score=1, away_score=1)

    # Use a uniform predictor (irrelevant — all matches are fixed)
    rng = np.random.default_rng(0)
    standings = simulate_group(grp, teams, _dummy_predictor, rng, fixed_results=fr)
    assert standings[0].team == favorite
    assert standings[0].points == 9   # 3 wins
    assert standings[0].goals_for == 15  # 3 × 5


def test_fixed_knockout_winner_respected():
    """If we fix match 73's winner, the simulator must use that winner."""
    # Build a minimal context: groups have 2 teams each (Argentina advances from A, Brazil from B)
    fr = FixedResults()
    # First we need full groups simulated so that the slot mapping resolves.
    # Easier: use a predictor that gives Argentina+Brazil the top finishes deterministically
    def predictor_for_setup(a, b, venue="neutral"):
        if "Argentina" in (a, b):
            return np.array([0.99, 0.005, 0.005]) if a == "Argentina" else np.array([0.005, 0.005, 0.99])
        return np.array([1/3, 1/3, 1/3])

    # Fix match 73 (2nd_Group_A vs 2nd_Group_B) to a specific outcome.
    # We don't know the 2nd-placed teams in advance, so we just check the
    # invariant: whoever the fixed winner is must be team_a or team_b of the match.
    # Easier: pick any consistent fix and verify it sticks across runs.
    rng = np.random.default_rng(7)
    # Run baseline
    res_base = simulate_tournament(predictor_for_setup, rng)
    winner_73_base = res_base.knockout_winners[73]
    teams_in_73 = (winner_73_base, res_base.knockout_losers[73])

    # Force the OTHER team to win
    forced_winner = teams_in_73[1]
    fr.add_knockout_result(73, forced_winner)
    # Re-simulate with the same seed and the fix applied
    rng2 = np.random.default_rng(7)
    res_fixed = simulate_tournament(predictor_for_setup, rng2, fixed_results=fr)
    # The winner of match 73 should now be the forced one
    assert res_fixed.knockout_winners[73] == forced_winner


def test_monte_carlo_with_fixed_results():
    """Monte Carlo with a fixed group result should produce consistent stats."""
    fr = FixedResults()
    # Fix all 6 matches of Group_A so that Mexico wins them all
    teams = WC2026_GROUPS["Group_A"]
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            if a == "Mexico":
                fr.add_group_match("Group_A", a, b, 3, 0)
            elif b == "Mexico":
                fr.add_group_match("Group_A", a, b, 0, 3)
            else:
                fr.add_group_match("Group_A", a, b, 1, 1)
    agg = monte_carlo(_dummy_predictor, n_iters=30, seed=0, progress=False,
                       fixed_results=fr)
    # Mexico must have advanced in 100% of the simulations
    counts = agg.progression_counts.get("Mexico", {})
    advanced = sum(c for s, c in counts.items() if s != "group")
    assert advanced == 30   # 100% advance rate
