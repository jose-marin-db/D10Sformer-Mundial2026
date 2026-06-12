"""
Monte Carlo simulator for the WC 2026.

Key design choices (justified):

1. **Stochastic sampling, NEVER argmax.** Each match outcome is drawn from the
   predicted probability distribution. Doing argmax → deterministic = 1 single
   bracket → useless for tournament prediction. Monte Carlo over the soft
   distribution gives proper marginal probabilities.

2. **Independent matches.** We do NOT update team form/ELO across simulated
   matches inside one Monte Carlo run. The features are taken at WC kickoff
   (last known values per team). This is a simplification documented in the
   paper: incorporating ELO/form propagation per simulated match would
   require ~50× more compute and a calibrated update model.

3. **Knockout = no draws.** If the predicted distribution gives draw, we
   redistribute the draw mass equally between home and away (50/50 over
   penalties). This is a standard approximation (e.g., FiveThirtyEight uses
   similar logic).

4. **Goals.** For scoring tallies, we sample from the marginal goal
   distributions implied by P(home) > P(away). A simple Poisson with rate
   derived from team ELO + form. NOT the Transformer's Score head (we chose
   LogReg as the engine per Vic's decision).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .bracket import (
    WC2026_GROUPS, GROUP_NAMES, ALL_KNOCKOUT_MATCHES,
    BracketMatch, parse_third_slot_eligible_groups,
    ROUND_OF_32, ROUND_OF_16, QUARTERFINALS, SEMIFINALS, THIRD_PLACE, FINAL,
    round_of_match,
)


# ---------------------------------------------------------------------------
# Type alias: predictor signature
# ---------------------------------------------------------------------------

# A predictor takes (team_a, team_b) and returns np.array([p_home, p_draw, p_away])
# It should also accept optional 'venue' kwarg ('home', 'away', 'neutral').
Predictor = Callable[..., np.ndarray]


# ---------------------------------------------------------------------------
# Precomputed predictor (optimization for Monte Carlo)
# ---------------------------------------------------------------------------

class PrecomputedPredictor:
    """Caches predictions for every directed pair of teams.

    With 48 teams there are 48*47 = 2256 directed pairs. Pre-computing them
    once turns each Monte Carlo simulation into a chain of dict lookups,
    cutting 10k-sim runtime from ~45 min to ~2-3 min.

    The wrapped raw predictor is called only once per pair at construction.
    """

    def __init__(self, raw_predictor: Predictor, teams: list[str],
                  venue: str = "neutral", verbose: bool = False):
        self.teams = list(teams)
        self.venue = venue
        self.table: dict[tuple[str, str], np.ndarray] = {}
        n = len(teams) * (len(teams) - 1)
        if verbose:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=n, desc="Precomputing predictions")
            except ImportError:
                pbar = None
        for a in teams:
            for b in teams:
                if a == b:
                    continue
                self.table[(a, b)] = raw_predictor(a, b, venue=venue)
                if verbose and pbar is not None:
                    pbar.update(1)
        if verbose and pbar is not None:
            pbar.close()

    def __call__(self, team_a: str, team_b: str, venue: str = "neutral") -> np.ndarray:
        # We ignore the `venue` argument because the table was built with a fixed one.
        return self.table[(team_a, team_b)]


# ---------------------------------------------------------------------------
# Fixed results — for live updates after real matches
# ---------------------------------------------------------------------------

@dataclass
class FixedResults:
    """Real-world results already known. Used to condition Monte Carlo on
    matches that have already been played.

    For group matches we identify by (group, frozenset({team_a, team_b})) so
    that the caller doesn't have to remember the nominal home/away order.
    For knockouts we identify by `match_id`.
    """
    group_matches: dict[tuple[str, frozenset], dict] = field(default_factory=dict)
    knockout_winners: dict[int, str] = field(default_factory=dict)

    def add_group_match(self, group: str, home: str, away: str,
                         home_score: int, away_score: int) -> None:
        key = (group, frozenset({home, away}))
        self.group_matches[key] = {
            "home": home, "away": away,
            "home_score": int(home_score), "away_score": int(away_score),
        }

    def add_knockout_result(self, match_id: int, winner: str) -> None:
        self.knockout_winners[int(match_id)] = winner

    def get_group_match(self, group: str, team_a: str, team_b: str) -> Optional[dict]:
        return self.group_matches.get((group, frozenset({team_a, team_b})))

    def get_knockout_winner(self, match_id: int) -> Optional[str]:
        return self.knockout_winners.get(int(match_id))

    def has_any(self) -> bool:
        return bool(self.group_matches) or bool(self.knockout_winners)

    def summary(self) -> str:
        return (f"FixedResults(group_matches={len(self.group_matches)}, "
                f"knockouts={len(self.knockout_winners)})")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GroupStanding:
    """One team's record in a group after the 3 group matches."""
    team: str
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against

    @property
    def sort_key(self) -> tuple:
        """FIFA tiebreaker: points, then goal diff, then GF."""
        return (-self.points, -self.goal_diff, -self.goals_for)


@dataclass
class TournamentResult:
    """The full set of match outcomes from one Monte Carlo iteration."""
    group_standings: dict[str, list[GroupStanding]] = field(default_factory=dict)
    knockout_winners: dict[int, str] = field(default_factory=dict)       # match_id → winner
    knockout_losers:  dict[int, str] = field(default_factory=dict)
    progressions: dict[str, str] = field(default_factory=dict)            # team → furthest stage
    champion: Optional[str] = None
    runner_up: Optional[str] = None
    third_place: Optional[str] = None


# Stages in canonical order (lowest → highest).
# Note: 'third_place' and 'final' are NOT sequential — they are sibling outcomes
# of the semifinal (loser → third_place, winner → final). 'champion' is reached
# only by winning the final.
STAGES = ["group", "round_of_32", "round_of_16", "quarterfinals", "semifinals",
          "third_place", "final", "champion"]


# Mapping from a team's "furthest stage" to the set of stages they reached.
# Used by MonteCarloAggregation.to_dataframe() to compute cumulative P(reaches X).
REACHED_AT_LEAST: dict[str, set[str]] = {
    "group":         {"group"},
    "round_of_32":   {"group", "round_of_32"},
    "round_of_16":   {"group", "round_of_32", "round_of_16"},
    "quarterfinals": {"group", "round_of_32", "round_of_16", "quarterfinals"},
    "semifinals":    {"group", "round_of_32", "round_of_16", "quarterfinals", "semifinals"},
    "third_place":   {"group", "round_of_32", "round_of_16", "quarterfinals", "semifinals", "third_place"},
    "final":         {"group", "round_of_32", "round_of_16", "quarterfinals", "semifinals", "final"},
    "champion":      {"group", "round_of_32", "round_of_16", "quarterfinals", "semifinals", "final", "champion"},
}


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_match_result(p_home: float, p_draw: float, p_away: float,
                         rng: np.random.Generator) -> str:
    """Sample 'home_win' / 'draw' / 'away_win' according to predicted probs."""
    r = rng.random()
    if r < p_home:
        return "home_win"
    if r < p_home + p_draw:
        return "draw"
    return "away_win"


def sample_knockout_winner(p_home: float, p_draw: float, p_away: float,
                            rng: np.random.Generator) -> str:
    """Knockout: distribute the draw mass equally to home and away (penalty shootout)."""
    p_h_eff = p_home + p_draw / 2
    return "home_win" if rng.random() < p_h_eff else "away_win"


def sample_goals(p_home: float, p_draw: float, p_away: float,
                  base_total: float = 2.5,
                  rng: Optional[np.random.Generator] = None) -> tuple[int, int]:
    """Quick & dirty: sample (home, away) goals from independent Poisson rates
    derived from the win probabilities.

    Calibration: total goals expected = base_total (~2.5 is the FIFA average
    in WCs since 2010). We split this total proportionally to (p_home + p_draw/2)
    vs. (p_away + p_draw/2).
    """
    if rng is None:
        rng = np.random.default_rng()
    p_h_eff = p_home + p_draw / 2
    p_a_eff = p_away + p_draw / 2
    rate_h = base_total * p_h_eff
    rate_a = base_total * p_a_eff
    return int(rng.poisson(rate_h)), int(rng.poisson(rate_a))


# ---------------------------------------------------------------------------
# Group stage simulation
# ---------------------------------------------------------------------------

def simulate_group(
    group_name: str,
    teams: list[str],
    predictor: Predictor,
    rng: np.random.Generator,
    fixed_results: Optional[FixedResults] = None,
) -> list[GroupStanding]:
    """Simulate the 6 matches of a group; return ranked standings.

    If `fixed_results` contains a real-world result for any of the 6 pairs,
    that match is NOT simulated — the known score is applied instead.
    """
    standings = {t: GroupStanding(team=t) for t in teams}

    # All 6 round-robin pairs (each pair plays once at WC group stage)
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]

            # Check if this match is already fixed (real result available)
            fixed = fixed_results.get_group_match(group_name, a, b) if fixed_results else None
            if fixed is not None:
                # `fixed` carries nominal home/away + scores. Apply as-is.
                # We map back to (a, b) ordering for the standings update.
                if fixed["home"] == a:
                    ga_h, ga_a = fixed["home_score"], fixed["away_score"]
                else:
                    # fixed['home'] == b → swap
                    ga_h, ga_a = fixed["away_score"], fixed["home_score"]
                if ga_h > ga_a:
                    outcome = "home_win"
                elif ga_h < ga_a:
                    outcome = "away_win"
                else:
                    outcome = "draw"
            else:
                probs = predictor(a, b, venue="neutral")
                p_h, p_d, p_a = float(probs[0]), float(probs[1]), float(probs[2])
                outcome = sample_match_result(p_h, p_d, p_a, rng)
                ga_h, ga_a = sample_goals(p_h, p_d, p_a, rng=rng)
                # Enforce outcome consistency
                if outcome == "home_win" and ga_h <= ga_a:
                    ga_h = ga_a + 1
                elif outcome == "away_win" and ga_a <= ga_h:
                    ga_a = ga_h + 1
                elif outcome == "draw" and ga_h != ga_a:
                    ga_h = ga_a = max(ga_h, ga_a)

            standings[a].goals_for += ga_h
            standings[a].goals_against += ga_a
            standings[b].goals_for += ga_a
            standings[b].goals_against += ga_h
            if outcome == "home_win":
                standings[a].points += 3
            elif outcome == "away_win":
                standings[b].points += 3
            else:
                standings[a].points += 1
                standings[b].points += 1

    sorted_standings = sorted(standings.values(), key=lambda s: s.sort_key)
    return sorted_standings


# ---------------------------------------------------------------------------
# Best-thirds selection + slot assignment
# ---------------------------------------------------------------------------

def select_best_thirds(all_standings: dict[str, list[GroupStanding]],
                       n: int = 8) -> list[tuple[str, GroupStanding]]:
    """Return the top-n 3rd-placed teams across all groups.

    Returns:
        List of (group_name, standing) tuples for the n best 3rd-placed teams.
    """
    thirds = [(grp, standings[2]) for grp, standings in all_standings.items()
              if len(standings) >= 3]
    thirds.sort(key=lambda x: x[1].sort_key)
    return thirds[:n]


def assign_thirds_to_slots(best_thirds: list[tuple[str, GroupStanding]],
                            r32: list[BracketMatch],
                            rng: np.random.Generator) -> dict[str, str]:
    """Match the 8 best 3rd-placed teams to the 8 third-slots in the R32 bracket.

    Each third-slot has a list of eligible groups (e.g., '3rd_Group_A_B_C_D_F').
    We solve this as a constrained assignment: try permutations until we find
    one where every third-slot is filled by an eligible team. With 8 slots and
    8 candidates this is fast (~40k permutations max).

    Returns a dict slot_str → team_name.
    """
    third_slots = []   # (match_id, slot_str, eligible_groups)
    for m in r32:
        for slot in (m.slot_a, m.slot_b):
            if slot.startswith("3rd_Group_"):
                third_slots.append((m.match_id, slot, set(parse_third_slot_eligible_groups(slot))))

    n_slots = len(third_slots)
    if n_slots != len(best_thirds):
        # Should be 8 == 8; if not, signal a config bug.
        raise RuntimeError(
            f"Slot count {n_slots} != best thirds count {len(best_thirds)}"
        )

    candidates = list(best_thirds)
    rng.shuffle(candidates)

    # Try a randomised greedy assignment: for each slot, pick a compatible candidate
    # at random. If we get stuck, restart up to N times.
    for attempt in range(200):
        used = [False] * len(candidates)
        assignment: dict[str, str] = {}
        ok = True
        order = list(range(n_slots))
        rng.shuffle(order)
        for slot_idx in order:
            _mid, slot_str, eligible = third_slots[slot_idx]
            choices = [i for i, (g, s) in enumerate(candidates)
                       if not used[i] and g in eligible]
            if not choices:
                ok = False; break
            pick = int(rng.choice(choices))
            used[pick] = True
            assignment[slot_str] = candidates[pick][1].team
        if ok:
            return assignment
        rng.shuffle(candidates)

    # Fallback: ignore eligibility (rare). Shouldn't happen with a well-formed bracket.
    fallback = {}
    for (_mid, slot_str, _), (_g, s) in zip(third_slots, candidates):
        fallback[slot_str] = s.team
    return fallback


# ---------------------------------------------------------------------------
# Knockout simulation
# ---------------------------------------------------------------------------

def resolve_slot(slot: str, ctx: dict) -> str:
    """Resolve a slot string to a concrete team name using the context dict.

    `ctx` is a dict that should contain entries like:
        '1st_Group_A' → team_name
        '2nd_Group_A' → team_name
        '3rd_Group_A_B_C_D_F' → team_name   (set by assign_thirds_to_slots)
        'winner_match_73' → team_name (set as knockout progresses)
        'loser_match_101' → team_name (only for the 3rd-place playoff)
    """
    if slot in ctx:
        return ctx[slot]
    raise KeyError(f"Slot not resolved: {slot}")


def simulate_knockout(
    ctx: dict[str, str],
    predictor: Predictor,
    rng: np.random.Generator,
    result: TournamentResult,
    fixed_results: Optional[FixedResults] = None,
) -> TournamentResult:
    """Simulate every knockout match in order, updating ctx as we go.

    If `fixed_results.knockout_winners[match_id]` is set, that winner is used
    instead of sampling.
    """
    for match in ALL_KNOCKOUT_MATCHES:
        team_a = resolve_slot(match.slot_a, ctx)
        team_b = resolve_slot(match.slot_b, ctx)

        fixed_winner = (fixed_results.get_knockout_winner(match.match_id)
                        if fixed_results else None)
        if fixed_winner is not None:
            if fixed_winner not in (team_a, team_b):
                # Inconsistent fix (e.g., the group stage produced a different
                # set of qualifiers than the real world). Fall back to sampling.
                probs = predictor(team_a, team_b, venue="neutral")
                p_h, p_d, p_a = float(probs[0]), float(probs[1]), float(probs[2])
                outcome = sample_knockout_winner(p_h, p_d, p_a, rng)
                winner, loser = (team_a, team_b) if outcome == "home_win" else (team_b, team_a)
            else:
                winner = fixed_winner
                loser  = team_b if winner == team_a else team_a
        else:
            probs = predictor(team_a, team_b, venue="neutral")
            p_h, p_d, p_a = float(probs[0]), float(probs[1]), float(probs[2])
            outcome = sample_knockout_winner(p_h, p_d, p_a, rng)
            if outcome == "home_win":
                winner, loser = team_a, team_b
            else:
                winner, loser = team_b, team_a

        result.knockout_winners[match.match_id] = winner
        result.knockout_losers[match.match_id]  = loser
        ctx[f"winner_match_{match.match_id}"] = winner
        ctx[f"loser_match_{match.match_id}"]  = loser

        # Track furthest stage reached by each team
        round_name = round_of_match(match.match_id)
        for t in (team_a, team_b):
            current = result.progressions.get(t, "group")
            # Both teams "reached" this round (the winner advances, the loser was here)
            if STAGES.index(round_name) > STAGES.index(current):
                result.progressions[t] = round_name
        # The winner reaches at least the NEXT round; we'll update when they play it.
        # (Champion gets bumped at the end.)

    # Determine champion / runner-up / third place
    result.champion = result.knockout_winners[FINAL.match_id]
    result.runner_up = result.knockout_losers[FINAL.match_id]
    result.third_place = result.knockout_winners[THIRD_PLACE.match_id]
    # Champion has reached the highest stage
    result.progressions[result.champion] = "champion"
    # Finalist has reached at least 'final'
    if STAGES.index(result.progressions.get(result.runner_up, "group")) < STAGES.index("final"):
        result.progressions[result.runner_up] = "final"
    return result


# ---------------------------------------------------------------------------
# End-to-end: one full tournament simulation
# ---------------------------------------------------------------------------

def simulate_tournament(
    predictor: Predictor,
    rng: np.random.Generator,
    fixed_results: Optional[FixedResults] = None,
) -> TournamentResult:
    """Simulate one full WC: groups → best thirds → knockouts → final.

    Args:
        predictor: callable (a, b, venue=...) → np.array([P_home, P_draw, P_away]).
        rng: numpy Generator.
        fixed_results: optional real-world results that override sampling.
    """
    result = TournamentResult()

    # --- 1. Group stage ---
    for grp, teams in WC2026_GROUPS.items():
        standings = simulate_group(grp, teams, predictor, rng,
                                    fixed_results=fixed_results)
        result.group_standings[grp] = standings

    # --- 2. Build context for knockouts ---
    ctx: dict[str, str] = {}
    for grp, standings in result.group_standings.items():
        ctx[f"1st_{grp}"] = standings[0].team
        ctx[f"2nd_{grp}"] = standings[1].team
        # Mark teams who exited at group stage (3rd or 4th)
        # 3rds are eligible for the wild card; 4ths are out.
        for s in standings[3:]:
            result.progressions[s.team] = "group"
        # Mark teams advancing to round_of_32 (top 2 + maybe 3rd)
        for s in standings[:2]:
            result.progressions[s.team] = "round_of_32"

    # --- 3. Pick 8 best thirds & assign to slots ---
    best_thirds = select_best_thirds(result.group_standings, n=8)
    third_assignment = assign_thirds_to_slots(best_thirds, ROUND_OF_32, rng)
    ctx.update(third_assignment)
    for slot, team in third_assignment.items():
        result.progressions[team] = "round_of_32"
    # 3rd-placed teams NOT picked → out at group stage
    picked_thirds = set(third_assignment.values())
    for grp, standings in result.group_standings.items():
        if len(standings) >= 3 and standings[2].team not in picked_thirds:
            result.progressions[standings[2].team] = "group"

    # --- 4. Knockouts ---
    simulate_knockout(ctx, predictor, rng, result, fixed_results=fixed_results)

    return result


# ---------------------------------------------------------------------------
# Monte Carlo aggregation
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloAggregation:
    n_iters: int
    progression_counts: dict[str, dict[str, int]]  # team → {stage → count}
    champion_counts: dict[str, int]                # team → count of championships
    runner_up_counts: dict[str, int]
    third_place_counts: dict[str, int]

    def to_dataframe(self) -> pd.DataFrame:
        """Returns a DataFrame: row per team, columns are P(reaches stage X).

        Uses the REACHED_AT_LEAST mapping so that:
          - a team whose furthest stage was 'final' counts toward P_semis (yes,
            they reached semis) AND P_final (they were in the final);
          - a team whose furthest stage was 'third_place' counts toward P_semis
            (they reached semis as a loser) but NOT toward P_final.
        """
        n = self.n_iters
        rows = []
        for team in sorted(self.progression_counts.keys()):
            counts = self.progression_counts.get(team, {})
            stages_reached_count = {stage: 0 for stage in STAGES}
            for furthest_stage, c in counts.items():
                for stage in REACHED_AT_LEAST[furthest_stage]:
                    stages_reached_count[stage] += c
            rows.append({
                "team": team,
                "P_group_advance": stages_reached_count["round_of_32"] / n,
                "P_round_of_16":   stages_reached_count["round_of_16"] / n,
                "P_quarters":      stages_reached_count["quarterfinals"] / n,
                "P_semis":         stages_reached_count["semifinals"] / n,
                "P_final":         stages_reached_count["final"] / n,
                "P_champion":      stages_reached_count["champion"] / n,
            })
        return pd.DataFrame(rows).sort_values("P_champion", ascending=False).reset_index(drop=True)


def monte_carlo(
    predictor: Predictor,
    n_iters: int = 10000,
    seed: int = 42,
    progress: bool = True,
    fixed_results: Optional[FixedResults] = None,
) -> MonteCarloAggregation:
    """Run `n_iters` independent tournament simulations.

    Returns an aggregation object with marginal probabilities per team per stage.

    Args:
        predictor: pairwise predictor (consider wrapping with PrecomputedPredictor
            for 10×+ speedup).
        n_iters: number of tournament simulations.
        seed: master seed for reproducibility.
        progress: show a tqdm bar.
        fixed_results: real-world results to condition the simulation on.
            Useful during the live tournament: as matches finish, add them
            via `fixed_results.add_group_match(...)` /
            `add_knockout_result(...)` and re-run to get updated probabilities.
    """
    rng = np.random.default_rng(seed)
    progression_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    champion_counts: dict[str, int] = defaultdict(int)
    runner_up_counts: dict[str, int] = defaultdict(int)
    third_place_counts: dict[str, int] = defaultdict(int)

    iterator = range(n_iters)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"Monte Carlo (n={n_iters})")
        except ImportError:
            pass

    for _ in iterator:
        result = simulate_tournament(predictor, rng, fixed_results=fixed_results)
        for team, stage in result.progressions.items():
            progression_counts[team][stage] += 1
        if result.champion is not None:
            champion_counts[result.champion] += 1
        if result.runner_up is not None:
            runner_up_counts[result.runner_up] += 1
        if result.third_place is not None:
            third_place_counts[result.third_place] += 1

    return MonteCarloAggregation(
        n_iters=n_iters,
        progression_counts={t: dict(d) for t, d in progression_counts.items()},
        champion_counts=dict(champion_counts),
        runner_up_counts=dict(runner_up_counts),
        third_place_counts=dict(third_place_counts),
    )
