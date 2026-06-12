"""WC 2026 Monte Carlo simulation.

Public API:
    WC2026_GROUPS              — dict of groups (English names)
    WC2026_GROUPS_RAW          — same, Spanish names
    SPANISH_TO_ENGLISH         — translation map
    ALL_KNOCKOUT_MATCHES       — full bracket
    monte_carlo                — main entry point
    simulate_tournament        — single iteration
    TournamentResult, MonteCarloAggregation
"""

from .bracket import (
    WC2026_GROUPS, WC2026_GROUPS_RAW, GROUP_NAMES, SPANISH_TO_ENGLISH,
    to_english, BracketMatch, ALL_KNOCKOUT_MATCHES,
    ROUND_OF_32, ROUND_OF_16, QUARTERFINALS, SEMIFINALS, THIRD_PLACE, FINAL,
    round_of_match, assert_bracket_consistency,
)
from .simulator import (
    GroupStanding, TournamentResult, MonteCarloAggregation,
    simulate_group, simulate_tournament, monte_carlo,
    sample_match_result, sample_knockout_winner, sample_goals,
    Predictor, STAGES,
    PrecomputedPredictor, FixedResults,
)

__all__ = [
    "WC2026_GROUPS", "WC2026_GROUPS_RAW", "GROUP_NAMES", "SPANISH_TO_ENGLISH",
    "to_english", "BracketMatch", "ALL_KNOCKOUT_MATCHES",
    "ROUND_OF_32", "ROUND_OF_16", "QUARTERFINALS", "SEMIFINALS",
    "THIRD_PLACE", "FINAL", "round_of_match", "assert_bracket_consistency",
    "GroupStanding", "TournamentResult", "MonteCarloAggregation",
    "simulate_group", "simulate_tournament", "monte_carlo",
    "sample_match_result", "sample_knockout_winner", "sample_goals",
    "Predictor", "STAGES",
    "PrecomputedPredictor", "FixedResults",
]
