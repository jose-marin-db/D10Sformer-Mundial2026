"""
World Cup 2026 bracket — 48 teams, 12 groups of 4, new FIFA format.

Knockout stage:
    - Round of 32 (matches 73-88, 16 games): top-2 of each group (24) + 8 best 3rd-placed teams
    - Round of 16 (matches 89-96, 8 games)
    - Quarterfinals (97-100, 4 games)
    - Semifinals (101-102, 2 games)
    - Third-place playoff (103) + Final (104)

The 8 best 3rd-placed teams are slotted into specific bracket positions whose
list of eligible groups depends on which 4 groups have already had their
3rd-placed teams selected (FIFA's 'best-of' rule). For Monte Carlo we
implement the simplified version: select the 8 best thirds by points/GD/GF,
then permute them into the 8 third-slots present in the bracket.

Source: official FIFA-released structure (see __init__ docstring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Name mapping: input groups are in Spanish; our corpus uses English names.
# ---------------------------------------------------------------------------

SPANISH_TO_ENGLISH = {
    "México": "Mexico",
    "Sudáfrica": "South Africa",
    "Corea del Sur": "South Korea",
    "República Checa": "Czech Republic",
    "Canadá": "Canada",
    "Bosnia y Herzegovina": "Bosnia and Herzegovina",
    "Catar": "Qatar",
    "Suiza": "Switzerland",
    "Brasil": "Brazil",
    "Marruecos": "Morocco",
    "Haití": "Haiti",
    "Escocia": "Scotland",
    "Estados Unidos": "United States",
    "Paraguay": "Paraguay",
    "Australia": "Australia",
    "Turquía": "Turkey",
    "Alemania": "Germany",
    "Curazao": "Curaçao",
    "Costa de Marfil": "Ivory Coast",
    "Ecuador": "Ecuador",
    "Países Bajos": "Netherlands",
    "Japón": "Japan",
    "Túnez": "Tunisia",
    "Suecia": "Sweden",
    "Bélgica": "Belgium",
    "Egipto": "Egypt",
    "Irán": "Iran",
    "Nueva Zelanda": "New Zealand",
    "España": "Spain",
    "Cabo Verde": "Cape Verde",
    "Arabia Saudita": "Saudi Arabia",
    "Uruguay": "Uruguay",
    "Francia": "France",
    "Senegal": "Senegal",
    "Irak": "Iraq",
    "Noruega": "Norway",
    "Argentina": "Argentina",
    "Argelia": "Algeria",
    "Austria": "Austria",
    "Jordania": "Jordan",
    "Portugal": "Portugal",
    "Uzbekistán": "Uzbekistan",
    "Colombia": "Colombia",
    "República Democrática del Congo": "DR Congo",
    "Inglaterra": "England",
    "Croacia": "Croatia",
    "Ghana": "Ghana",
    "Panamá": "Panama",
}


def to_english(name: str) -> str:
    """Normalises team names to the English form used in the martj42 corpus."""
    return SPANISH_TO_ENGLISH.get(name, name)


# ---------------------------------------------------------------------------
# 2. Groups (composition)
# ---------------------------------------------------------------------------

WC2026_GROUPS_RAW = {
    "Group_A": ["México", "Sudáfrica", "Corea del Sur", "República Checa"],
    "Group_B": ["Canadá", "Bosnia y Herzegovina", "Catar", "Suiza"],
    "Group_C": ["Brasil", "Marruecos", "Haití", "Escocia"],
    "Group_D": ["Estados Unidos", "Paraguay", "Australia", "Turquía"],
    "Group_E": ["Alemania", "Curazao", "Costa de Marfil", "Ecuador"],
    "Group_F": ["Países Bajos", "Japón", "Túnez", "Suecia"],
    "Group_G": ["Bélgica", "Egipto", "Irán", "Nueva Zelanda"],
    "Group_H": ["España", "Cabo Verde", "Arabia Saudita", "Uruguay"],
    "Group_I": ["Francia", "Senegal", "Irak", "Noruega"],
    "Group_J": ["Argentina", "Argelia", "Austria", "Jordania"],
    "Group_K": ["Portugal", "Uzbekistán", "Colombia", "República Democrática del Congo"],
    "Group_L": ["Inglaterra", "Croacia", "Ghana", "Panamá"],
}

# English version (used internally)
WC2026_GROUPS = {
    grp: [to_english(t) for t in teams]
    for grp, teams in WC2026_GROUPS_RAW.items()
}

GROUP_NAMES = list(WC2026_GROUPS.keys())   # ['Group_A', ..., 'Group_L']


# ---------------------------------------------------------------------------
# 3. Knockout bracket
# ---------------------------------------------------------------------------

@dataclass
class BracketMatch:
    match_id: int
    slot_a: str   # e.g., '1st_Group_A', 'winner_match_73', '3rd_Group_A_B_C_D_F'
    slot_b: str


# Source: official FIFA bracket for WC 2026 (provided by user 2026-05-22)
ROUND_OF_32 = [
    BracketMatch(73, "2nd_Group_A", "2nd_Group_B"),
    BracketMatch(74, "1st_Group_E", "3rd_Group_A_B_C_D_F"),
    BracketMatch(75, "1st_Group_F", "2nd_Group_C"),
    BracketMatch(76, "1st_Group_C", "2nd_Group_F"),
    BracketMatch(77, "1st_Group_I", "3rd_Group_C_D_F_G_H"),
    BracketMatch(78, "2nd_Group_E", "2nd_Group_I"),
    BracketMatch(79, "1st_Group_A", "3rd_Group_C_E_F_H_I"),
    BracketMatch(80, "1st_Group_L", "3rd_Group_E_H_I_J_K"),
    BracketMatch(81, "1st_Group_D", "3rd_Group_B_E_F_I_J"),
    BracketMatch(82, "1st_Group_G", "3rd_Group_A_E_H_I_J"),
    BracketMatch(83, "2nd_Group_K", "2nd_Group_L"),
    BracketMatch(84, "1st_Group_H", "2nd_Group_J"),
    BracketMatch(85, "1st_Group_B", "3rd_Group_E_F_G_I_J"),
    BracketMatch(86, "1st_Group_J", "2nd_Group_H"),
    BracketMatch(87, "1st_Group_K", "3rd_Group_D_E_I_J_L"),
    BracketMatch(88, "2nd_Group_D", "2nd_Group_G"),
]

ROUND_OF_16 = [
    BracketMatch(89, "winner_match_74", "winner_match_77"),
    BracketMatch(90, "winner_match_73", "winner_match_75"),
    BracketMatch(91, "winner_match_76", "winner_match_78"),
    BracketMatch(92, "winner_match_79", "winner_match_80"),
    BracketMatch(93, "winner_match_83", "winner_match_84"),
    BracketMatch(94, "winner_match_81", "winner_match_82"),
    BracketMatch(95, "winner_match_86", "winner_match_88"),
    BracketMatch(96, "winner_match_85", "winner_match_87"),
]

QUARTERFINALS = [
    BracketMatch(97, "winner_match_89", "winner_match_90"),
    BracketMatch(98, "winner_match_91", "winner_match_92"),
    BracketMatch(99, "winner_match_93", "winner_match_94"),
    BracketMatch(100, "winner_match_95", "winner_match_96"),
]

SEMIFINALS = [
    BracketMatch(101, "winner_match_97", "winner_match_98"),
    BracketMatch(102, "winner_match_99", "winner_match_100"),
]

THIRD_PLACE = BracketMatch(103, "loser_match_101", "loser_match_102")
FINAL = BracketMatch(104, "winner_match_101", "winner_match_102")

ALL_KNOCKOUT_MATCHES: list[BracketMatch] = (
    ROUND_OF_32 + ROUND_OF_16 + QUARTERFINALS + SEMIFINALS + [THIRD_PLACE, FINAL]
)

ROUND_NAMES = {
    "round_of_32":  set(m.match_id for m in ROUND_OF_32),
    "round_of_16":  set(m.match_id for m in ROUND_OF_16),
    "quarterfinals":set(m.match_id for m in QUARTERFINALS),
    "semifinals":   set(m.match_id for m in SEMIFINALS),
    "third_place":  {THIRD_PLACE.match_id},
    "final":        {FINAL.match_id},
}


def round_of_match(match_id: int) -> str:
    """Returns the round name for a given match_id (e.g., 'quarterfinals')."""
    for name, ids in ROUND_NAMES.items():
        if match_id in ids:
            return name
    raise ValueError(f"Unknown match_id: {match_id}")


# ---------------------------------------------------------------------------
# 4. Slot resolution helpers
# ---------------------------------------------------------------------------

def parse_third_slot_eligible_groups(slot: str) -> list[str]:
    """Parses a slot like '3rd_Group_A_B_C_D_F' into ['Group_A', ..., 'Group_F']."""
    if not slot.startswith("3rd_Group_"):
        raise ValueError(f"Not a 3rd-place slot: {slot}")
    letters = slot[len("3rd_Group_"):].split("_")
    return [f"Group_{l}" for l in letters]


# ---------------------------------------------------------------------------
# 5. Sanity
# ---------------------------------------------------------------------------

def assert_bracket_consistency() -> None:
    """Verifies internal consistency of the bracket structure."""
    n_teams = sum(len(g) for g in WC2026_GROUPS.values())
    assert n_teams == 48, f"Expected 48 teams, got {n_teams}"
    assert len(WC2026_GROUPS) == 12
    assert all(len(g) == 4 for g in WC2026_GROUPS.values())

    # 32 teams advance to round of 32: 24 (top-2) + 8 (best thirds)
    # Round of 32 has 16 matches → 32 slots.
    assert len(ROUND_OF_32) == 16
    # Round of 16 has 8 matches, etc.
    assert len(ROUND_OF_16) == 8
    assert len(QUARTERFINALS) == 4
    assert len(SEMIFINALS) == 2

    # No team can be in two groups
    seen = set()
    for g, teams in WC2026_GROUPS.items():
        for t in teams:
            assert t not in seen, f"Team {t} appears twice"
            seen.add(t)
    assert len(seen) == 48
