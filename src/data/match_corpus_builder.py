"""
Builders that convert raw DataFrames into lists of `MatchDocument`s.

We have two heterogeneous data sources:

1. **martj42 / international_results** — DataFrame with columns:
       date, home_team, away_team, home_score, away_score, tournament,
       neutral, home_elo (computed in Fase 0b), away_elo
   NO lineups, NO events. Used for the "sparse" half of the corpus.

2. **StatsBomb open-data** — JSON files with matches + lineups + events.
   Used for the "rich" half. Provides lineups and (some) events.

The builders return Python lists of `MatchDocument`. The MatchDataset
consumes those.

Note: this module does NOT load files from disk — the caller is expected to
pass already-loaded DataFrames. This keeps the module side-effect-free and
testable.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .tokenizer import MatchDocument, RollingFeatures, PlayerRef, MatchEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result_from_scores(home: int, away: int) -> str:
    if home > away:
        return "home_win"
    if home < away:
        return "away_win"
    return "draw"


def _venue_for_team_a(row, team_a_col: str = "home_team") -> str:
    """Returns 'home', 'away', or 'neutral' for team_a's perspective."""
    neutral = bool(row.get("neutral", False))
    if neutral:
        return "neutral"
    # In martj42, the home team is the venue host. Since we set team_a = home_team,
    # team_a is at home.
    return "home"


# ---------------------------------------------------------------------------
# martj42 (international) → MatchDocument
# ---------------------------------------------------------------------------

def _first_present_column(row: pd.Series, candidates: list[str]) -> float | None:
    """Return the first non-NaN value among `candidates` columns, or None."""
    for col in candidates:
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                return float(val)
    return None


def build_international_documents(
    df: pd.DataFrame,
    include_features: bool = True,
    min_date: str | None = None,
    max_date: str | None = None,
) -> list[MatchDocument]:
    """Convert the martj42 DataFrame to MatchDocuments.

    Required columns: date, home_team, away_team, home_score, away_score,
    tournament, neutral.

    Optional feature columns (used if `include_features=True`). The builder
    is robust to multiple naming conventions:
      - ELO:   `home_elo_before` (parquet de Fase 0b) o `home_elo` (legacy)
      - Form:  `home_form_pts_5` o `home_form_pts`
      - Goals: `home_recent_goals_5` o `home_recent_goals`
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if min_date:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date:
        df = df[df["date"] <= pd.Timestamp(max_date)]

    # Column resolution — tolera ambas convenciones de nombres
    elo_home_cols = ["home_elo_before", "home_elo"]
    elo_away_cols = ["away_elo_before", "away_elo"]
    form_home_cols = ["home_form_pts_5", "home_form_pts"]
    form_away_cols = ["away_form_pts_5", "away_form_pts"]
    goals_home_cols = ["home_recent_goals_5", "home_recent_goals"]
    goals_away_cols = ["away_recent_goals_5", "away_recent_goals"]

    documents: list[MatchDocument] = []
    for _, row in df.iterrows():
        feats = None
        if include_features:
            home_elo = _first_present_column(row, elo_home_cols)
            away_elo = _first_present_column(row, elo_away_cols)
            # At least one ELO needed to emit a features section
            if home_elo is not None or away_elo is not None:
                feats = RollingFeatures(
                    home_elo=home_elo,
                    away_elo=away_elo,
                    home_form_pts=_first_present_column(row, form_home_cols),
                    away_form_pts=_first_present_column(row, form_away_cols),
                    home_recent_goals=_first_present_column(row, goals_home_cols),
                    away_recent_goals=_first_present_column(row, goals_away_cols),
                )

        doc = MatchDocument(
            tournament=str(row.tournament),
            stage=None,
            team_a=str(row.home_team),
            team_b=str(row.away_team),
            venue=_venue_for_team_a(row),
            features=feats,
            lineup_a=None,   # martj42 has no lineups
            bench_a=None,
            lineup_b=None,
            bench_b=None,
            events=None,
            result=_result_from_scores(int(row.home_score), int(row.away_score)),
            home_score=int(row.home_score),
            away_score=int(row.away_score),
        )
        documents.append(doc)
    return documents


# ---------------------------------------------------------------------------
# StatsBomb → MatchDocument
# ---------------------------------------------------------------------------

def _map_statsbomb_position(sb_pos: str | None) -> str:
    p = (sb_pos or "").lower()
    if "goalkeeper" in p:
        return "GK"
    if "back" in p or "centre-back" in p or "defender" in p:
        return "DF"
    if "midfield" in p:
        return "MF"
    if "forward" in p or "wing" in p or "striker" in p:
        return "FW"
    return "UNKNOWN"


def build_statsbomb_document(
    match_meta: dict,
    lineup_home: list[dict] | None,
    lineup_away: list[dict] | None,
    events: list[dict] | None = None,
) -> MatchDocument:
    """Build a single MatchDocument from raw StatsBomb JSON dicts.

    Args:
        match_meta: a single match dict from a StatsBomb matches/<comp>/<season>.json.
        lineup_home: parsed `lineups/<match_id>.json` → entry for home team.
        lineup_away: same for away team.
        events: optional list of parsed event dicts (`events/<match_id>.json`).

    The schema follows StatsBomb's open-data documentation.
    """
    def _to_player_refs(lineup: list[dict] | None) -> list[PlayerRef] | None:
        if not lineup:
            return None
        refs = []
        for p in lineup:
            pid = p.get("player_id")
            if pid is None:
                continue
            # StatsBomb stores position under "positions" list or "position"
            pos = "UNKNOWN"
            if "positions" in p and p["positions"]:
                pos_name = p["positions"][0].get("position", "")
                pos = _map_statsbomb_position(pos_name)
            elif "position" in p:
                pos = _map_statsbomb_position(p["position"].get("name", ""))
            refs.append(PlayerRef(player_id=str(pid), position=pos))
        return refs

    home_team = match_meta["home_team"]["home_team_name"]
    away_team = match_meta["away_team"]["away_team_name"]
    competition = match_meta["competition"]["competition_name"]
    season = match_meta.get("season", {}).get("season_name", "")
    home_score = int(match_meta.get("home_score", 0))
    away_score = int(match_meta.get("away_score", 0))

    lineup_a = _to_player_refs(lineup_home)
    lineup_b = _to_player_refs(lineup_away)

    # Stage heuristic: StatsBomb's "stage" or "competition_stage"
    stage_name = None
    if "competition_stage" in match_meta and match_meta["competition_stage"]:
        s = match_meta["competition_stage"].get("name", "").lower()
        if "group" in s:
            stage_name = "group"
        elif "final" in s and "semi" not in s and "quarter" not in s:
            stage_name = "final"
        elif "semi" in s:
            stage_name = "sf"
        elif "quarter" in s:
            stage_name = "qf"
        elif "round of 16" in s or "r16" in s:
            stage_name = "r16"
        elif "playoff" in s:
            stage_name = "playoff"
        else:
            stage_name = "regular_season"

    # Events parsing — keep it lean: goals, cards, subs only
    parsed_events: list[MatchEvent] | None = None
    if events:
        parsed_events = []
        for ev in events:
            ev_type = ev.get("type", {}).get("name", "").lower()
            minute = int(ev.get("minute", 0))
            team_name = ev.get("team", {}).get("name")
            team = "a" if team_name == home_team else "b"
            player_id = ev.get("player", {}).get("id")
            player_pos = "UNKNOWN"
            if "position" in ev:
                player_pos = _map_statsbomb_position(ev["position"].get("name", ""))

            mapped_type = None
            if ev_type == "shot" and ev.get("shot", {}).get("outcome", {}).get("name") == "Goal":
                mapped_type = "goal"
            elif "card" in ev_type:
                # foul committed with card
                card = ev.get("foul_committed", {}).get("card", {}).get("name", "").lower()
                if "yellow" in card:
                    mapped_type = "yellow_card"
                elif "red" in card:
                    mapped_type = "red_card"
            elif ev_type == "substitution":
                mapped_type = "substitution"

            if mapped_type is None:
                continue

            parsed_events.append(MatchEvent(
                minute=minute,
                team=team,
                event_type=mapped_type,
                player_id=str(player_id) if player_id else None,
                player_position=player_pos,
            ))

    return MatchDocument(
        tournament=competition,
        stage=stage_name,
        team_a=home_team,
        team_b=away_team,
        venue="home",   # StatsBomb matches assume home_team is home host
        features=None,  # StatsBomb has no rolling-ELO; merged separately later
        lineup_a=lineup_a,
        bench_a=None,
        lineup_b=lineup_b,
        bench_b=None,
        events=parsed_events,
        result=_result_from_scores(home_score, away_score),
        home_score=home_score,
        away_score=away_score,
    )


def build_statsbomb_documents(
    matches_iter: Iterable[tuple[dict, list[dict] | None, list[dict] | None, list[dict] | None]],
) -> list[MatchDocument]:
    """Convenience batch wrapper around `build_statsbomb_document`.

    `matches_iter` yields tuples of (match_meta, lineup_home, lineup_away, events).
    """
    return [build_statsbomb_document(m, lh, la, ev) for (m, lh, la, ev) in matches_iter]
