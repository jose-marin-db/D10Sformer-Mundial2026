"""
Feature engineering for the tabular baselines.

Builds a leakage-free feature matrix from the international matches dataset
(with ELO already computed). Each row corresponds to a match and contains only
features that would have been available BEFORE that match was played.

Key principle: iterate chronologically and maintain per-team history. No row's
features depend on its own outcome or any later match.

Output schema (per match row):

    Identifiers
        date, home_team, away_team, tournament, tournament_class, neutral

    Target
        result       int  {0: HOME_WIN, 1: DRAW, 2: AWAY_WIN}
        home_score   int
        away_score   int

    ELO (already in input)
        home_elo, away_elo, elo_diff, expected_home_win_prob

    Rolling form (last N matches before this one, per team)
        home_form{N}_pts, home_form{N}_gf, home_form{N}_ga, home_form{N}_gd
        away_form{N}_pts, away_form{N}_gf, away_form{N}_ga, away_form{N}_gd
        (for N in WINDOWS_FORM)

    Head-to-head (last K direct meetings before this one)
        h2h_n_matches, h2h_home_wins, h2h_draws, h2h_away_wins,
        h2h_avg_gd_for_home

    Rest
        home_rest_days, away_rest_days

    Categorical encodings handled by the model layer; this module only
    produces the raw values plus simple binary flags.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOWS_FORM = (5, 10)            # rolling form windows
H2H_WINDOW = 5                     # head-to-head last K meetings

RESULT_HOME_WIN = 0
RESULT_DRAW = 1
RESULT_AWAY_WIN = 2

LABEL_NAMES = ("HOME_WIN", "DRAW", "AWAY_WIN")


# ---------------------------------------------------------------------------
# Per-team history container
# ---------------------------------------------------------------------------

@dataclass
class TeamMatchRecord:
    """A single past match, stored from the perspective of a given team."""

    date: pd.Timestamp
    opponent: str
    goals_for: int
    goals_against: int
    points: int          # 3 win, 1 draw, 0 loss
    venue: str           # 'home', 'away', or 'neutral'

    @property
    def result_code(self) -> int:
        if self.points == 3:
            return RESULT_HOME_WIN if self.venue != "away" else RESULT_AWAY_WIN
        if self.points == 1:
            return RESULT_DRAW
        return RESULT_AWAY_WIN if self.venue != "away" else RESULT_HOME_WIN


class TeamHistory:
    """Maintains the chronological history of a single team."""

    def __init__(self) -> None:
        self.matches: list[TeamMatchRecord] = []

    def append(self, record: TeamMatchRecord) -> None:
        self.matches.append(record)

    # ---- form features ----

    def rolling_form(self, n: int) -> dict[str, float]:
        """Stats over the last `n` matches (or fewer if not enough history)."""
        recent = self.matches[-n:] if self.matches else []
        if not recent:
            # No prior data: return neutral-ish defaults so model can flag "cold start"
            return {
                f"form{n}_pts": np.nan,
                f"form{n}_gf": np.nan,
                f"form{n}_ga": np.nan,
                f"form{n}_gd": np.nan,
                f"form{n}_n": 0,
            }
        return {
            f"form{n}_pts": float(np.mean([m.points for m in recent])),
            f"form{n}_gf": float(np.mean([m.goals_for for m in recent])),
            f"form{n}_ga": float(np.mean([m.goals_against for m in recent])),
            f"form{n}_gd": float(np.mean([m.goals_for - m.goals_against for m in recent])),
            f"form{n}_n": len(recent),
        }

    def days_since_last_match(self, as_of: pd.Timestamp) -> float:
        if not self.matches:
            return np.nan
        return float((as_of - self.matches[-1].date).days)


# ---------------------------------------------------------------------------
# Head-to-head
# ---------------------------------------------------------------------------

def h2h_summary(
    home_history: TeamHistory,
    away_team: str,
    k: int = H2H_WINDOW,
) -> dict[str, float]:
    """Stats over the last `k` direct meetings between home_team and away_team,
    extracted from home_team's perspective.
    """
    meetings = [m for m in home_history.matches if m.opponent == away_team][-k:]
    if not meetings:
        return {
            "h2h_n_matches": 0,
            "h2h_home_wins": 0,
            "h2h_draws": 0,
            "h2h_away_wins": 0,
            "h2h_avg_gd_for_home": np.nan,
        }
    home_wins = sum(1 for m in meetings if m.points == 3)
    draws = sum(1 for m in meetings if m.points == 1)
    away_wins = len(meetings) - home_wins - draws
    avg_gd = float(np.mean([m.goals_for - m.goals_against for m in meetings]))
    return {
        "h2h_n_matches": len(meetings),
        "h2h_home_wins": home_wins,
        "h2h_draws": draws,
        "h2h_away_wins": away_wins,
        "h2h_avg_gd_for_home": avg_gd,
    }


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df_with_elo: pd.DataFrame,
    windows_form: Iterable[int] = WINDOWS_FORM,
    h2h_window: int = H2H_WINDOW,
    min_date: str | pd.Timestamp | None = "2014-01-01",
) -> pd.DataFrame:
    """Construct a leakage-free feature matrix.

    Parameters
    ----------
    df_with_elo : DataFrame
        Output of compute_elo_history (notebook 00b). Must contain at least:
        date, home_team, away_team, home_score, away_score, tournament,
        tournament_class, neutral, home_elo_before, away_elo_before,
        expected_home_win_prob.
    windows_form : iterable of int
        Rolling windows for form features.
    h2h_window : int
        Number of past direct meetings to consider for head-to-head.
    min_date : str or Timestamp or None
        Drop rows before this date AFTER feature computation. (We still use
        all prior history for computing features.) Set to None to keep all.
    """
    if not pd.api.types.is_datetime64_any_dtype(df_with_elo["date"]):
        df = df_with_elo.copy()
        df["date"] = pd.to_datetime(df["date"])
    else:
        df = df_with_elo.copy()

    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

    histories: dict[str, TeamHistory] = defaultdict(TeamHistory)
    rows: list[dict] = []

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        date = row["date"]
        hs, as_ = row["home_score"], row["away_score"]

        if pd.isna(hs) or pd.isna(as_):
            continue

        hs, as_ = int(hs), int(as_)
        home_h = histories[home]
        away_h = histories[away]

        # ----- Build feature dict (using only PAST data) -----

        feat: dict = {
            # identifiers / metadata
            "date": date,
            "home_team": home,
            "away_team": away,
            "tournament": row["tournament"],
            "tournament_class": row["tournament_class"],
            "neutral": bool(row["neutral"]),

            # ELO
            "home_elo": row["home_elo_before"],
            "away_elo": row["away_elo_before"],
            "elo_diff": row["home_elo_before"] - row["away_elo_before"],
            "expected_home_win_prob": row["expected_home_win_prob"],

            # rest
            "home_rest_days": home_h.days_since_last_match(date),
            "away_rest_days": away_h.days_since_last_match(date),
        }

        # rolling form
        for n in windows_form:
            for key, val in home_h.rolling_form(n).items():
                feat[f"home_{key}"] = val
            for key, val in away_h.rolling_form(n).items():
                feat[f"away_{key}"] = val

        # head-to-head (from home perspective)
        feat.update(h2h_summary(home_h, away, k=h2h_window))

        # ----- target -----
        if hs > as_:
            feat["result"] = RESULT_HOME_WIN
        elif hs == as_:
            feat["result"] = RESULT_DRAW
        else:
            feat["result"] = RESULT_AWAY_WIN
        feat["home_score"] = hs
        feat["away_score"] = as_

        rows.append(feat)

        # ----- update histories AFTER recording features (no leakage) -----
        neutral = bool(row["neutral"])
        home_venue = "neutral" if neutral else "home"
        away_venue = "neutral" if neutral else "away"

        home_points = 3 if hs > as_ else (1 if hs == as_ else 0)
        away_points = 3 if as_ > hs else (1 if hs == as_ else 0)

        home_h.append(TeamMatchRecord(
            date=date, opponent=away, goals_for=hs, goals_against=as_,
            points=home_points, venue=home_venue,
        ))
        away_h.append(TeamMatchRecord(
            date=date, opponent=home, goals_for=as_, goals_against=hs,
            points=away_points, venue=away_venue,
        ))

    feature_df = pd.DataFrame(rows)

    if min_date is not None:
        feature_df = feature_df[feature_df["date"] >= pd.Timestamp(min_date)].reset_index(drop=True)

    return feature_df


# ---------------------------------------------------------------------------
# Helpers for downstream model training
# ---------------------------------------------------------------------------

# Columns that are identifiers / target — must NOT be passed to the model.
NON_FEATURE_COLS = {
    "date", "home_team", "away_team", "tournament",
    "result", "home_score", "away_score",
}

# Categorical features that need one-hot or ordinal encoding.
CATEGORICAL_COLS = {"tournament_class", "neutral"}


def get_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    """All columns that should be passed to a model (excludes ids + target)."""
    return [c for c in feature_df.columns if c not in NON_FEATURE_COLS]


def temporal_split(
    feature_df: pd.DataFrame,
    train_cutoff: str = "2023-01-01",
    val_cutoff: str = "2024-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically: train < train_cutoff <= val < val_cutoff <= test."""
    train_cutoff = pd.Timestamp(train_cutoff)
    val_cutoff = pd.Timestamp(val_cutoff)
    train = feature_df[feature_df["date"] < train_cutoff].reset_index(drop=True)
    val = feature_df[(feature_df["date"] >= train_cutoff) & (feature_df["date"] < val_cutoff)].reset_index(drop=True)
    test = feature_df[feature_df["date"] >= val_cutoff].reset_index(drop=True)
    return train, val, test


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    onehot_categoricals: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns (X, y) ready for sklearn / xgboost / lightgbm.

    Notes:
    - One-hot encoding aligned across splits is the caller's responsibility
      (pass the same `feature_cols` list to train/val/test).
    - NaN values are NOT imputed here; XGBoost / LightGBM handle them natively.
      For sklearn LogReg, use SimpleImputer in a Pipeline.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    if onehot_categoricals:
        for col in [c for c in CATEGORICAL_COLS if c in X.columns]:
            X = pd.get_dummies(X, columns=[col], prefix=col, dtype=float)
    y = df["result"].values
    return X, y
