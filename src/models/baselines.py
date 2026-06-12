"""
Tabular baselines for match outcome prediction (W / D / L, multi-class).

Wrappers around sklearn / XGBoost / LightGBM with a common interface so that
training and evaluation code is model-agnostic.

Every baseline exposes:
    - fit(X_train, y_train, X_val=None, y_val=None)
    - predict_proba(X) -> (N, 3) probabilities

Models implemented:
    - LogisticRegressionBaseline   : sklearn multinomial LR, with imputation + scaling
    - XGBoostBaseline              : XGBoost multi:softprob, native NaN handling
    - LightGBMBaseline             : LightGBM multiclass, native NaN handling
    - ELOBaseline                  : NO ML — use ELO expected probability directly.
                                     Crucial sanity-check baseline: any ML
                                     model that doesn't beat ELO alone is wasted effort.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Common interface
# ---------------------------------------------------------------------------

class BaselineModel(ABC):
    """Abstract base for all tabular baselines."""

    name: str = "abstract"

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> "BaselineModel":
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def feature_importance(self) -> pd.Series | None:
        """Override in subclasses that support it."""
        return None


# ---------------------------------------------------------------------------
# ELO baseline (no ML)
# ---------------------------------------------------------------------------

class ELOBaseline(BaselineModel):
    """ELO-only predictor: maps expected_home_win_prob → 3-class distribution.

    Conversion logic (heuristic but standard in football literature):
        - We have P(home wins) directly from ELO formula
        - We split P(not_home_win) = 1 - P(home_wins) between draw and away_win
          using a fixed draw_rate ~ 0.25 (empirical mean) adjusted by closeness
          of the matchup

    More principled approaches (Dixon-Coles, Poisson) exist but this is a
    well-known sanity check.
    """

    name = "elo_only"

    def __init__(self, draw_baseline: float = 0.23):
        """draw_baseline: fraction of remaining mass to assign to draw.
        0.23 is empirically what we see in our int. dataset (Fase 0.5)."""
        self.draw_baseline = draw_baseline

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if "expected_home_win_prob" not in X_train.columns:
            raise ValueError("ELOBaseline requires 'expected_home_win_prob' column")
        # Optionally calibrate the draw rate on training data
        # Simple version: just use the configured value.
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_home = X["expected_home_win_prob"].values.astype(np.float64)
        # Higher draw probability when matchup is closer (|p_home - 0.5| small)
        closeness = 1.0 - 2.0 * np.abs(p_home - 0.5)  # 1 if 50/50, 0 if 0/100
        draw_weight = self.draw_baseline * (0.7 + 0.6 * closeness)  # 0.16..0.30
        # Distribute remaining mass
        remaining = 1.0 - p_home
        p_draw = remaining * draw_weight / (draw_weight + (1.0 - draw_weight))
        p_away = remaining - p_draw
        probs = np.column_stack([p_home, p_draw, p_away])
        # Normalize defensively
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

class LogisticRegressionBaseline(BaselineModel):
    """Multinomial logistic regression with imputation + standard scaling.

    Imputation strategy: median for numeric features. Categorical features
    are expected to already be one-hot encoded.
    """

    name = "logistic_regression"

    def __init__(self, C: float = 1.0, max_iter: int = 2000, random_state: int = 42):
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=C,
                max_iter=max_iter,
                solver="lbfgs",   # multinomial is default behavior in sklearn>=1.5
                random_state=random_state,
            )),
        ])
        self.feature_names_: list[str] = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.feature_names_ = list(X_train.columns)
        self.pipeline.fit(X_train.values, y_train)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.feature_names_].values)

    def feature_importance(self) -> pd.Series:
        coefs = self.pipeline.named_steps["model"].coef_
        importance = np.abs(coefs).mean(axis=0)
        return pd.Series(importance, index=self.feature_names_).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBoostBaseline(BaselineModel):
    """XGBoost with native NaN handling and early stopping if val set given."""

    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        random_state: int = 42,
        early_stopping_rounds: int = 30,
    ):
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("xgboost is required. pip install xgboost") from e
        self._xgb = xgb
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            random_state=random_state,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.model_: Any = None
        self.feature_names_: list[str] = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.feature_names_ = list(X_train.columns)
        kwargs = {}
        if X_val is not None and y_val is not None:
            kwargs["eval_set"] = [(X_val.values, y_val)]
            kwargs["verbose"] = False
            self.model_ = self._xgb.XGBClassifier(
                **self.params,
                early_stopping_rounds=self.early_stopping_rounds,
            )
        else:
            self.model_ = self._xgb.XGBClassifier(**self.params)
        self.model_.fit(X_train.values, y_train, **kwargs)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[self.feature_names_].values)

    def feature_importance(self) -> pd.Series:
        importances = self.model_.feature_importances_
        return pd.Series(importances, index=self.feature_names_).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class LightGBMBaseline(BaselineModel):
    """LightGBM with native NaN + categorical handling, early stopping."""

    name = "lightgbm"

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        early_stopping_rounds: int = 30,
    ):
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError("lightgbm is required. pip install lightgbm") from e
        self._lgb = lgb
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective="multiclass",
            num_class=3,
            metric="multi_logloss",
            verbosity=-1,
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.model_: Any = None
        self.feature_names_: list[str] = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.feature_names_ = list(X_train.columns)
        callbacks = []
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val.values, y_val)]
            callbacks = [self._lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
        self.model_ = self._lgb.LGBMClassifier(**self.params)
        fit_kwargs = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["callbacks"] = callbacks
        self.model_.fit(X_train.values, y_train, **fit_kwargs)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[self.feature_names_].values)

    def feature_importance(self) -> pd.Series:
        importances = self.model_.feature_importances_
        return pd.Series(importances, index=self.feature_names_).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_all_baselines() -> dict[str, BaselineModel]:
    """Return a fresh dict of all baseline models keyed by name."""
    return {
        "elo_only": ELOBaseline(),
        "logistic_regression": LogisticRegressionBaseline(),
        "xgboost": XGBoostBaseline(),
        "lightgbm": LightGBMBaseline(),
    }
