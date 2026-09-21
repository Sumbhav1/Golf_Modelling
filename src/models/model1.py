"""Model 1: field-adjusted ratings and field-relative features to probabilities.

  made_cut, top10  logistic regression or gradient boosting on standardised features. Top-10
                   probabilities are shifted within each event so they add up to about the number
                   of top-10 places (the field's strength does not change how many there are).
  win              softmax over the field on the player-level features (ridge-penalised)

Features come from `src.features.ratings.build_features`, so every one is known before the event.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.baseline import event_layout, fit_softmax, softmax_within_events

SG_CATEGORIES = ["sg_off_the_tee", "sg_approach", "sg_around_the_green", "sg_putting"]
SG_WINDOW = 3

# Feature groups, used to build the full model and for the ablation.
RATING = ["rating_rel"]
FIELD = ["field_mean_rating", "field_size_log", "rank_pct"]
FORM = ["form", "log_rounds", "no_history", "days_since_last", "amateur"]
SG = [*SG_CATEGORIES, "no_sg"]
FULL_COLUMNS = RATING + FIELD + FORM + SG
# Within one event the field-level features are the same for everyone, so they drop out of a
# softmax; `rating_sq` lets very strong players stand out more than a straight line would.
WIN_COLUMNS = ["rating_rel", "rating_sq", "rank_pct", *FORM, *SG]


def design_frame(features: pd.DataFrame) -> pd.DataFrame:
    """The model inputs, with the season strokes-gained categories filled and flagged if missing."""
    frame = features.copy()
    frame["no_sg"] = frame[f"sg_off_the_tee_rolling_{SG_WINDOW}"].isna().astype(float)
    for category in SG_CATEGORIES:
        frame[category] = frame[f"{category}_rolling_{SG_WINDOW}"].fillna(0.0)
    frame["rating_sq"] = frame["rating_rel"] ** 2
    return frame


def normalise_to_target(
    probabilities: np.ndarray, events: np.ndarray, target: float, iterations: int = 30
) -> np.ndarray:
    """Shift each event's logits by one constant so its probabilities sum to `target`.

    The order of players within an event does not change. The target is capped below the size of
    the field.
    """
    order, starts, sizes = event_layout(events)
    scores = logit(np.clip(probabilities[order], 1e-9, 1 - 1e-9))
    wanted = np.minimum(target, sizes - 0.5)
    shift = np.zeros(len(starts))
    for _ in range(iterations):
        shifted = expit(scores + np.repeat(shift, sizes))
        excess = np.add.reduceat(shifted, starts) - wanted
        slope = np.add.reduceat(shifted * (1 - shifted), starts)
        shift -= np.clip(excess / np.maximum(slope, 1e-12), -2, 2)
    out = np.empty(len(probabilities))
    out[order] = expit(scores + np.repeat(shift, sizes))
    return out


@dataclass
class Model1:
    label: str
    kind: str  # "logistic", "boosting" or "softmax"
    columns: list[str]
    scaler: StandardScaler
    estimator: object = None
    weights: np.ndarray | None = None
    top10_target: float | None = None
    settings: dict = field(default_factory=dict)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        inputs = self.scaler.transform(design_frame(features)[self.columns])
        if self.kind == "softmax":
            return softmax_within_events(
                inputs @ self.weights, features["tournament_id"].to_numpy()
            )
        probabilities = self.estimator.predict_proba(inputs)[:, 1]
        if self.top10_target is not None:
            probabilities = normalise_to_target(
                probabilities, features["tournament_id"].to_numpy(), self.top10_target
            )
        return probabilities

    def coefficients(self) -> pd.Series:
        """Weights on standardised inputs (larger magnitude means more influence)."""
        values = self.weights if self.kind == "softmax" else self.estimator.coef_.ravel()
        return pd.Series(values, index=self.columns)


def fit_model1(
    train: pd.DataFrame, label: str, kind: str, columns: list[str], settings: dict, seed: int
) -> Model1:
    """Fit on rows whose label is defined. For `win`, whole events must be present."""
    frame = design_frame(train)
    scaler = StandardScaler().fit(frame[columns])
    inputs = scaler.transform(frame[columns])
    outcome = train[label].astype(int).to_numpy()
    top10_target = None
    if label == "top10" and settings["top10_normalise"]:
        top10_target = float(train.groupby("tournament_id")[label].sum().mean())

    if kind == "softmax":
        weights = fit_softmax(
            inputs, train["tournament_id"].to_numpy(), outcome, l2=settings["win_l2"]
        )
        return Model1(label, kind, columns, scaler, weights=weights, settings=settings)
    if kind == "logistic":
        estimator = LogisticRegression(C=settings["logistic_c"], max_iter=2000)
    else:
        estimator = HistGradientBoostingClassifier(**settings["boosting"], random_state=seed)
    estimator.fit(inputs, outcome)
    return Model1(
        label, kind, columns, scaler, estimator=estimator, top10_target=top10_target,
        settings=settings,
    )  # fmt: skip


def walk_forward_predictions(
    features: pd.DataFrame,
    label: str,
    kind: str,
    columns: list[str],
    folds: list[tuple[list[int], int]],
    settings: dict,
    seed: int,
) -> pd.DataFrame:
    """Predictions for each validation season from a model fitted on earlier seasons only."""
    parts = []
    for train_seasons, test_season in folds:
        train = features[features["season"].isin(train_seasons) & features[label].notna()]
        test = features[(features["season"] == test_season) & features[label].notna()]
        model = fit_model1(train, label, kind, columns, settings, seed)
        part = test[["season", "tournament_id", "player_id"]].copy()
        part["event"] = test["tournament_id"].to_numpy()
        part["y"] = test[label].astype(int).to_numpy()
        part["p"] = model.predict(test)
        part["p_naive"] = train[label].astype(float).mean()
        part["has_history"] = (test["no_history"] == 0).to_numpy()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
