"""Baseline 1: turn a player's rolling strokes gained into probabilities.

Ranking by rolling strokes gained is not a probability, so each label gets a small calibrating
step with two parameters: a slope on the (centred) rolling strokes gained, and an effect for
players with no history, whose feature is missing.

  made_cut, top10  logistic regression on [strokes gained, missing flag]
  win              softmax across each event's field (probabilities sum to 1 per event), fitted by
                   maximum likelihood on the events' winners

The feature must come from earlier seasons only (the tidy tables guarantee this).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

WIN_LABEL = "win"


def design_matrix(frame: pd.DataFrame, column: str, center: float) -> np.ndarray:
    """[centred feature (0 where missing), missing flag]."""
    missing = frame[column].isna().to_numpy()
    feature = np.where(missing, 0.0, frame[column].to_numpy(dtype=float) - center)
    return np.column_stack([feature, missing.astype(float)])


def event_layout(events: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row order that groups events together, the start of each event, and its size."""
    order = np.argsort(events, kind="stable")
    sorted_events = events[order]
    starts = np.flatnonzero(np.r_[True, sorted_events[1:] != sorted_events[:-1]])
    sizes = np.diff(np.r_[starts, len(sorted_events)])
    return order, starts, sizes


def softmax_within_events(scores: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Probabilities that sum to 1 within each event."""
    order, starts, sizes = event_layout(events)
    sorted_scores = scores[order]
    log_total = np.repeat(np.logaddexp.reduceat(sorted_scores, starts), sizes)
    probabilities = np.empty(len(scores))
    probabilities[order] = np.exp(sorted_scores - log_total)
    return probabilities


def fit_softmax(
    design: np.ndarray, events: np.ndarray, winner: np.ndarray, l2: float = 0.0
) -> np.ndarray:
    """Weights for 'the winner is drawn from the field with these scores' (maximum likelihood).

    `l2` adds a ridge penalty on the weights; use it with standardised features.
    """
    order, starts, sizes = event_layout(events)
    design, winner = design[order], winner[order].astype(bool)

    def negative_log_likelihood(weights: np.ndarray) -> float:
        scores = design @ weights
        likelihood = np.logaddexp.reduceat(scores, starts).sum() - scores[winner].sum()
        return float(likelihood + l2 * weights @ weights)

    return minimize(negative_log_likelihood, np.zeros(design.shape[1]), method="BFGS").x


@dataclass
class Baseline:
    label: str
    column: str
    center: float
    logistic: LogisticRegression | None = None
    weights: np.ndarray | None = None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        design = design_matrix(frame, self.column, self.center)
        if self.label == WIN_LABEL:
            return softmax_within_events(design @ self.weights, frame["tournament_id"].to_numpy())
        return self.logistic.predict_proba(design)[:, 1]


def fit_baseline(train: pd.DataFrame, label: str, column: str) -> Baseline:
    """Fit on rows whose label is defined. For `win`, whole events must be present."""
    center = float(train[column].mean())
    design = design_matrix(train, column, center)
    outcome = train[label].astype(int).to_numpy()
    if label == WIN_LABEL:
        weights = fit_softmax(design, train["tournament_id"].to_numpy(), outcome)
        return Baseline(label, column, center, weights=weights)
    logistic = LogisticRegression(C=np.inf, max_iter=1000).fit(design, outcome)
    return Baseline(label, column, center, logistic=logistic)
