"""Shared core for the API and dashboard: reads the exported prediction bundle and serves it by
tournament. Regenerate the bundle with `uv run python -m src.models.export_predictions`; this
module only reads what that script wrote - it never re-fits a model at request time, so a request
is instant and can never accidentally take a new "look" at anything (see docs/DECISIONS.md).
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src.models.export_predictions import EVENTS_PATH, PREDICTIONS_PATH


class BundleNotFoundError(RuntimeError):
    """The exported prediction bundle hasn't been built yet."""


@lru_cache(maxsize=1)
def load_bundle() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not EVENTS_PATH.exists() or not PREDICTIONS_PATH.exists():
        raise BundleNotFoundError(
            f"No exported predictions at {EVENTS_PATH} / {PREDICTIONS_PATH}. Run "
            "`uv run python -m src.models.export_predictions` first."
        )
    return pd.read_csv(EVENTS_PATH), pd.read_csv(PREDICTIONS_PATH)


def list_events() -> pd.DataFrame:
    events, _ = load_bundle()
    return events.sort_values(["season", "tournament_id"]).reset_index(drop=True)


def event_predictions(tournament_id: str) -> tuple[pd.Series, pd.DataFrame]:
    """The event's own row from `events`, and its prediction rows from `predictions`.

    Raises `KeyError` if `tournament_id` isn't in the bundle.
    """
    events, predictions = load_bundle()
    matches = events[events["tournament_id"] == tournament_id]
    if matches.empty:
        raise KeyError(tournament_id)
    return matches.iloc[0], predictions[predictions["tournament_id"] == tournament_id]
