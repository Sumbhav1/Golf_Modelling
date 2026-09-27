"""FastAPI service over precomputed predictions (see `src.models.export_predictions`).

Run with `uv run uvicorn src.api.main:app --reload`. Serves Baseline 1, Model 1, Simulator and
naive predictions for validation, holdout and (ongoing) forward-test events - never re-fits a
model at request time; see `src.api.serving`. Research only; this API does not place bets.
"""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.api import serving

app = FastAPI(
    title="Golf Pricing Engine API",
    description=(
        "Precomputed make-cut / top-10 / win probabilities from Baseline 1, Model 1 and the "
        "Monte Carlo simulator. Research only; does not place bets."
    ),
)


class EventSummary(BaseModel):
    tournament_id: str
    tournament_name: str
    season: int
    era: str
    start_date: str
    end_date: str
    field_size: int


class PlayerPrediction(BaseModel):
    player_id: int
    display_name: str
    made_cut: dict[str, float]
    top10: dict[str, float]
    win: dict[str, float]
    made_cut_actual: bool | None
    top10_actual: bool | None
    win_actual: bool | None


class EventPredictionsResponse(BaseModel):
    event: EventSummary
    players: list[PlayerPrediction]


def _event_summary(row: pd.Series) -> EventSummary:
    return EventSummary(
        tournament_id=row["tournament_id"],
        tournament_name=row["tournament_name"],
        season=int(row["season"]),
        era=row["era"],
        start_date=row["start_date"],
        end_date=row["end_date"],
        field_size=int(row["field_size"]),
    )


def _win_probability(player: PlayerPrediction) -> float:
    """Best available win probability, for sorting the field - Model 1 if present, else
    Baseline 1, else 0.0. Not `dict.get(...) or dict.get(...)`, since a real probability of
    exactly 0.0 would then wrongly fall through to the next model."""
    for model in ("Model 1", "Baseline 1"):
        if model in player.win:
            return player.win[model]
    return 0.0


def _label_probabilities(group: pd.DataFrame, label: str) -> dict[str, float]:
    sub = group[group["label"] == label]
    return dict(zip(sub["model"], sub["p"], strict=True))


def _label_actual(group: pd.DataFrame, label: str) -> bool | None:
    sub = group[group["label"] == label]
    return None if sub.empty else bool(sub["y"].iloc[0])


def _player_predictions(frame: pd.DataFrame) -> list[PlayerPrediction]:
    players = []
    for (player_id, display_name), group in frame.groupby(["player_id", "display_name"]):
        players.append(
            PlayerPrediction(
                player_id=int(player_id),
                display_name=display_name,
                made_cut=_label_probabilities(group, "made_cut"),
                top10=_label_probabilities(group, "top10"),
                win=_label_probabilities(group, "win"),
                made_cut_actual=_label_actual(group, "made_cut"),
                top10_actual=_label_actual(group, "top10"),
                win_actual=_label_actual(group, "win"),
            )
        )
    return sorted(players, key=_win_probability, reverse=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/events", response_model=list[EventSummary])
def list_events() -> list[EventSummary]:
    try:
        table = serving.list_events()
    except serving.BundleNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [_event_summary(row) for _, row in table.iterrows()]


@app.get("/events/{tournament_id}/predictions", response_model=EventPredictionsResponse)
def get_event_predictions(tournament_id: str) -> EventPredictionsResponse:
    try:
        event_row, frame = serving.event_predictions(tournament_id)
    except serving.BundleNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Unknown tournament_id {tournament_id!r}"
        ) from exc
    return EventPredictionsResponse(
        event=_event_summary(event_row), players=_player_predictions(frame)
    )
