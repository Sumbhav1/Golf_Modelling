import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import main, serving

EVENTS = pd.DataFrame(
    [
        {
            "season": 2025,
            "tournament_id": "R2025001",
            "tournament_name": "Synthetic Open",
            "start_date": "2025-01-02",
            "end_date": "2025-01-05",
            "field_size": 2,
            "era": "holdout",
        }
    ]
)

PREDICTIONS = pd.DataFrame(
    [
        {
            "season": 2025,
            "tournament_id": "R2025001",
            "player_id": 1,
            "display_name": "Player One",
            "label": "made_cut",
            "model": "Baseline 1",
            "p": 0.7,
            "y": 1,
            "era": "holdout",
        },
        {
            "season": 2025,
            "tournament_id": "R2025001",
            "player_id": 1,
            "display_name": "Player One",
            "label": "made_cut",
            "model": "Model 1",
            "p": 0.8,
            "y": 1,
            "era": "holdout",
        },
        {
            "season": 2025,
            "tournament_id": "R2025001",
            "player_id": 1,
            "display_name": "Player One",
            "label": "win",
            "model": "Model 1",
            "p": 0.0,  # exactly zero, to check the sort fallback doesn't misfire on it
            "y": 0,
            "era": "holdout",
        },
        {
            "season": 2025,
            "tournament_id": "R2025001",
            "player_id": 2,
            "display_name": "Player Two",
            "label": "made_cut",
            "model": "Baseline 1",
            "p": 0.3,
            "y": 0,
            "era": "holdout",
        },
        {
            "season": 2025,
            "tournament_id": "R2025001",
            "player_id": 2,
            "display_name": "Player Two",
            "label": "win",
            "model": "Model 1",
            "p": 0.4,
            "y": 1,
            "era": "holdout",
        },
    ]
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    events_path = tmp_path / "served_events.csv"
    predictions_path = tmp_path / "served_predictions.csv"
    EVENTS.to_csv(events_path, index=False)
    PREDICTIONS.to_csv(predictions_path, index=False)
    monkeypatch.setattr(serving, "EVENTS_PATH", events_path)
    monkeypatch.setattr(serving, "PREDICTIONS_PATH", predictions_path)
    serving.load_bundle.cache_clear()
    yield TestClient(main.app)
    serving.load_bundle.cache_clear()


def test_health_reports_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_events_lists_the_bundled_event(client):
    response = client.get("/events")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["tournament_id"] == "R2025001"
    assert body[0]["era"] == "holdout"


def test_event_predictions_returns_both_players_with_their_models_and_outcomes(client):
    response = client.get("/events/R2025001/predictions")

    assert response.status_code == 200
    body = response.json()
    assert body["event"]["tournament_id"] == "R2025001"
    players = {p["display_name"]: p for p in body["players"]}
    assert players["Player One"]["made_cut"] == {"Baseline 1": 0.7, "Model 1": 0.8}
    assert players["Player One"]["made_cut_actual"] is True
    assert players["Player One"]["win_actual"] is False
    assert players["Player Two"]["top10_actual"] is None  # no top10 row for this player


def test_event_predictions_sorts_by_win_probability_without_a_zero_falling_through(client):
    """Player One's Model 1 win probability is exactly 0.0 - the sort must use that value, not
    fall back to a missing Baseline 1 entry and treat it as ranked above Player Two."""
    response = client.get("/events/R2025001/predictions")

    names = [p["display_name"] for p in response.json()["players"]]
    assert names == ["Player Two", "Player One"]  # 0.4 win probability ranks above 0.0


def test_unknown_tournament_returns_404(client):
    response = client.get("/events/does-not-exist/predictions")

    assert response.status_code == 404


def test_missing_bundle_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(serving, "EVENTS_PATH", tmp_path / "missing_events.csv")
    monkeypatch.setattr(serving, "PREDICTIONS_PATH", tmp_path / "missing_predictions.csv")
    serving.load_bundle.cache_clear()

    response = TestClient(main.app).get("/events")

    assert response.status_code == 503
    serving.load_bundle.cache_clear()
