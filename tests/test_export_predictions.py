import copy

import numpy as np
import pandas as pd

from src.config import load_config
from src.models import export_predictions

SEASONS = [2019, 2020, 2021, 2022, 2023]  # 2023 plays the "forward test" role in this test
FIELD = 12
SG_COLUMNS = ["sg_total", "sg_off_the_tee", "sg_approach", "sg_around_the_green", "sg_putting"]


def _synthetic(seasons, seed=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small multi-season events and rounds tables shaped like the real tidy tables."""
    rng = np.random.default_rng(seed)
    event_rows, round_rows = [], []
    for season in seasons:
        for e in range(4):
            skill = rng.normal(0, 1, FIELD)
            start = pd.Timestamp(year=season, month=1 + e * 2, day=5)
            end = start + pd.Timedelta(days=3)
            scores = 70 - skill + rng.normal(0, 1, (4, FIELD))
            rank = (-skill).argsort().argsort() + 1
            tournament_id = f"R{season}{e:03d}"
            for i in range(FIELD):
                event_rows.append(
                    {
                        "tournament_id": tournament_id,
                        "tournament_name": f"Synthetic Open {season}-{e}",
                        "season": season,
                        "player_id": i,
                        "display_name": f"Player {i}",
                        "start_date": start.date().isoformat(),
                        "end_date": end.date().isoformat(),
                        "field_size": FIELD,
                        "amateur": False,
                        "is_standard_event": True,
                        "has_cut": True,
                        "made_cut": bool(rank[i] <= FIELD // 2),
                        "top10": bool(rank[i] <= 4),
                        "win": bool(rank[i] == 1),
                        **{
                            f"{c}_rolling_{w}": rng.normal(0, 0.3)
                            for c in SG_COLUMNS
                            for w in [3, 5, 10]
                        },
                    }
                )
                for number in range(1, 5):
                    round_rows.append(
                        {
                            "tournament_id": tournament_id,
                            "season": season,
                            "player_id": i,
                            "round": number,
                            "score": float(scores[number - 1, i]),
                            "round_date": (start + pd.Timedelta(days=number - 1))
                            .date()
                            .isoformat(),
                            "is_standard_event": True,
                        }
                    )
    events = pd.DataFrame(event_rows).astype(
        {"made_cut": "boolean", "top10": "boolean", "win": "boolean"}
    )
    return events, pd.DataFrame(round_rows)


def _config(tmp_path):
    config = copy.deepcopy(load_config())
    config["splits"] = {
        "first_train_season": 2019,
        "validation_seasons": [2021],
        "holdout_seasons": [2022],
        "forward_test_seasons": [2023],
    }
    config["model1"]["rating"]["long_half_life_days"] = [180, 365]
    config["model1"]["rating"]["shrinkage_rounds"] = [10, 30]
    config["sim"] = {**config["sim"], "n_sims": 200}

    events, rounds = _synthetic(SEASONS)
    history = events[events["season"] != 2023]
    forward = events[events["season"] == 2023]
    rounds_history = rounds[rounds["season"] != 2023]
    rounds_forward = rounds[rounds["season"] == 2023]
    history.to_csv(tmp_path / "history_events.csv", index=False)
    forward.to_csv(tmp_path / "forward_events.csv", index=False)
    rounds_history.to_csv(tmp_path / "history_rounds.csv", index=False)
    rounds_forward.to_csv(tmp_path / "forward_rounds.csv", index=False)
    config["data"] = {
        **config["data"],
        "events_history": str(tmp_path / "history_events.csv"),
        "rounds_history": str(tmp_path / "history_rounds.csv"),
        "events_2026": str(tmp_path / "forward_events.csv"),
        "rounds_2026": str(tmp_path / "forward_rounds.csv"),
    }
    return config


def test_main_writes_one_events_row_per_tournament_and_covers_every_era(tmp_path):
    config = _config(tmp_path)
    export_predictions.EVENTS_PATH = tmp_path / "served_events.csv"
    export_predictions.PREDICTIONS_PATH = tmp_path / "served_predictions.csv"

    original_load_config = export_predictions.load_config
    export_predictions.load_config = lambda *a, **k: config
    try:
        export_predictions.main()
    finally:
        export_predictions.load_config = original_load_config

    events = pd.read_csv(export_predictions.EVENTS_PATH)
    predictions = pd.read_csv(export_predictions.PREDICTIONS_PATH)

    assert set(events["era"]) == {"validation", "holdout", "forward_test"}
    assert events["tournament_id"].is_unique
    assert set(predictions["era"]) == {"validation", "holdout", "forward_test"}
    assert set(predictions["model"]) == {"Baseline 1", "Model 1", "Simulator", "Naive"}
    assert set(predictions["label"]) == {"made_cut", "top10", "win"}
    assert predictions[["p", "y"]].notna().all().all()
    assert predictions["p"].between(0, 1).all()
    # every (tournament, player, label, model) combination appears at most once
    key = ["tournament_id", "player_id", "label", "model"]
    assert not predictions.duplicated(key).any()
    # every tournament in the predictions table is also in the events table
    assert set(predictions["tournament_id"]) <= set(events["tournament_id"])
