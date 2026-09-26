import copy

import numpy as np
import pandas as pd

from src.config import load_config
from src.models import run_holdout

SEASONS = [2019, 2020, 2021, 2022]  # 2022 plays the role of the "holdout" season in this test
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
            for i in range(FIELD):
                tournament_id = f"R{season}{e:03d}"
                event_rows.append(
                    {
                        "tournament_id": tournament_id,
                        "season": season,
                        "player_id": i,
                        "start_date": start.date().isoformat(),
                        "end_date": end.date().isoformat(),
                        "field_size": FIELD,
                        "amateur": False,
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


def _config():
    config = copy.deepcopy(load_config())
    config["splits"] = {
        "first_train_season": 2019,
        "validation_seasons": [2021],
        "holdout_seasons": [2022],
        "forward_test_seasons": [2099],
    }
    config["model1"]["rating"]["long_half_life_days"] = [180, 365]
    config["model1"]["rating"]["shrinkage_rounds"] = [10, 30]
    return config


def _write(tmp_path, config, events, rounds) -> dict:
    events.to_csv(tmp_path / "player_events.csv", index=False)
    rounds.to_csv(tmp_path / "player_rounds.csv", index=False)
    return {
        **config,
        "data": {
            **config["data"],
            "events_history": str(tmp_path / "player_events.csv"),
            "rounds_history": str(tmp_path / "player_rounds.csv"),
        },
    }


def test_choose_settings_ignores_rows_from_the_holdout_season(tmp_path):
    """Leakage: appending the holdout season to the input files must not change the settings."""
    config = _config()
    events, rounds = _synthetic(SEASONS)
    events_without = events[events["season"] != 2022]
    rounds_without = rounds[rounds["season"] != 2022]

    with_holdout = run_holdout.choose_settings(_write(tmp_path, config, events, rounds))
    without_holdout = run_holdout.choose_settings(
        _write(tmp_path, config, events_without, rounds_without)
    )

    assert with_holdout == without_holdout


def test_choose_settings_returns_a_window_for_every_label(tmp_path):
    config = _config()
    events, rounds = _synthetic(SEASONS)

    half_life, shrink, windows = run_holdout.choose_settings(
        _write(tmp_path, config, events, rounds)
    )

    assert half_life in config["model1"]["rating"]["long_half_life_days"]
    assert shrink in config["model1"]["rating"]["shrinkage_rounds"]
    assert set(windows) == set(run_holdout.LABELS)
    assert all(w in config["baseline1"]["windows"] for w in windows.values())
