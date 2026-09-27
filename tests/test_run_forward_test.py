import copy

import numpy as np
import pandas as pd

from src.config import load_config
from src.models import run_forward_test

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


def _config():
    config = copy.deepcopy(load_config())
    config["splits"] = {
        "first_train_season": 2019,
        "validation_seasons": [2021],
        "holdout_seasons": [2022],
        "forward_test_seasons": [2023],
    }
    config["model1"]["rating"]["long_half_life_days"] = [180, 365]
    config["model1"]["rating"]["shrinkage_rounds"] = [10, 30]
    return config


def _write(tmp_path, config, events, rounds) -> dict:
    history = events[events["season"] != 2023]
    forward = events[events["season"] == 2023]
    rounds_history = rounds[rounds["season"] != 2023]
    rounds_forward = rounds[rounds["season"] == 2023]
    history.to_csv(tmp_path / "history_events.csv", index=False)
    forward.to_csv(tmp_path / "forward_events.csv", index=False)
    rounds_history.to_csv(tmp_path / "history_rounds.csv", index=False)
    rounds_forward.to_csv(tmp_path / "forward_rounds.csv", index=False)
    return {
        **config,
        "data": {
            **config["data"],
            "events_history": str(tmp_path / "history_events.csv"),
            "rounds_history": str(tmp_path / "history_rounds.csv"),
            "events_2026": str(tmp_path / "forward_events.csv"),
            "rounds_2026": str(tmp_path / "forward_rounds.csv"),
        },
    }


def test_load_forward_inputs_trains_on_every_earlier_season_including_the_holdout(tmp_path):
    config = _config()
    events, rounds = _synthetic(SEASONS)
    config = _write(tmp_path, config, events, rounds)

    combined_events, combined_rounds, fold = run_forward_test.load_forward_inputs(config)
    (train_seasons, test_season) = fold[0]

    assert test_season == 2023
    assert train_seasons == [2019, 2020, 2021, 2022]  # includes the holdout season, 2022
    assert set(combined_events["season"].unique()) == set(SEASONS)
    assert set(combined_rounds["season"].unique()) == set(SEASONS)


def test_main_runs_end_to_end_and_writes_reports(tmp_path):
    """Not a leakage test on its own (see test_run_holdout.py for that); just checks the pipeline
    runs and produces the report files, since this module duplicates real plumbing from
    run_holdout.py and run_simulator.py that is easy to get wrong when combining sources."""
    config = _config()
    events, rounds = _synthetic(SEASONS)
    config = _write(tmp_path, config, events, rounds)
    config["evaluation"] = {**config["evaluation"], "report_dir": str(tmp_path / "reports")}
    config["sim"] = {**config["sim"], "n_sims": 200}

    original_load_config = run_forward_test.load_config
    run_forward_test.load_config = lambda *a, **k: config
    try:
        run_forward_test.main()
    finally:
        run_forward_test.load_config = original_load_config

    report_dir = tmp_path / "reports"
    assert (report_dir / "forward_test_2026_results.md").exists()
    assert (report_dir / "forward_test_2026_metrics.csv").exists()
    assert (report_dir / "forward_test_2026_calibration.png").exists()
