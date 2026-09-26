import numpy as np
import pandas as pd
import pytest

from src.sim import run_simulator

SIM_SETTINGS = {"n_sims": 500, "fallback_cut_fraction": 0.5}


def _features(
    seasons, n_players=16, per_season_events=3, seed=0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A minimal features-shaped frame plus a matching rounds frame for one no-cut, standard event
    and several cut events per season."""
    rng = np.random.default_rng(seed)
    event_rows, round_rows = [], []
    for season in seasons:
        for e in range(per_season_events):
            has_cut = e != 0  # the first event of each season has no cut
            tournament_id = f"R{season}{e:03d}"
            rating = rng.normal(0, 1, n_players)
            cut_count = n_players // 2
            for i in range(n_players):
                made_cut = (i < cut_count) if has_cut else pd.NA
                event_rows.append(
                    {
                        "season": season,
                        "tournament_id": tournament_id,
                        "player_id": i,
                        "rating": rating[i],
                        "no_history": int(i == 0),  # one player per event has no history
                        "is_standard_event": True,
                        "end_date": f"{season}-01-05",
                        "has_cut": has_cut,
                        "made_cut": made_cut,
                        "top10": i < 10,
                        "win": i == 0,
                    }
                )
                for number in range(1, 5):
                    round_rows.append(
                        {
                            "season": season,
                            "tournament_id": tournament_id,
                            "player_id": i,
                            "round": number,
                            "score": 70 - rating[i] + rng.normal(0, 2, 1)[0],
                            "round_date": f"{season}-01-{1 + number:02d}",
                            "event_end_date": f"{season}-01-05",
                            "is_standard_event": True,
                        }
                    )
    events = pd.DataFrame(event_rows).astype(
        {"made_cut": "boolean", "top10": "boolean", "win": "boolean"}
    )
    return events, pd.DataFrame(round_rows)


def test_simulate_predictions_returns_valid_probabilities_for_every_label():
    features, rounds = _features([2019, 2020, 2021])
    folds = [([2019, 2020], 2021)]

    predictions = run_simulator.simulate_predictions(
        features, rounds, folds, SIM_SETTINGS, "actual", seed=0
    )

    assert set(predictions) == set(run_simulator.LABELS)
    for frame in predictions.values():
        assert ((frame["p"] >= 0) & (frame["p"] <= 1)).all()
        assert frame["y"].isin([0, 1]).all()
        assert (frame["season"] == 2021).all()
        assert frame["p_naive"].nunique() == 1  # one training base rate per fold


def test_simulate_predictions_excludes_rows_where_the_label_is_undefined():
    """The first event of each season has no cut, so made_cut is undefined for all its players."""
    features, rounds = _features([2019, 2020, 2021], per_season_events=2)
    folds = [([2019, 2020], 2021)]
    no_cut_event = "R2021000"

    predictions = run_simulator.simulate_predictions(
        features, rounds, folds, SIM_SETTINGS, "actual", seed=0
    )

    assert no_cut_event not in set(predictions["made_cut"]["tournament_id"])
    assert no_cut_event in set(predictions["top10"]["tournament_id"])  # top10 is always defined


def test_simulate_predictions_probability_matches_a_direct_simulate_events_call():
    """Regression test for the column-name collision between simulated probabilities and labels."""
    features, rounds = _features([2019, 2020, 2021], per_season_events=1)
    folds = [([2019, 2020], 2021)]

    predictions = run_simulator.simulate_predictions(
        features, rounds, folds, SIM_SETTINGS, "actual", seed=0
    )

    from src.sim.tournament import estimate_residual_sigma, simulate_events

    train = features[features["season"].isin([2019, 2020])]
    sigma = estimate_residual_sigma(train, rounds[rounds["season"].isin([2019, 2020])])
    direct = simulate_events(
        features[features["season"] == 2021], sigma, SIM_SETTINGS["n_sims"], 0, "actual", 0.5
    )

    merged = predictions["top10"].merge(direct, on=["tournament_id", "player_id"])
    assert merged["p"].to_numpy() == pytest.approx(merged["top10"].to_numpy())


def test_time_ten_thousand_sims_reports_the_biggest_field():
    features, _ = _features([2021], n_players=20, per_season_events=1)

    field_size, seconds = run_simulator.time_ten_thousand_sims(features, sigma=2.5)

    assert field_size == 20
    assert seconds < 3.0
