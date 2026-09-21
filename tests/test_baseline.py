import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.models import baseline
from src.models.run_baseline1 import select_seasons, walk_forward_predictions

COLUMN = "sg_total_rolling_3"


def _simulate(seasons, events_per_season=30, field=25, seed=0, slope=1.2) -> pd.DataFrame:
    """Fields where results depend on a rolling-SG feature; some players are missing it."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for e in range(events_per_season):
            skill = rng.normal(0, 1, field)
            feature = skill + rng.normal(0, 0.5, field)
            score = skill + rng.gumbel(0, 1 / slope, field)  # higher wins
            rank = (-score).argsort().argsort() + 1
            for i in range(field):
                rows.append(
                    {
                        "season": season,
                        "tournament_id": f"R{season}{e:03d}",
                        "player_id": i,
                        COLUMN: np.nan if i % 7 == 0 else feature[i],
                        "win": rank[i] == 1,
                        "top10": rank[i] <= 10,
                        "made_cut": rank[i] <= field // 2,
                    }
                )
    frame = pd.DataFrame(rows)
    return frame.astype({"win": "boolean", "top10": "boolean", "made_cut": "boolean"})


def test_design_matrix_centres_and_flags_missing_values():
    frame = pd.DataFrame({COLUMN: [1.0, np.nan, 3.0]})

    design = baseline.design_matrix(frame, COLUMN, center=2.0)

    assert design.tolist() == [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]


def test_win_probabilities_sum_to_one_in_every_event_even_with_shuffled_rows():
    train = _simulate([2019, 2020])
    model = baseline.fit_baseline(train, "win", COLUMN)
    test = _simulate([2021], events_per_season=5, seed=9).sample(frac=1, random_state=3)

    probabilities = pd.Series(model.predict(test), index=test.index)

    sums = probabilities.groupby(test["tournament_id"]).sum()
    assert sums.to_numpy() == pytest.approx(1.0)


def test_win_softmax_recovers_the_slope_and_penalises_missing_history():
    model = baseline.fit_baseline(_simulate(range(2019, 2025), seed=1), "win", COLUMN)

    slope, missing_effect = model.weights
    assert slope == pytest.approx(1.2 * 1.0, abs=0.35)  # skill noise makes the fitted slope smaller
    assert slope > 0.5
    assert missing_effect == pytest.approx(0, abs=0.7)  # missing is random in the simulation


@pytest.mark.parametrize("label", ["made_cut", "top10"])
def test_logistic_labels_rank_better_players_higher(label):
    model = baseline.fit_baseline(_simulate(range(2019, 2023)), label, COLUMN)
    probes = pd.DataFrame({COLUMN: [-2.0, 0.0, 2.0], "tournament_id": "x"})

    low, mid, high = model.predict(probes)

    assert low < mid < high


def test_predictions_do_not_depend_on_test_labels():
    """Leakage: a walk-forward fold is fitted on earlier seasons, so test outcomes cannot matter."""
    events = _simulate([2019, 2020, 2021])
    flipped = events.copy()
    for label in ["win", "top10", "made_cut"]:
        flipped.loc[flipped["season"] == 2021, label] = ~flipped.loc[
            flipped["season"] == 2021, label
        ].astype(bool)
    folds = [([2019, 2020], 2021)]

    for label in ["win", "top10", "made_cut"]:
        original = walk_forward_predictions(events, label, COLUMN, folds)
        changed = walk_forward_predictions(flipped, label, COLUMN, folds)
        assert original["p"].to_numpy() == pytest.approx(changed["p"].to_numpy())
        assert (original["season"] == 2021).all()


def test_naive_forecast_is_the_training_base_rate():
    events = _simulate([2019, 2020, 2021])
    train = events[events["season"].isin([2019, 2020])]

    predictions = walk_forward_predictions(events, "top10", COLUMN, [([2019, 2020], 2021)])

    assert predictions["p_naive"].nunique() == 1
    assert predictions["p_naive"].iloc[0] == pytest.approx(train["top10"].astype(float).mean())


def test_reserved_seasons_are_removed_before_anything_is_fitted():
    config = load_config()
    events = _simulate(range(2018, 2027), events_per_season=2)

    kept, folds = select_seasons(events, config)

    assert set(kept["season"]) == {2019, 2020, 2021, 2022, 2023, 2024}
    assert not kept["season"].isin(config["splits"]["holdout_seasons"]).any()
    assert not kept["season"].isin(config["splits"]["forward_test_seasons"]).any()
    assert all(max(train) < test for train, test in folds)


def test_a_config_that_validates_on_the_holdout_is_refused():
    config = load_config()
    config["splits"]["validation_seasons"] = [2024, 2025]

    with pytest.raises(ValueError, match="reserved"):
        select_seasons(_simulate(range(2018, 2027), events_per_season=2), config)
