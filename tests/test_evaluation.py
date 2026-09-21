import numpy as np
import pandas as pd
import pytest

from src.config import assert_no_holdout, load_config
from src.models import evaluation


def test_log_loss_and_brier_terms():
    assert evaluation.log_loss_terms([1], [0.5])[0] == pytest.approx(np.log(2))
    assert evaluation.brier_terms([1, 0], [0.8, 0.3]) == pytest.approx([0.04, 0.09])


def test_log_loss_is_finite_for_certain_wrong_forecasts():
    assert np.isfinite(evaluation.log_loss_terms([1], [0.0])[0])


def test_walk_forward_folds_expand_and_never_train_on_the_test_season():
    """Leakage: every training season is strictly earlier than the season being predicted."""
    folds = evaluation.walk_forward_folds(range(2018, 2026), [2023, 2024], first_train_season=2019)

    assert folds == [([2019, 2020, 2021, 2022], 2023), ([2019, 2020, 2021, 2022, 2023], 2024)]
    assert all(max(train) < test for train, test in folds)


def test_walk_forward_folds_need_a_training_season():
    with pytest.raises(ValueError, match="No training seasons"):
        evaluation.walk_forward_folds([2019, 2020], [2019], first_train_season=2019)


def _forecasts(n_events=40, per_event=10, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.6, size=n_events * per_event)
    return pd.DataFrame(
        {
            "event": np.repeat(np.arange(n_events), per_event),
            "y": (rng.uniform(size=p.size) < p).astype(int),
            "p": p,
            "p_naive": p.mean(),
        }
    )


def test_bootstrap_is_reproducible_and_seed_dependent():
    sums, counts = np.array([3.0, 5.0, 2.0, 8.0]), np.array([10.0, 10.0, 10.0, 10.0])

    first = evaluation.bootstrap_interval(sums, counts, 500, seed=1)
    assert first == evaluation.bootstrap_interval(sums, counts, 500, seed=1)
    assert first != evaluation.bootstrap_interval(sums, counts, 500, seed=2)
    assert first[0] <= sums.sum() / counts.sum() <= first[1]


def test_bootstrap_resamples_events_not_rows():
    """Two events with very different loss give a wide interval however many rows each has."""
    counts = np.array([1000.0, 1000.0])
    interval = evaluation.bootstrap_interval(np.array([100.0, 900.0]), counts, 1000, seed=0)

    assert interval[1] - interval[0] > 0.3  # a row-level bootstrap would give a width near 0.03


def test_score_forecasts_naive_against_itself_has_zero_difference():
    frame = _forecasts().assign(p=lambda f: f["p_naive"])

    scores = evaluation.score_forecasts(frame, samples=200, seed=0)

    assert scores["log_loss_delta"] == pytest.approx(0)
    assert scores["log_loss_delta_lo"] == pytest.approx(0)
    assert scores["log_loss_delta_hi"] == pytest.approx(0)


def test_score_forecasts_true_probabilities_beat_the_naive_forecast():
    scores = evaluation.score_forecasts(_forecasts(n_events=200), samples=200, seed=0)

    assert scores["log_loss_delta"] < 0
    assert scores["log_loss_delta_hi"] < 0  # the whole interval is below zero
    assert scores["rows"] == 2000 and scores["events"] == 200


def test_calibration_table_of_true_probabilities_is_close_to_the_diagonal():
    frame = _forecasts(n_events=500)

    table = evaluation.calibration_table(frame["y"], frame["p"], bins=5)

    assert table["rows"].sum() == len(frame)
    assert (table["mean_predicted"] - table["observed_rate"]).abs().max() < 0.05


def test_plot_calibration_writes_a_png(tmp_path):
    frame = _forecasts()
    table = evaluation.calibration_table(frame["y"], frame["p"], bins=5)

    evaluation.plot_calibration({"A": table, "B": table}, tmp_path / "plot.png", "Title")

    assert (tmp_path / "plot.png").stat().st_size > 1000


def test_holdout_and_forward_test_seasons_are_refused():
    config = load_config()

    assert_no_holdout(config, [2019, 2023, 2024])
    with pytest.raises(ValueError, match="reserved"):
        assert_no_holdout(config, [2024, 2025])
    with pytest.raises(ValueError, match="reserved"):
        assert_no_holdout(config, [2026])
