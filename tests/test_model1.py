import numpy as np
import pandas as pd
import pytest

from src.models import model1

SETTINGS = {
    "top10_normalise": True,
    "logistic_c": 1.0,
    "win_l2": 1.0,
    "boosting": {
        "max_depth": 2,
        "learning_rate": 0.1,
        "max_iter": 30,
        "min_samples_leaf": 10,
        "l2_regularization": 1.0,
    },
}
SG_COLUMNS = [f"{c}_rolling_3" for c in model1.SG_CATEGORIES]


def _features(seasons, events_per_season=25, field=20, seed=0) -> pd.DataFrame:
    """Fields where results follow a rating; some players have no strokes-gained history."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for e in range(events_per_season):
            skill = rng.normal(0, 1, field)
            rating = skill + rng.normal(0, 0.5, field)
            rank = (-(skill + rng.gumbel(0, 1, field))).argsort().argsort() + 1
            for i in range(field):
                no_sg = i % 6 == 0
                row = {
                    "season": season,
                    "tournament_id": f"R{season}{e:03d}",
                    "player_id": i,
                    "rating_rel": rating[i] - rating.mean(),
                    "field_mean_rating": rating.mean(),
                    "field_size_log": np.log(field),
                    "rank_pct": (-rating).argsort().argsort()[i] / field,
                    "form": rng.normal(0, 0.2),
                    "log_rounds": 4.0,
                    "no_history": 0.0,
                    "days_since_last": 10.0,
                    "amateur": 0.0,
                    "win": rank[i] == 1,
                    "top10": rank[i] <= 10,
                    "made_cut": rank[i] <= field // 2,
                }
                row.update({c: np.nan if no_sg else rng.normal(0, 0.3) for c in SG_COLUMNS})
                rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.astype({"win": "boolean", "top10": "boolean", "made_cut": "boolean"})


def test_design_frame_fills_missing_strokes_gained_and_flags_it():
    frame = model1.design_frame(_features([2020], events_per_season=1))

    assert frame[model1.FULL_COLUMNS + ["rating_sq"]].notna().all().all()
    assert frame["no_sg"].sum() > 0
    assert (frame.loc[frame["no_sg"] == 1, model1.SG_CATEGORIES] == 0).all().all()


def test_normalise_to_target_sums_within_each_event_and_keeps_the_order():
    rng = np.random.default_rng(1)
    events = np.repeat(["a", "b", "c"], [30, 20, 12])
    probabilities = rng.uniform(0.02, 0.4, size=len(events))

    out = model1.normalise_to_target(probabilities, events, target=6.0)

    sums = pd.Series(out).groupby(events).sum()
    assert sums.to_numpy() == pytest.approx(6.0, abs=1e-6)
    assert ((out > 0) & (out < 1)).all()
    for event in ["a", "b", "c"]:  # ranking inside an event is unchanged
        mask = events == event
        assert (np.argsort(out[mask]) == np.argsort(probabilities[mask])).all()


def test_normalise_to_target_is_capped_below_the_field_size():
    out = model1.normalise_to_target(np.full(4, 0.3), np.array(["x"] * 4), target=10.0)

    assert out.sum() == pytest.approx(3.5, abs=1e-4)


@pytest.mark.parametrize("kind", ["logistic", "boosting"])
def test_top10_probabilities_are_valid_and_add_up_to_the_target(kind):
    train, test = _features([2019, 2020]), _features([2021], events_per_season=6, seed=5)
    model = model1.fit_model1(train, "top10", kind, model1.FULL_COLUMNS, SETTINGS, seed=0)

    probabilities = pd.Series(model.predict(test), index=test.index)

    assert ((probabilities > 0) & (probabilities < 1)).all()
    sums = probabilities.groupby(test["tournament_id"]).sum()
    assert sums.to_numpy() == pytest.approx(train.groupby("tournament_id")["top10"].sum().mean())


def test_better_rated_players_get_higher_probabilities():
    train = _features([2019, 2020])
    for label, kind, columns in [
        ("made_cut", "logistic", model1.FULL_COLUMNS),
        ("top10", "logistic", model1.FULL_COLUMNS),
        ("win", "softmax", model1.WIN_COLUMNS),
    ]:
        model = model1.fit_model1(train, label, kind, columns, SETTINGS, seed=0)
        assert model.coefficients()["rating_rel"] > 0


def test_win_probabilities_sum_to_one_in_every_event():
    train, test = _features([2019, 2020]), _features([2021], events_per_season=5, seed=3)
    model = model1.fit_model1(train, "win", "softmax", model1.WIN_COLUMNS, SETTINGS, seed=0)

    sums = pd.Series(model.predict(test)).groupby(test["tournament_id"].to_numpy()).sum()

    assert sums.to_numpy() == pytest.approx(1.0)


def test_predictions_do_not_depend_on_test_labels():
    """Leakage: a fold is fitted on earlier seasons, so test-season outcomes cannot matter."""
    features = _features([2019, 2020, 2021])
    flipped = features.copy()
    in_test = flipped["season"] == 2021
    for label in ["win", "top10", "made_cut"]:
        flipped.loc[in_test, label] = ~flipped.loc[in_test, label].astype(bool)
    folds = [([2019, 2020], 2021)]

    for label, kind, columns in [
        ("made_cut", "logistic", model1.FULL_COLUMNS),
        ("top10", "boosting", model1.FULL_COLUMNS),
        ("win", "softmax", model1.WIN_COLUMNS),
    ]:
        a = model1.walk_forward_predictions(features, label, kind, columns, folds, SETTINGS, 0)
        b = model1.walk_forward_predictions(flipped, label, kind, columns, folds, SETTINGS, 0)
        assert a["p"].to_numpy() == pytest.approx(b["p"].to_numpy())
        assert (a["season"] == 2021).all()


def test_boosting_is_reproducible_for_a_fixed_seed():
    features = _features([2019, 2020, 2021])
    folds = [([2019, 2020], 2021)]

    first = model1.walk_forward_predictions(
        features, "top10", "boosting", model1.FULL_COLUMNS, folds, SETTINGS, 7
    )
    second = model1.walk_forward_predictions(
        features, "top10", "boosting", model1.FULL_COLUMNS, folds, SETTINGS, 7
    )

    assert first["p"].to_numpy() == pytest.approx(second["p"].to_numpy())
