import numpy as np
import pandas as pd
import pytest

from src.sim import tournament as sim


def test_probabilities_sum_to_the_right_totals_with_a_cut():
    rng = np.random.default_rng(0)
    rating = np.linspace(-1, 1, 40)

    result = sim.simulate_event(rating, sigma=2.0, n_sims=4000, rng=rng, cut_keep=20)

    assert result["win"].sum() == pytest.approx(1.0, abs=1e-9)
    assert result["top5"].sum() == pytest.approx(5.0, abs=1e-9)
    assert result["top10"].sum() == pytest.approx(10.0, abs=1e-9)
    assert result["made_cut"].sum() == pytest.approx(20.0, abs=1e-6)
    for key in result:
        assert ((result[key] >= 0) & (result[key] <= 1)).all()


def test_no_cut_means_everyone_finishes():
    rng = np.random.default_rng(0)
    result = sim.simulate_event(np.zeros(30), sigma=2.0, n_sims=1000, rng=rng, cut_keep=None)

    assert (result["made_cut"] == 1.0).all()
    assert result["top10"].sum() == pytest.approx(10.0, abs=1e-9)


def test_cut_keep_at_least_the_field_size_is_the_same_as_no_cut():
    rng1, rng2 = np.random.default_rng(5), np.random.default_rng(5)
    rating = np.linspace(-1, 1, 10)

    with_cut = sim.simulate_event(rating, 2.0, 2000, rng1, cut_keep=10)
    without_cut = sim.simulate_event(rating, 2.0, 2000, rng2, cut_keep=None)

    for key in with_cut:
        assert with_cut[key] == pytest.approx(without_cut[key])


def test_better_rated_players_win_and_place_more_often():
    rng = np.random.default_rng(1)
    rating = np.array([1.5, 0.5, 0.0, -0.5, -1.5])

    result = sim.simulate_event(rating, sigma=2.0, n_sims=30000, rng=rng, cut_keep=None)

    for key in ["win", "top5", "top10"]:
        assert list(result[key]) == sorted(result[key], reverse=True)


def test_a_big_skill_gap_almost_guarantees_the_cut_outcome():
    rng = np.random.default_rng(2)
    rating = np.array([10.0] * 5 + [-10.0] * 5)  # ratings are strokes per round; 10 is enormous

    result = sim.simulate_event(rating, sigma=2.0, n_sims=5000, rng=rng, cut_keep=5)

    assert (result["made_cut"][:5] > 0.999).all()
    assert (result["made_cut"][5:] < 0.001).all()


def test_zero_sigma_is_fully_determined_by_rating():
    rng = np.random.default_rng(3)
    rating = np.array([2.0, 1.0, 0.0, -1.0])

    result = sim.simulate_event(rating, sigma=0.0, n_sims=100, rng=rng, cut_keep=None)

    assert result["win"].tolist() == [1.0, 0.0, 0.0, 0.0]  # the best rating always wins


@pytest.mark.parametrize(
    ("has_cut", "made_cut", "mode", "field_size", "expected"),
    [
        (False, None, "actual", 150, None),
        (True, pd.array([True, True, False], dtype="boolean"), "actual", 3, 2),
        (True, pd.array([True, pd.NA], dtype="boolean"), "actual", 2, 1),  # NA is not counted
        (True, None, "fraction", 150, 75),
        (True, None, "fraction", 3, 2),  # rounds to at least 1, and up in this case
    ],
)
def test_cut_survivor_count(has_cut, made_cut, mode, field_size, expected):
    assert (
        sim.cut_survivor_count(field_size, has_cut, made_cut, mode, fallback_fraction=0.5)
        == expected
    )


def test_cut_survivor_count_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="Unknown cut mode"):
        sim.cut_survivor_count(100, True, pd.array([True], dtype="boolean"), "bogus", 0.5)


def _synthetic_history(n_players=30, n_events=25, true_sigma=2.5, seed=0):
    """Rounds and a features frame where every round is exactly rating + noise(true_sigma)."""
    rng = np.random.default_rng(seed)
    true_rating = rng.normal(0, 1, n_players)
    event_rows, round_rows = [], []
    for e in range(n_events):
        for player in range(n_players):
            for _round_number in range(4):
                round_rows.append(
                    {
                        "tournament_id": f"E{e}",
                        "player_id": player,
                        "round_date": f"2020-01-{1 + e:02d}",
                        "event_end_date": f"2020-01-{1 + e:02d}",
                        "score": 0.0,  # unused: we bypass round_relative_scores below
                    }
                )
        for player in range(n_players):
            event_rows.append(
                {
                    "tournament_id": f"E{e}",
                    "player_id": player,
                    "rating": true_rating[player],
                    "no_history": 0,
                }
            )
    return pd.DataFrame(event_rows), pd.DataFrame(round_rows), true_rating


def test_estimate_residual_sigma_recovers_a_known_noise_level(monkeypatch):
    features, rounds, true_rating = _synthetic_history(true_sigma=2.5)
    rng = np.random.default_rng(0)

    # Bypass round_relative_scores (it needs full tidy columns) with hand-built, known residuals.
    history = pd.DataFrame(
        {
            "tournament_id": rounds["tournament_id"],
            "player_id": rounds["player_id"],
            "rel": [
                -true_rating[p] + n
                for p, n in zip(rounds["player_id"], rng.normal(0, 2.5, len(rounds)), strict=True)
            ],
        }
    )
    monkeypatch.setattr("src.sim.tournament.round_relative_scores", lambda rounds, events: history)

    sigma = sim.estimate_residual_sigma(features, rounds)

    assert sigma == pytest.approx(2.5, rel=0.05)


def test_estimate_residual_sigma_excludes_players_with_no_history(monkeypatch):
    features = pd.DataFrame(
        {
            "tournament_id": ["E1", "E1", "E1"],
            "player_id": [1, 2, 3],
            "rating": [0.0, 0.0, 0.0],
            "no_history": [1, 0, 0],
        }
    )
    # Player 1 (no history) has a huge, misleading residual; players 2 and 3 have small, known ones.
    history = pd.DataFrame(
        {"tournament_id": ["E1"] * 3, "player_id": [1, 2, 3], "rel": [100.0, 0.3, -0.3]}
    )
    monkeypatch.setattr("src.sim.tournament.round_relative_scores", lambda rounds, events: history)

    sigma = sim.estimate_residual_sigma(features, pd.DataFrame())

    assert sigma == pytest.approx(0.3 * np.sqrt(2), abs=1e-9)  # matches players 2 and 3 alone


def _features_for_events(tournament_ids, n_players=15, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for tournament_id in tournament_ids:
        rating = rng.normal(0, 1, n_players)
        cut_count = n_players // 2
        made_cut = pd.array([i < cut_count for i in range(n_players)], dtype="boolean")
        for i in range(n_players):
            rows.append(
                {
                    "tournament_id": tournament_id,
                    "player_id": i,
                    "rating": rating[i],
                    "is_standard_event": True,
                    "has_cut": True,
                    "made_cut": made_cut[i],
                }
            )
    return pd.DataFrame(rows)


def test_simulate_events_matches_simulate_event_for_a_single_event():
    features = _features_for_events(["E1"])
    group = features[features["tournament_id"] == "E1"]

    combined = sim.simulate_events(
        features, sigma=2.0, n_sims=3000, seed=7, cut_mode="actual", fallback_cut_fraction=0.5
    )
    alone = sim.simulate_event(
        group["rating"].to_numpy(),
        2.0,
        3000,
        sim.event_rng(7, "E1"),
        cut_keep=int(group["made_cut"].sum()),
    )

    for key in ["win", "top5", "top10", "made_cut"]:
        assert combined[key].to_numpy() == pytest.approx(alone[key])


def test_simulate_events_is_reproducible_and_independent_of_other_events():
    features = _features_for_events(["E1", "E2", "E3"])

    full = sim.simulate_events(
        features, 2.0, 2000, seed=1, cut_mode="actual", fallback_cut_fraction=0.5
    )
    again = sim.simulate_events(
        features, 2.0, 2000, seed=1, cut_mode="actual", fallback_cut_fraction=0.5
    )
    without_e1 = sim.simulate_events(
        features[features["tournament_id"] != "E1"],
        2.0,
        2000,
        seed=1,
        cut_mode="actual",
        fallback_cut_fraction=0.5,
    )

    pd.testing.assert_frame_equal(full, again)
    e2_full = full[full["tournament_id"] == "E2"].reset_index(drop=True)
    e2_without = without_e1[without_e1["tournament_id"] == "E2"].reset_index(drop=True)
    pd.testing.assert_frame_equal(e2_full, e2_without)


def test_ten_thousand_simulations_of_a_large_field_run_in_well_under_a_second():
    import time

    rng = np.random.default_rng(0)
    rating = np.random.default_rng(1).normal(0, 0.7, 156)

    start = time.perf_counter()
    sim.simulate_event(rating, sigma=2.5, n_sims=10000, rng=rng, cut_keep=70)
    elapsed = time.perf_counter() - start

    assert elapsed < 3.0


def test_cut_keep_zero_means_nobody_survives():
    rng = np.random.default_rng(4)
    rating = np.array([1.0, 0.0, -1.0])

    result = sim.simulate_event(rating, sigma=2.0, n_sims=500, rng=rng, cut_keep=0)

    assert (result["made_cut"] == 0.0).all()
    assert result["win"].sum() == pytest.approx(0.0)  # nobody survives to be ranked
    assert result["top10"].sum() == pytest.approx(0.0)


def test_cut_keep_negative_is_treated_like_zero():
    rng = np.random.default_rng(4)
    result = sim.simulate_event(np.zeros(3), sigma=2.0, n_sims=500, rng=rng, cut_keep=-1)

    assert (result["made_cut"] == 0.0).all()
