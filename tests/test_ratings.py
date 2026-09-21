import numpy as np
import pandas as pd
import pytest

from src.features import ratings

PLAYERS = [1, 2, 3, 4]
# Three events a month apart, each four days long.
EVENT_DATES = {
    "E1": ("2022-01-06", "2022-01-09"),
    "E2": ("2022-02-03", "2022-02-06"),
    "E3": ("2022-03-03", "2022-03-06"),
}


def _world(scores: dict[str, dict[int, list[int]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Events and rounds for the four players; `scores[event][player]` lists the round scores."""
    event_rows, round_rows = [], []
    for tournament_id, (start, end) in EVENT_DATES.items():
        for player in PLAYERS:
            event_rows.append(
                {
                    "tournament_id": tournament_id,
                    "player_id": player,
                    "start_date": start,
                    "end_date": end,
                    "field_size": len(PLAYERS),
                    "amateur": False,
                }
            )
            for number, score in enumerate(scores[tournament_id][player], start=1):
                round_rows.append(
                    {
                        "tournament_id": tournament_id,
                        "player_id": player,
                        "round": number,
                        "score": score,
                        "round_date": str(pd.Timestamp(start) + pd.Timedelta(days=number - 1))[:10],
                        "is_standard_event": True,
                    }
                )
    return pd.DataFrame(event_rows), pd.DataFrame(round_rows)


BASE = {
    "E1": {1: [68, 69], 2: [70, 71], 3: [72, 73], 4: [74, 75]},
    "E2": {1: [69, 70], 2: [71, 70], 3: [73, 72], 4: [75, 74]},
    "E3": {1: [70, 68], 2: [72, 70], 3: [74, 72], 4: [76, 74]},
}


def _features(scores) -> pd.DataFrame:
    events, rounds = _world(scores)
    return ratings.build_features(events, rounds, 365, 60, 10)


def test_round_relative_scores_are_measured_against_the_rounds_field():
    events, rounds = _world(BASE)
    rounds.loc[0, "is_standard_event"] = False  # E1 round 1 of player 1 (68) is left out

    relative = ratings.round_relative_scores(rounds, events)

    assert len(relative) == len(rounds) - 1
    assert relative.groupby(["tournament_id", "round_date"])["rel"].sum().abs().max() < 1e-9
    first_round = relative[
        (relative["tournament_id"] == "E1") & (relative["round_date"] == "2022-01-06")
    ]
    assert sorted(first_round["rel"]) == [-2.0, 0.0, 2.0]  # 70, 72, 74 against their mean of 72


def test_decayed_history_matches_a_hand_calculation():
    history = pd.DataFrame(
        {
            "player_id": [1, 1],
            "rel": [-2.0, 1.0],  # 2 strokes better than the field, then 1 worse
            "round_date": ["2022-01-01", "2023-01-01"],
            "event_end_date": ["2022-01-04", "2023-01-04"],
        }
    )
    targets = pd.DataFrame({"player_id": [1], "as_of": ["2023-02-05"]})

    out = ratings.decayed_history(history, targets, half_life_days=365)

    old = 2.0 ** (-400 / 365)  # 2022-01-01 to 2023-02-05
    recent = 2.0 ** (-35 / 365)
    assert out["weight_sum"].iloc[0] == pytest.approx(old + recent)
    assert out["strength_sum"].iloc[0] == pytest.approx(2 * old - 1 * recent)
    assert out["days_since_last"].iloc[0] == 32  # from 2023-01-04


def test_only_events_that_ended_strictly_before_as_of_are_used():
    history = pd.DataFrame(
        {
            "player_id": [1],
            "rel": [-1.0],
            "round_date": ["2022-01-06"],
            "event_end_date": ["2022-01-09"],
        }
    )
    on_end = ratings.decayed_history(
        history, pd.DataFrame({"player_id": [1], "as_of": ["2022-01-09"]}), 365
    )
    after = ratings.decayed_history(
        history, pd.DataFrame({"player_id": [1], "as_of": ["2022-01-10"]}), 365
    )

    assert on_end["weight_sum"].iloc[0] == 0
    assert after["weight_sum"].iloc[0] > 0


def test_shrunk_rating_pulls_thin_evidence_toward_zero():
    # Both players average 2 strokes better than the field, one over 5 rounds, one over 50.
    strength, weight = pd.Series([10.0, 100.0]), pd.Series([5.0, 50.0])

    thin, thick = ratings.shrunk_rating(strength, weight, shrinkage_rounds=10)

    assert thin == pytest.approx(10 / 15) and thick == pytest.approx(100 / 60)
    assert 0 < thin < thick < 2  # both shrink toward 0, the thin one much more
    assert ratings.shrunk_rating(pd.Series([0.0]), pd.Series([0.0]), 10).iloc[0] == 0


def test_first_event_has_no_history_and_later_events_do():
    features = _features(BASE)
    first = features[features["tournament_id"] == "E1"]
    second = features[features["tournament_id"] == "E2"]

    assert (first["no_history"] == 1).all() and (first["rating"] == 0).all()
    assert (second["no_history"] == 0).all()
    assert (second["days_since_last"] == 25).all()  # 2022-01-09 to 2022-02-03


def test_better_players_get_higher_ratings():
    third = _features(BASE)
    third = third[third["tournament_id"] == "E3"].set_index("player_id")["rating"]

    assert list(third.sort_values(ascending=False).index) == [1, 2, 3, 4]


def test_field_relative_features_are_centred_and_ranked_within_each_event():
    features = _features(BASE)

    assert features.groupby("tournament_id")["rating_rel"].sum().abs().max() < 1e-9
    last = features[features["tournament_id"] == "E3"].set_index("player_id")
    assert last.loc[1, "rank_pct"] == pytest.approx(0.25)
    assert last.loc[4, "rank_pct"] == pytest.approx(1.0)


def test_features_of_an_event_never_depend_on_its_own_or_later_scores():
    """Leakage: as-of features use only events that ended before the event starts."""
    base = _features(BASE)
    columns = ["rating", "form", "log_rounds", "no_history", "days_since_last", "rating_rel"]

    changed_last = {**BASE, "E3": {p: [60, 60] for p in PLAYERS}}  # rewrite the last event
    after = _features(changed_last)
    pd.testing.assert_frame_equal(base[columns], after[columns])

    changed_middle = {**BASE, "E2": {1: [80, 80], 2: [60, 60], 3: [80, 80], 4: [60, 60]}}
    after = _features(changed_middle)
    for event in ["E1", "E2"]:  # the middle event does not change its own or earlier features
        mask = base["tournament_id"] == event
        pd.testing.assert_frame_equal(base.loc[mask, columns], after.loc[mask, columns])
    assert not np.allclose(
        base.loc[base["tournament_id"] == "E3", "rating"],
        after.loc[after["tournament_id"] == "E3", "rating"],
    )  # but the information does flow forward to the later event
