import numpy as np
import pandas as pd
import pytest

from src.ingest import tidy


def _event(
    tournament_id: str,
    positions: list[str],
    rounds: int = 4,
    round_score: float = 70.0,
    name: str = "Test Open",
    start_date: str = "2022-03-10",
) -> pd.DataFrame:
    """One event; every player shoots `round_score` in each round they play.

    CUT players play 2 rounds, W/D players play 1, everyone else plays `rounds`.
    """
    rows = []
    for player_id, position in enumerate(positions, start=1):
        played = {"CUT": 2, "W/D": 1}.get(position, rounds)
        row = {
            "tournament_id": tournament_id,
            "tournament_name": name,
            "year": 2022,
            "start_date": start_date,
            "end_date": "2022-03-13",
            "player_id": player_id,
            "display_name": f"Player {player_id}",
            "amateur": False,
            "position": position,
            "sg_total_rolling_3": 0.5,
        }
        for number in range(1, 5):
            has_round = number <= played
            row[f"round_{number}"] = round_score if has_round else np.nan
            row[f"round_{number}_to_par"] = "-2" if has_round else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# 15 players: a winner, ties for 10th, 11th place, cut, and a withdrawal.
CUT_EVENT_POSITIONS = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "T10",
    "T10",
    "12",
    "CUT",
    "CUT",
    "W/D",
]


def _events(results: pd.DataFrame) -> pd.DataFrame:
    return tidy.build_player_events(tidy.prepare_rows(results))


def _by_player(events: pd.DataFrame, column: str) -> pd.Series:
    return events.set_index("player_id")[column]


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        ("1", (1.0, "finished")),
        ("T13", (13.0, "finished")),
        ("CUT", (np.nan, "cut")),
        ("W/D", (np.nan, "wd")),
        ("DQ", (np.nan, "dq")),
    ],
)
def test_parse_position(position, expected):
    finish, status = tidy.parse_position(position)
    assert status == expected[1]
    assert (np.isnan(finish) and np.isnan(expected[0])) or finish == expected[0]


def test_parse_position_rejects_unknown_values():
    with pytest.raises(ValueError, match="Unknown position"):
        tidy.parse_position("MDF")


@pytest.mark.parametrize(
    ("text", "expected"), [("E", 0.0), ("+3", 3.0), ("-5", -5.0), (-2, -2.0), ("2", 2.0)]
)
def test_parse_to_par(text, expected):
    assert tidy.parse_to_par(text) == expected


def test_parse_to_par_bad_values_are_nan():
    assert np.isnan(tidy.parse_to_par("-"))
    assert np.isnan(tidy.parse_to_par(np.nan))


def test_labels_in_a_standard_cut_event():
    events = _events(_event("R2022001", CUT_EVENT_POSITIONS))

    win = _by_player(events, "win")
    top10 = _by_player(events, "top10")
    made_cut = _by_player(events, "made_cut")

    assert win.sum() == 1 and win[1]
    assert top10[[1, 9, 10, 11]].all()  # 1st, 9th and both T10 count
    assert not top10[12]  # 12th does not
    assert not top10[[13, 14, 15]].any()  # cut and W/D players are not top 10
    assert made_cut[list(range(1, 13))].all()
    assert not made_cut[[13, 14]].any()
    assert pd.isna(made_cut[15])  # W/D: no clean answer


def test_no_cut_event_has_no_made_cut_label_but_keeps_the_others():
    events = _events(_event("R2022002", [str(n) for n in range(1, 13)]))

    assert events["made_cut"].isna().all()
    assert events["win"].sum() == 1
    assert events["top10"].sum() == 10


@pytest.mark.parametrize(
    "event",
    [
        _event("R2022003", CUT_EVENT_POSITIONS, rounds=3),  # three-round event
        _event("R2022004", CUT_EVENT_POSITIONS, round_score=4.0),  # Stableford points
        _event("R2022005", CUT_EVENT_POSITIONS, name="WGC Dell Technologies Match Play"),
    ],
    ids=["three_rounds", "stableford", "match_play"],
)
def test_non_standard_events_have_no_labels(event):
    events = _events(event)

    assert not events["is_standard_event"].any()
    assert events[["win", "top10", "made_cut"]].isna().all().all()


def test_labels_of_one_event_do_not_depend_on_other_events():
    first = _event("R2022001", CUT_EVENT_POSITIONS)
    other = _event("R2022009", [str(n) for n in range(1, 13)], start_date="2022-06-01")

    alone = _events(first)
    together = _events(pd.concat([first, other], ignore_index=True))
    together = together[together["tournament_id"] == "R2022001"].reset_index(drop=True)

    pd.testing.assert_frame_equal(alone, together)


def test_player_rounds_has_one_row_per_round_played_with_dates():
    rows = tidy.prepare_rows(_event("R2022001", CUT_EVENT_POSITIONS))
    rounds = tidy.build_player_rounds(rows)

    assert len(rounds) == 12 * 4 + 2 * 2 + 1  # finishers, CUT players, the W/D player
    player_13 = rounds[rounds["player_id"] == 13]
    assert list(player_13["round"]) == [1, 2]
    assert list(rounds[rounds["player_id"] == 1]["round_date"]) == [
        "2022-03-10",
        "2022-03-11",
        "2022-03-12",
        "2022-03-13",
    ]
    assert (rounds["to_par"] == -2).all()
    assert (rounds["par"] == 72).all()  # 70 strokes at 2 under par


def test_rounds_never_start_before_their_event():
    """As-of safety: a round is never dated earlier than the event's start."""
    rows = tidy.prepare_rows(_event("R2022001", CUT_EVENT_POSITIONS))
    rounds = tidy.build_player_rounds(rows)

    assert (pd.to_datetime(rounds["round_date"]) >= pd.to_datetime(rounds["start_date"])).all()


def test_impossible_rounds_are_dropped_from_standard_events():
    results = _event("R2022001", CUT_EVENT_POSITIONS)
    results.loc[results["player_id"] == 15, "round_1"] = 42.0  # W/D player, 42 strokes
    results.loc[results["player_id"] == 15, "round_1_to_par"] = "-30"

    rounds = tidy.build_player_rounds(tidy.prepare_rows(results))

    assert rounds[rounds["player_id"] == 15].empty


def test_build_tidy_tables_passes_validation_for_a_clean_event():
    events, rounds = tidy.build_tidy_tables(_event("R2022001", CUT_EVENT_POSITIONS))

    assert len(events) == 15
    assert len(rounds) == 12 * 4 + 2 * 2 + 1


def test_validation_catches_a_standard_event_with_no_winner():
    positions = ["T1", "T1"] + [str(n) for n in range(3, 15)]  # two tied winners
    with pytest.raises(ValueError, match="exactly one winner"):
        tidy.build_tidy_tables(_event("R2022001", positions))


def test_validation_catches_duplicate_players():
    results = pd.concat([_event("R2022001", CUT_EVENT_POSITIONS)] * 2, ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        tidy.build_tidy_tables(results)
