"""Tidy player-event and player-round tables from the scraper output.

Run with `uv run python -m src.ingest.tidy`. The rules for which events count and how the
labels are defined are logged in docs/DECISIONS.md and described in docs/DATA_STRUCTURE.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Label definitions.
TOP_N = 10  # "top 10" includes ties (T10 counts)
STANDARD_ROUNDS = 4  # a standard event is four rounds of stroke play
# Stableford events (the Barracuda Championship) report points, not strokes. Their round scores
# sit far below any real stroke round, so a low median flags them.
MIN_STROKE_MEDIAN = 40

# No 18-hole round on tour is better than 13 under par, so anything below -15 is a data error
# (one W/D record shows 42 strokes and -30 to par).
MAX_UNDER_PAR = 15

# The 2020 season was cut short by COVID (fewer events and about a quarter fewer measured rounds
# per player), so its season strokes gained is noisier. Flagged, not dropped: decide on
# validation seasons only whether to exclude it.
COVID_SEASON = 2020

ROUND_COLUMNS = ["round_1", "round_2", "round_3", "round_4"]
MATCH_PLAY_COLUMNS = ["round_5", "round_6", "round_7"]

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
INPUTS = {
    "2018_2025": DATA_DIR / "pga_training_data.csv",
    "2026": DATA_DIR / "pga_prediction_data_2026.csv",
}


def parse_position(position: object) -> tuple[float, str]:
    """Split a position such as `T13`, `CUT`, `W/D` or `DQ` into (finish_position, status).

    The position is NaN unless the player finished. Anything unrecognised raises, so a new
    format in the data cannot slip through as a wrong label.
    """
    text = str(position).strip().upper().replace("/", "")
    if text == "CUT":
        return np.nan, "cut"
    if text == "WD":
        return np.nan, "wd"
    if text == "DQ":
        return np.nan, "dq"
    match = re.fullmatch(r"T?(\d+)", text)
    if match is None:
        raise ValueError(f"Unknown position: {position!r}")
    return float(match.group(1)), "finished"


def parse_to_par(value: object) -> float:
    """`E` -> 0, `+3` -> 3, `-5` -> -5; anything else (for example `-`) -> NaN."""
    text = str(value).strip().upper()
    if text == "E":
        return 0.0
    try:
        return float(text)
    except ValueError:
        return np.nan


def prepare_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Add finish position, status, rounds played and the event flags to each result row."""
    rows = results.copy()
    rows = rows.reindex(columns=[*rows.columns, *(c for c in MATCH_PLAY_COLUMNS if c not in rows)])

    parsed = rows["position"].map(parse_position)
    rows["finish_position"] = [finish for finish, _ in parsed]
    rows["status"] = [status for _, status in parsed]
    rows["rounds_played"] = rows[ROUND_COLUMNS].notna().sum(axis=1)

    is_match_play_row = rows[MATCH_PLAY_COLUMNS].notna().any(axis=1) | rows[
        "tournament_name"
    ].str.contains("Match Play", case=False)
    grouped = rows.assign(_match_play=is_match_play_row).groupby("tournament_id")
    events = pd.DataFrame(
        {
            "n_rounds_event": grouped["rounds_played"].max(),
            "has_cut": grouped["status"].agg(lambda s: bool((s == "cut").any())),
            "is_match_play": grouped["_match_play"].any(),
            "median_round_1": grouped["round_1"].median(),
            "field_size": grouped["player_id"].size(),
        }
    )
    events["is_stroke_play"] = (~events["is_match_play"]) & (
        events["median_round_1"] >= MIN_STROKE_MEDIAN
    )
    events["is_standard_event"] = events["is_stroke_play"] & (
        events["n_rounds_event"] == STANDARD_ROUNDS
    )
    return rows.merge(events.drop(columns="median_round_1"), on="tournament_id", how="left")


def _label(condition: pd.Series, defined: pd.Series) -> pd.Series:
    """A boolean label that is missing wherever it is not defined."""
    return condition.astype("boolean").where(defined)


def build_player_events(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per player per event, with labels and the rolling strokes-gained features.

    Labels (all missing for events that are not standard four-round stroke play):
      win       finished first
      top10     finished in the top 10, ties included; false for cut, W/D and DQ
      made_cut  true if the player finished, false if cut; missing for W/D and DQ, and for
                events with no cut
    """
    finished = rows["status"] == "finished"
    standard = rows["is_standard_event"]

    events = rows.copy()
    events["win"] = _label(finished & (rows["finish_position"] == 1), standard)
    events["top10"] = _label(finished & (rows["finish_position"] <= TOP_N), standard)
    made_cut_defined = standard & rows["has_cut"] & rows["status"].isin(["finished", "cut"])
    events["made_cut"] = _label(finished, made_cut_defined)
    events["season"] = rows["year"]
    events["covid_season"] = events["season"] == COVID_SEASON

    keep = [
        "tournament_id",
        "tournament_name",
        "season",
        "start_date",
        "end_date",
        "player_id",
        "display_name",
        "amateur",
        "position",
        "finish_position",
        "status",
        "rounds_played",
        "n_rounds_event",
        "field_size",
        "has_cut",
        "is_stroke_play",
        "is_standard_event",
        "covid_season",
        "made_cut",
        "top10",
        "win",
    ]
    rolling = [c for c in events.columns if "rolling" in c]
    return events[keep + rolling].reset_index(drop=True)


def build_player_rounds(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per player per round played.

    `round_date` is the event start date plus the round number minus one. It is approximate
    (weather delays, Monday finishes) but is enough for an as-of rule of "only rounds before".
    `par` is the score minus the to-par value, taken per row because some events use several
    courses. Standard-event rounds better than MAX_UNDER_PAR under par are dropped as data errors.
    """
    parts = []
    for number, column in enumerate(ROUND_COLUMNS, start=1):
        part = rows.loc[rows[column].notna()].copy()
        part["round"] = number
        part["score"] = part[column].astype("Int64")
        part["to_par"] = part[f"{column}_to_par"].map(parse_to_par)
        parts.append(part)
    rounds = pd.concat(parts, ignore_index=True)
    impossible = rounds["is_standard_event"] & (rounds["to_par"] < -MAX_UNDER_PAR)
    rounds = rounds[~impossible]

    rounds["par"] = rounds["score"] - rounds["to_par"]
    rounds["season"] = rounds["year"]
    rounds["round_date"] = (
        pd.to_datetime(rounds["start_date"]) + pd.to_timedelta(rounds["round"] - 1, unit="D")
    ).dt.strftime("%Y-%m-%d")

    keep = [
        "tournament_id",
        "season",
        "player_id",
        "round",
        "score",
        "to_par",
        "par",
        "start_date",
        "round_date",
        "is_standard_event",
    ]
    return rounds[keep].sort_values(["tournament_id", "player_id", "round"]).reset_index(drop=True)


def validate_tidy(events: pd.DataFrame, rounds: pd.DataFrame) -> None:
    """Raise if the tables break a basic rule."""
    if events.duplicated(["tournament_id", "player_id"]).any():
        raise ValueError("Duplicate (tournament_id, player_id) rows in player_events.")
    if rounds.duplicated(["tournament_id", "player_id", "round"]).any():
        raise ValueError("Duplicate rounds in player_rounds.")
    if (pd.to_datetime(rounds["round_date"]) < pd.to_datetime(rounds["start_date"])).any():
        raise ValueError("A round is dated before its event starts.")

    standard = events[events["is_standard_event"]]
    winners = standard.groupby("tournament_id")["win"].sum()
    if (winners != 1).any():
        raise ValueError(
            f"Standard events without exactly one winner: {list(winners[winners != 1].index)}"
        )
    top10 = standard.groupby("tournament_id")["top10"].sum()
    if (top10 < TOP_N).any():
        raise ValueError(f"Standard events with fewer than {TOP_N} in the top 10.")
    cut_events = standard[standard["has_cut"]]
    if (
        cut_events.groupby("tournament_id")["made_cut"]
        .agg(lambda s: s.dropna().nunique())
        .lt(2)
        .any()
    ):
        raise ValueError("A cut event has only one made_cut outcome.")


def build_tidy_tables(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(player_events, player_rounds) for a results table with the scraper's columns."""
    rows = prepare_rows(results)
    events = build_player_events(rows)
    rounds = build_player_rounds(rows)
    validate_tidy(events, rounds)
    return events, rounds


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for label, path in INPUTS.items():
        events, rounds = build_tidy_tables(pd.read_csv(path, low_memory=False))
        events.to_csv(PROCESSED_DIR / f"player_events_{label}.csv", index=False)
        rounds.to_csv(PROCESSED_DIR / f"player_rounds_{label}.csv", index=False)
        standard = events[events["is_standard_event"]]
        print(
            f"{label}: {len(events)} player-events "
            f"({standard['tournament_id'].nunique()} of {events['tournament_id'].nunique()} "
            f"events standard), {len(rounds)} player-rounds"
        )


if __name__ == "__main__":
    main()
