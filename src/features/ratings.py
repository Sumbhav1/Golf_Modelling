"""Field-adjusted player ratings and field-relative features, computed as of an event's start.

Every rating uses only rounds from events that ended strictly before the `as_of` date, so nothing
from the event being predicted (or anything after it) can leak in.

Rating, in strokes per round better than the field:
  1. For every round of a standard event, score minus the average score of everyone in that round
     (`rel`). This judges difficulty by the field, not by the winner's score.
  2. A recency-weighted average of `-rel` over the player's earlier rounds (weights halve every
     `half_life_days`).
  3. Shrunk toward the tour average by `shrinkage_rounds` pseudo-rounds, so a rating built on few
     rounds counts for less.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAY = np.timedelta64(1, "D")


CUT_ROUNDS = (3, 4)


def _survivor_field_gap(standard: pd.DataFrame) -> pd.Series:
    """Per event, how many strokes per round better the round-3/4 field is than the full field.

    Rounds 3 and 4 are only played by players who made the cut, a stronger sub-field than rounds
    1-2's full field. Measuring a round-3/4 score against only that stronger sub-field understates
    a good player's performance (beating tough peers looks merely average) and inflates measured
    round-to-round noise. This returns the gap so it can be added back.
    """
    early = standard[standard["round"].isin([1, 2])]
    per_player = early.groupby(["tournament_id", "player_id"])["score"].mean()  # per-round pace
    full_field = per_player.groupby("tournament_id").mean()

    later = standard[standard["round"].isin(CUT_ROUNDS)][["tournament_id", "player_id", "round"]]
    survivor_pace = later.join(per_player.rename("pace"), on=["tournament_id", "player_id"])
    survivor_field = survivor_pace.groupby(["tournament_id", "round"])["pace"].mean()

    gap = full_field.reindex(survivor_field.index, level="tournament_id") - survivor_field
    return gap.rename("gap")


def round_relative_scores(rounds: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Standard-event rounds with `rel` (score minus the round's field average) and the event's end.

    Rounds of non-standard events are left out: Stableford points, scrambles and match play are
    not comparable strokes. Rounds 3 and 4 are corrected for the cut's selection effect (see
    `_survivor_field_gap`); a no-cut event's round-3/4 field is close to the full field, so the
    correction is close to zero there.
    """
    standard = rounds[rounds["is_standard_event"]].copy()
    standard["score"] = standard["score"].astype(float)
    standard["rel"] = standard["score"] - standard.groupby(["tournament_id", "round"])[
        "score"
    ].transform("mean")

    gap = _survivor_field_gap(standard)
    later = standard["round"].isin(CUT_ROUNDS)
    aligned_gap = (
        standard.loc[later].join(gap, on=["tournament_id", "round"], how="left")["gap"].fillna(0.0)
    )
    standard.loc[later, "rel"] = standard.loc[later, "rel"].to_numpy() - aligned_gap.to_numpy()

    ends = events.drop_duplicates("tournament_id")[["tournament_id", "end_date"]]
    standard = standard.merge(ends.rename(columns={"end_date": "event_end_date"}), how="left")
    return standard[["tournament_id", "player_id", "round_date", "event_end_date", "rel"]]


def decayed_history(
    history: pd.DataFrame, targets: pd.DataFrame, half_life_days: float
) -> pd.DataFrame:
    """Recency-weighted sums over each target's earlier rounds.

    `targets` needs `player_id` and `as_of` (a date). Only rounds whose event ended strictly
    before `as_of` are used. Returns, aligned to `targets`:
      strength_sum    sum of weight * (-rel)
      weight_sum      sum of weight, the effective number of rounds behind the rating
      days_since_last days from the player's last earlier event to `as_of` (NaN if none)
    """
    out = pd.DataFrame(
        {"strength_sum": 0.0, "weight_sum": 0.0, "days_since_last": np.nan}, index=targets.index
    )
    as_of = pd.to_datetime(targets["as_of"]).to_numpy("datetime64[D]")
    round_day = pd.to_datetime(history["round_date"]).to_numpy("datetime64[D]")
    end_day = pd.to_datetime(history["event_end_date"]).to_numpy("datetime64[D]")
    origin = min(round_day.min(), as_of.min())
    by_player = {pid: idx for pid, idx in history.groupby("player_id").indices.items()}

    for pid, target_rows in targets.groupby("player_id").indices.items():
        if pid not in by_player:
            continue
        rows = by_player[pid]
        order = np.argsort(end_day[rows], kind="stable")
        rows = rows[order]
        ends = end_day[rows]
        # weight = 2 ** (-(as_of - round_day) / half_life), split so it can be cumulated once
        grow = 2.0 ** ((round_day[rows] - origin) / DAY / half_life_days)
        cum_weight = np.cumsum(grow)
        cum_strength = np.cumsum(grow * -history["rel"].to_numpy()[rows])

        when = as_of[target_rows]
        count = np.searchsorted(ends, when, side="left")  # rounds from events ended before as_of
        has = count > 0
        decay = 2.0 ** (-((when - origin) / DAY) / half_life_days)
        last = np.where(has, count - 1, 0)
        positions = targets.index[target_rows]
        out.loc[positions, "weight_sum"] = np.where(has, cum_weight[last] * decay, 0.0)
        out.loc[positions, "strength_sum"] = np.where(has, cum_strength[last] * decay, 0.0)
        out.loc[positions, "days_since_last"] = np.where(has, (when - ends[last]) / DAY, np.nan)
    return out


def shrunk_rating(strength_sum: pd.Series, weight_sum: pd.Series, shrinkage_rounds: float):
    """Weighted mean strokes better than the field, pulled toward 0 by pseudo-rounds."""
    return strength_sum / (weight_sum + shrinkage_rounds)


def build_features(
    events: pd.DataFrame,
    rounds: pd.DataFrame,
    long_half_life_days: float,
    short_half_life_days: float,
    shrinkage_rounds: float,
) -> pd.DataFrame:
    """Model 1 features for every (event, player) row of the tidy `events` table.

    `as_of` is the event's start date. Adds the rating, its recent-form deviation, the rounds
    behind it, rest, and the field-relative features. No outcome of the event is used.
    """
    history = round_relative_scores(rounds, events)
    targets = events[["player_id", "start_date"]].rename(columns={"start_date": "as_of"})
    long = decayed_history(history, targets, long_half_life_days)
    short = decayed_history(history, targets, short_half_life_days)

    features = events.copy()
    features["rating"] = shrunk_rating(long["strength_sum"], long["weight_sum"], shrinkage_rounds)
    short_rating = shrunk_rating(short["strength_sum"], short["weight_sum"], shrinkage_rounds)
    features["form"] = short_rating - features["rating"]
    features["log_rounds"] = np.log1p(long["weight_sum"])
    features["no_history"] = (long["weight_sum"] == 0).astype(float)
    features["days_since_last"] = long["days_since_last"].fillna(365).clip(upper=365)

    by_event = features.groupby("tournament_id")["rating"]
    features["field_mean_rating"] = by_event.transform("mean")
    features["rating_rel"] = features["rating"] - features["field_mean_rating"]
    features["rank_pct"] = by_event.rank(ascending=False, method="average") / by_event.transform(
        "size"
    )
    features["field_size_log"] = np.log(features["field_size"])
    features["amateur"] = features["amateur"].astype(float)
    return features
