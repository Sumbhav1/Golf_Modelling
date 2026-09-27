"""Monte Carlo tournament simulator: player skill (the Model 1 rating) plus round-to-round noise.

A round score, relative to an average tour field, is modelled as `-rating + noise`, where `rating`
is the as-of field-adjusted rating from `src.features.ratings` (in strokes per round better than
an average field) and `noise` is drawn independently each round from a Normal(0, sigma). Four
rounds are simulated, with a cut applied after round 2 for events that have one. Nothing here uses
information from after the event's start (the rating is as-of; sigma is estimated on rounds from
earlier seasons than the one being simulated).

Limitations (v1): noise is independent across rounds and identical across players (no player- or
course-specific variance, no 54-hole cuts); the cut boundary counts ties as "made the cut", matching
the real rule.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from src.features.ratings import round_relative_scores


def event_rng(seed: int, tournament_id: str) -> np.random.Generator:
    """A generator seeded from `seed` and the event's own ID.

    Each event gets an independent random stream, so simulating one event's numbers never
    changes because another event was added to or removed from the run.
    """
    return np.random.default_rng(zlib.crc32(f"{seed}:{tournament_id}".encode()))


def estimate_residual_sigma(features: pd.DataFrame, rounds: pd.DataFrame) -> float:
    """Standard deviation of a round's field-relative score around what the rating predicts.

    Only rounds from players with rating history are used (a `rating` of exactly 0 for a
    no-history player is not a skill estimate, so including it would bias the spread).
    """
    history = round_relative_scores(rounds, features)
    with_rating = features.loc[
        features["no_history"] == 0, ["tournament_id", "player_id", "rating"]
    ]
    merged = history.merge(with_rating, on=["tournament_id", "player_id"], how="inner")
    residual = merged["rel"] + merged["rating"]  # predicted rel is -rating
    return float(residual.std(ddof=1))


def cut_survivor_count(
    field_size: int, has_cut: bool, made_cut: pd.Series | None, mode: str, fallback_fraction: float
) -> int | None:
    """How many players are treated as making the cut, or None for a no-cut event.

    `mode="actual"` replays the real number of players who made the cut (for validating the
    simulator against known events). `mode="fraction"` approximates it as a fixed share of the
    field, for events whose outcome is not yet known.
    """
    if not has_cut:
        return None
    if mode == "actual":
        return int(made_cut.sum())
    if mode == "fraction":
        return max(1, round(field_size * fallback_fraction))
    raise ValueError(f"Unknown cut mode: {mode!r}")


def simulate_event(
    rating: np.ndarray, sigma: float, n_sims: int, rng: np.random.Generator, cut_keep: int | None
) -> dict[str, np.ndarray]:
    """Win, top-5, top-10 and made-cut probabilities for one field of players.

    `rating` is one value per player (strokes per round better than an average field; higher is
    better). Probabilities are the share of simulations in which each event happens.
    """
    n_players = len(rating)
    mean = -rating  # predicted round score, relative to an average field

    rounds_1_2 = rng.normal(mean, sigma, size=(n_sims, n_players)) + rng.normal(
        mean, sigma, size=(n_sims, n_players)
    )

    if cut_keep is not None and cut_keep <= 0:
        made_cut = np.zeros_like(rounds_1_2, dtype=bool)  # nobody survives (a degenerate field)
    elif cut_keep is not None and cut_keep < n_players:
        cut_line = np.partition(rounds_1_2, cut_keep - 1, axis=1)[:, cut_keep - 1 : cut_keep]
        made_cut = rounds_1_2 <= cut_line
    else:
        made_cut = np.ones_like(rounds_1_2, dtype=bool)

    rounds_3_4 = rng.normal(mean, sigma, size=(n_sims, n_players)) + rng.normal(
        mean, sigma, size=(n_sims, n_players)
    )
    total = rounds_1_2 + rounds_3_4
    for_ranking = np.where(made_cut, total, np.inf)  # eliminated players cannot win or place

    rank = for_ranking.argsort(axis=1, kind="stable").argsort(axis=1, kind="stable") + 1
    any_survivor = made_cut.any(axis=1, keepdims=True)  # False in the degenerate all-cut field

    return {
        "win": ((rank == 1) & any_survivor).mean(axis=0),
        "top5": ((rank <= 5) & any_survivor).mean(axis=0),
        "top10": ((rank <= 10) & any_survivor).mean(axis=0),
        "made_cut": made_cut.mean(axis=0),
    }


def simulate_events(
    features: pd.DataFrame,
    sigma: float,
    n_sims: int,
    seed: int,
    cut_mode: str,
    fallback_cut_fraction: float,
) -> pd.DataFrame:
    """`simulate_event` for every standard event in `features`, one row per (event, player)."""
    rows = []
    standard = features[features["is_standard_event"]]
    for tournament_id, group in standard.groupby("tournament_id", sort=True):
        cut_keep = cut_survivor_count(
            len(group),
            bool(group["has_cut"].iloc[0]),
            group["made_cut"] if group["has_cut"].iloc[0] else None,
            cut_mode,
            fallback_cut_fraction,
        )
        rng = event_rng(seed, tournament_id)
        result = simulate_event(
            group["rating"].fillna(0.0).to_numpy(), sigma, n_sims, rng, cut_keep
        )
        rows.append(
            pd.DataFrame(
                {
                    "tournament_id": tournament_id,
                    "player_id": group["player_id"].to_numpy(),
                    **result,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
