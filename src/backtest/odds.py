"""Bookmaker odds: American-odds conversion and vig removal.

Source-agnostic: works with American odds from any win market, historical or live. Nothing here
reads a specific data source, so it stays public and reusable regardless of where the odds come
from or what that source's licence allows.
"""

from __future__ import annotations

import numpy as np


def implied_probability(american_odds) -> np.ndarray:
    """Raw (vig-included) implied probability from American odds.

    +150 (an underdog) -> 100 / (150 + 100) = 0.40
    -150 (a favourite)  -> 150 / (150 + 100) = 0.60
    """
    odds = np.asarray(american_odds, dtype=float)
    # np.where evaluates both branches before picking one, and at the +100/-100 boundary the
    # branch that gets discarded divides by exactly zero (100 + -100). Harmless (its result is
    # never used), but silence the warning rather than leave it in normal output.
    with np.errstate(divide="ignore"):
        positive = 100.0 / (odds + 100.0)
        negative = -odds / (-odds + 100.0)
    return np.where(odds >= 0, positive, negative)


def overround(probabilities) -> float:
    """The bookmaker's margin: how far the raw implied probabilities sum above 1.0.

    A win market's probabilities should sum to exactly 1 (someone always wins); real books quote
    every player slightly worse than fair to guarantee a profit, so they sum to a bit over 1.
    """
    return float(np.asarray(probabilities, dtype=float).sum() - 1.0)


def remove_vig(probabilities) -> np.ndarray:
    """Fair (de-vigged) probabilities: proportional devigging.

    Divides each raw implied probability by the field's total, so they sum to exactly 1. This
    assumes the book inflates every probability by the same multiplicative factor - the simplest
    and most common devigging method. A documented alternative, "power" devigging (raising each
    probability to a shared exponent instead of scaling), better accounts for longshot bias
    (books usually shade long shots more than favourites so total profit doesn't depend on who
    wins), but needs fitting an extra parameter; not used here, kept simple and transparent for a
    first pass. Every probability given must belong to the same market (the same event's win
    field) - do not mix markets or call this on a single probability.
    """
    p = np.asarray(probabilities, dtype=float)
    return p / p.sum()


def fair_win_probabilities(american_odds) -> np.ndarray:
    """Fair, de-vigged win probabilities for one event's full field, from American odds."""
    return remove_vig(implied_probability(american_odds))
