import numpy as np
import pytest

from src.backtest import odds


@pytest.mark.parametrize(
    ("american", "expected"),
    [
        (150, 0.40),  # underdog: 100 / (150 + 100)
        (-150, 0.60),  # favourite: 150 / (150 + 100)
        (100, 0.50),  # even money, either sign convention
        (-100, 0.50),
        (900, 0.10),  # Scheffler's +900 preseason line, roughly a 1-in-10 shot before the vig
    ],
)
def test_implied_probability_known_values(american, expected):
    assert odds.implied_probability(american) == pytest.approx(expected, abs=1e-6)


def test_implied_probability_is_vectorised():
    result = odds.implied_probability([150, -150, 100])
    assert result == pytest.approx([0.40, 0.60, 0.50])


def test_implied_probability_favourite_beats_underdog():
    favourite, underdog = odds.implied_probability([-200, 300])
    assert favourite > underdog


def test_overround_is_zero_for_a_fair_market():
    assert odds.overround([0.5, 0.3, 0.2]) == pytest.approx(0.0, abs=1e-9)


def test_overround_is_positive_for_a_real_book():
    # Every player quoted a little worse than fair, so the raw probabilities sum above 1.
    assert odds.overround([0.55, 0.35, 0.2]) == pytest.approx(0.10, abs=1e-9)


def test_remove_vig_sums_to_one():
    raw = odds.implied_probability([150, 150, -300, 500, 2000])
    fair = odds.remove_vig(raw)
    assert fair.sum() == pytest.approx(1.0)


def test_remove_vig_preserves_relative_order_and_ratios():
    raw = np.array([0.5, 0.3, 0.2]) * 1.1  # a 10% margin market
    fair = odds.remove_vig(raw)
    assert list(fair) == sorted(fair, reverse=True)
    assert fair[0] / fair[1] == pytest.approx(raw[0] / raw[1])  # proportional scaling


def test_fair_win_probabilities_sums_to_one_and_matches_the_two_step_version():
    american = [900, 2500, 3500, 12500, 50000, 50000]

    combined = odds.fair_win_probabilities(american)
    two_step = odds.remove_vig(odds.implied_probability(american))

    assert combined.sum() == pytest.approx(1.0)
    assert combined == pytest.approx(two_step)
