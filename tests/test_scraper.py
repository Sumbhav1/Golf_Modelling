import time

import numpy as np
import pandas as pd
import pytest

import scraper


@pytest.mark.parametrize(
    ("calendar_year", "display_date", "expected"),
    [
        (2020, "Sep 10 - 13", ("2020-09-10", "2020-09-13")),
        (2021, "Apr 29 - May 2", ("2021-04-29", "2021-05-02")),
        (2023, "Sep 29 - Oct 1", ("2023-09-29", "2023-10-01")),
        (2020, "Dec 30 - Jan 2", ("2020-12-30", "2021-01-02")),
    ],
)
def test_parse_event_dates(calendar_year, display_date, expected):
    assert scraper.parse_event_dates(calendar_year, display_date) == expected


@pytest.mark.parametrize("bad", ["", "TBD", "Sep 10", "Foo 10 - 13", None])
def test_parse_event_dates_bad_text_returns_none(bad):
    assert scraper.parse_event_dates(2021, bad) == (None, None)


def test_extract_year_from_tournament_id():
    assert scraper.extract_year_from_tournament_id("R2025464") == 2025
    assert scraper.extract_year_from_tournament_id("R20") is None
    assert scraper.extract_year_from_tournament_id(None) is None


def test_extract_position():
    assert scraper.extract_position("T13") == 13
    assert scraper.extract_position("1") == 1
    assert np.isnan(scraper.extract_position("CUT"))
    assert np.isnan(scraper.extract_position("W/D"))


def _results(tournament_id: str, totals: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tournament_id": tournament_id,
            "player_id": range(len(totals)),
            "total": totals,
            "round_1": 70.0,
        }
    )


def test_validate_results_are_distinct_accepts_different_events():
    first = _results("R2021003", ["265", "270"])
    second = _results("R2022003", ["268", "270"])
    results = pd.concat([first, second])
    scraper.validate_results_are_distinct(results)


def test_validate_results_are_distinct_catches_mislabelled_copies():
    """The old bug: one edition returned under several tournament IDs."""
    first = _results("R2021003", ["265", "270"])
    copy = _results("R2022003", ["265", "270"])
    results = pd.concat([first, copy])
    with pytest.raises(RuntimeError, match="mislabelled"):
        scraper.validate_results_are_distinct(results)


def test_results_have_scores():
    scored = pd.DataFrame({"round_1": [70.0, np.nan], "round_2": [68.0, np.nan]})
    unscored = pd.DataFrame({"round_1": [np.nan, np.nan], "round_2": [np.nan, np.nan]})

    assert scraper.results_have_scores(scored)
    assert not scraper.results_have_scores(unscored)  # the 2018 Ryder Cup case
    assert not scraper.results_have_scores(pd.DataFrame({"position": ["1"]}))  # no round columns

    dashes = pd.DataFrame({"round_1": ["-", "-"], "round_2": ["-", "-"]})
    assert not scraper.results_have_scores(dashes)  # placeholders are not scores


def test_call_with_deadline_returns_the_value_and_passes_arguments():
    assert scraper.call_with_deadline(lambda a, b=0: a + b, 2, b=3) == 5


def test_call_with_deadline_reraises_errors():
    def fail():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        scraper.call_with_deadline(fail)


def test_call_with_deadline_gives_up_on_a_stalled_call():
    with pytest.raises(TimeoutError):
        scraper.call_with_deadline(time.sleep, 2, seconds=0.1)
