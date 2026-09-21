import pandas as pd
import pytest

import scraper

SG_COLUMNS = [
    "sg_total",
    "sg_off_the_tee",
    "sg_approach",
    "sg_around_the_green",
    "sg_putting",
]


def _history(values_by_year: dict[int, float]) -> pd.DataFrame:
    """One player, season-level SG, every category set to the same value."""
    rows = [{"player_id": 1, "player_name": "A", "year": y, **dict.fromkeys(SG_COLUMNS, v)}
            for y, v in values_by_year.items()]  # fmt: skip
    return pd.DataFrame(rows)


def _feature(features: pd.DataFrame, year: int, name: str = "sg_total_rolling_3") -> float:
    return features.loc[features["year"] == year, name].iloc[0]


def test_first_season_has_no_rolling_history():
    features = scraper.build_rolling_sg_features(_history({2021: 1.0, 2022: 2.0}))
    assert pd.isna(_feature(features, 2021))


def test_rolling_uses_only_earlier_seasons():
    features = scraper.build_rolling_sg_features(_history({2021: 1.0, 2022: 3.0, 2023: 5.0}))
    assert _feature(features, 2022) == pytest.approx(1.0)
    assert _feature(features, 2023) == pytest.approx(2.0)  # mean of 2021 and 2022 only


@pytest.mark.parametrize("changed_year", [2021, 2022, 2023])
def test_changing_a_season_never_changes_its_own_or_earlier_features(changed_year):
    """Leakage: a season's SG must not reach features of that season or before."""
    base = {2021: 1.0, 2022: 3.0, 2023: 5.0, 2024: 7.0}
    changed = {**base, changed_year: 99.0}

    before = scraper.build_rolling_sg_features(_history(base))
    after = scraper.build_rolling_sg_features(_history(changed))

    for year in base:
        if year <= changed_year:
            for column in [c for c in before.columns if "rolling" in c]:
                assert _feature(before, year, column) == pytest.approx(
                    _feature(after, year, column), nan_ok=True
                )


def test_player_features_for_a_new_season_use_only_history():
    history = _history({2021: 1.0, 2022: 3.0})
    features = scraper.build_player_features_for_season(history, 2023)

    assert list(features["year"]) == [2023]
    assert _feature(features, 2023) == pytest.approx(2.0)
    assert features["sg_total"].isna().all()  # the new season's own SG is never present


def test_player_features_refuse_history_that_overlaps_the_season():
    """A partial season must never feed its own predictions."""
    with pytest.raises(ValueError, match="before season 2023"):
        scraper.build_player_features_for_season(_history({2021: 1.0, 2023: 3.0}), 2023)
