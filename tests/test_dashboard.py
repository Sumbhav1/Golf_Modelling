from pathlib import Path

import pandas as pd

from src.dashboard.app import (
    REPORT_DIR,
    REPORTS,
    _best_model_column,
    _event_insight,
    _parse_odds,
    _player_table,
    _verdict,
)

FRAME = pd.DataFrame(
    [
        {
            "player_id": 1,
            "display_name": "Player One",
            "label": "win",
            "model": "Model 1",
            "p": 0.3,
            "y": 1,
        },
        {
            "player_id": 1,
            "display_name": "Player One",
            "label": "win",
            "model": "Baseline 1",
            "p": 0.2,
            "y": 1,
        },
        {
            "player_id": 2,
            "display_name": "Player Two",
            "label": "win",
            "model": "Model 1",
            "p": 0.1,
            "y": 0,
        },
        {
            "player_id": 2,
            "display_name": "Player Two",
            "label": "win",
            "model": "Baseline 1",
            "p": 0.15,
            "y": 0,
        },
        {
            "player_id": 1,
            "display_name": "Player One",
            "label": "made_cut",
            "model": "Model 1",
            "p": 0.9,
            "y": 1,
        },
    ]
)


def test_player_table_sorts_by_model1_descending_and_keeps_the_actual_outcome():
    table = _player_table(FRAME, "win")

    assert list(table["display_name"]) == ["Player One", "Player Two"]
    assert list(table["actual"]) == [1, 0]
    assert set(table.columns) == {"display_name", "Model 1", "Baseline 1", "actual"}


def test_player_table_is_empty_for_a_label_with_no_rows():
    table = _player_table(FRAME, "top10")

    assert table.empty


def test_verdict_reads_the_interval_not_just_the_point_estimate():
    assert _verdict(-0.05, -0.01) == "clearly better"  # whole interval negative
    assert _verdict(0.01, 0.05) == "clearly worse"  # whole interval positive
    assert _verdict(-0.02, 0.03) == "not clearly different - could be noise"  # crosses zero
    assert _verdict(float("nan"), 0.03) == ""  # no comparison available (e.g. no naive baseline)


def test_best_model_column_prefers_model1_over_the_others():
    assert _best_model_column(pd.DataFrame(columns=["Baseline 1", "Model 1", "Naive"])) == "Model 1"
    assert _best_model_column(pd.DataFrame(columns=["Naive"])) == "Naive"
    assert _best_model_column(pd.DataFrame(columns=["actual"])) is None


WIN_TABLE = pd.DataFrame(
    [
        {"display_name": "Player One", "Model 1": 0.4, "actual": 1},
        {"display_name": "Player Two", "Model 1": 0.6, "actual": 0},  # ranked above the winner
        {"display_name": "Player Three", "Model 1": 0.1, "actual": 0},
    ]
)


def test_event_insight_for_win_names_the_winner_and_their_rank():
    insight = _event_insight(WIN_TABLE, "win")

    assert "Player One" in insight
    assert "40.0%" in insight or "40%" in insight
    assert "#2 of 3" in insight  # Player Two's higher probability puts the winner second


def test_event_insight_for_top10_compares_medians_of_the_two_groups():
    table = pd.DataFrame(
        [
            {"display_name": "A", "Model 1": 0.8, "actual": 1},
            {"display_name": "B", "Model 1": 0.6, "actual": 1},
            {"display_name": "C", "Model 1": 0.1, "actual": 0},
            {"display_name": "D", "Model 1": 0.05, "actual": 0},
        ]
    )

    insight = _event_insight(table, "top10")

    assert "70%" in insight  # median of 0.8, 0.6
    assert "8%" in insight  # median of 0.1, 0.05 (rounded)
    assert "separated them well" in insight


def test_event_insight_says_plainly_when_nobody_achieved_the_outcome():
    table = pd.DataFrame(
        [
            {"display_name": "A", "Model 1": 0.2, "actual": 0},
            {"display_name": "B", "Model 1": 0.1, "actual": 0},
        ]
    )

    assert "Nobody" in _event_insight(table, "top10")


def test_every_report_points_at_metrics_and_calibration_files_that_actually_exist():
    """Regression guard: an earlier version's metrics stem didn't match the real report filenames
    (e.g. "holdout_2025_results_metrics.csv" instead of the real "holdout_2025_metrics.csv"), and
    nothing caught it because the dashboard was only smoke-tested on the first selectbox option."""
    for _choice, (stem, image, _fallback) in REPORTS.items():
        assert (REPORT_DIR / f"{stem}_metrics.csv").exists(), stem
        assert (REPORT_DIR / image).exists(), image


def test_report_dir_is_the_real_reports_directory():
    assert REPORT_DIR == Path("reports")


def test_parse_odds_reads_american_odds_with_or_without_a_leading_plus():
    assert _parse_odds("-150, +200, +500") == [-150.0, 200.0, 500.0]
    assert _parse_odds("-150,200,500") == [-150.0, 200.0, 500.0]  # a bare "+" is optional


def test_parse_odds_is_none_for_unparseable_or_empty_input():
    assert _parse_odds("not odds") is None
    assert _parse_odds("") is None
    assert _parse_odds("   ") is None
    assert _parse_odds("-150, banana, +500") is None  # one bad entry spoils the whole field
