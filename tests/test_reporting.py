import numpy as np
import pandas as pd
import pytest

from src.models import reporting


def _predictions(season, n_events=8, per_event=6, seed=0, has_history=True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_events * per_event
    p = rng.uniform(0.1, 0.5, size=n)
    tournament_id = [f"R{season}{i:03d}" for i in np.repeat(range(n_events), per_event)]
    return pd.DataFrame(
        {
            "season": season,
            "tournament_id": tournament_id,
            "event": tournament_id,
            "player_id": np.tile(range(per_event), n_events),
            "y": (rng.uniform(size=n) < p).astype(int),
            "p": p,
            "p_naive": p.mean(),
            "has_history": has_history,
        }
    )


def test_slices_of_covers_pooled_seasons_and_history():
    predictions = pd.concat([_predictions(2023), _predictions(2024)], ignore_index=True)
    predictions.loc[predictions.index[:5], "has_history"] = False

    slices = reporting.slices_of(predictions)

    assert set(slices) == {
        "pooled",
        "season 2023",
        "season 2024",
        "pooled, players with rating history",
    }
    assert len(slices["pooled"]) == len(predictions)
    assert len(slices["season 2023"]) + len(slices["season 2024"]) == len(predictions)
    assert len(slices["pooled, players with rating history"]) == len(predictions) - 5


def test_paired_scores_is_zero_against_itself_however_the_rows_are_ordered():
    frame = _predictions(2023, seed=1)
    shuffled_reference = frame.sample(frac=1, random_state=5)  # same rows, different order

    scores = reporting.paired_scores(frame, shuffled_reference, samples=200, seed=0)

    assert scores["log_loss_delta"] == pytest.approx(0, abs=1e-9)
    assert scores["brier_delta"] == pytest.approx(0, abs=1e-9)


def test_paired_scores_reflects_a_real_difference():
    frame = _predictions(2023, n_events=200, seed=1)
    worse_reference = frame.assign(p=0.5)  # always predicts 50%, ignoring the true rate

    scores = reporting.paired_scores(frame, worse_reference, samples=200, seed=0)

    assert scores["log_loss_delta"] < 0  # frame's real probabilities beat the constant 0.5
    assert scores["log_loss_delta_hi"] < 0


def test_score_model_includes_baseline_comparison_only_when_given_one():
    predictions = _predictions(2023)
    baseline = _predictions(2023, seed=9)

    alone = reporting.score_model("top10", "Model 1", predictions, None, samples=50, seed=0)
    versus = reporting.score_model("top10", "Model 1", predictions, baseline, samples=50, seed=0)

    assert not any("b1_log_loss_delta" in row for row in alone)
    assert all("b1_log_loss_delta" in row for row in versus)
    assert {row["slice"] for row in alone} == {
        "pooled",
        "season 2023",
        "pooled, players with rating history",
    }


def test_results_table_renders_a_row_per_model_and_dash_for_missing_baseline():
    rows = reporting.score_model(
        "top10", "Baseline 1", _predictions(2023), None, samples=50, seed=0
    )
    rows += reporting.score_model(
        "top10", "Model 1", _predictions(2023, seed=2), _predictions(2023), samples=50, seed=0
    )

    table = reporting.results_table(rows, "pooled")
    lines = table.splitlines()

    assert len(lines) == 4  # header, separator, one row per model
    baseline_row, model_row = lines[2], lines[3]
    assert "Baseline 1" in baseline_row and "| - |" in baseline_row  # no vs-Baseline1 column for it
    assert "Model 1" in model_row and "| - |" not in model_row  # Model 1 has one


def test_markdown_table_formats_floats_and_nan():
    frame = pd.DataFrame({"feature": ["a", "b"], "value": [1.23456, np.nan]})

    table = reporting.markdown_table(frame, digits=2)

    assert "| feature | value |" in table
    assert "1.23" in table
    assert "| b | - |" in table


def test_align_has_history_uses_the_rating_based_definition():
    """Baseline 1's own SG-based has_history must not silently differ from Model 1's population."""
    predictions = pd.DataFrame(
        {
            "tournament_id": ["E1", "E1", "E1"],
            "player_id": [1, 2, 3],
            "has_history": [True, True, False],  # Baseline 1's own (SG-based) flag
        }
    )
    features = pd.DataFrame(
        {
            "tournament_id": ["E1", "E1", "E1"],
            "player_id": [1, 2, 3],
            "no_history": [0, 1, 0],  # Model 1's (rating-based) flag disagrees with both above
        }
    )

    aligned = reporting.align_has_history(predictions, features)

    assert aligned.set_index("player_id")["has_history"].to_dict() == {1: True, 2: False, 3: True}
    assert "no_history" not in aligned.columns


def test_align_has_history_does_not_change_row_count_or_other_columns():
    predictions = pd.DataFrame(
        {
            "tournament_id": ["E1", "E1"],
            "player_id": [1, 2],
            "has_history": [True, True],
            "p": [0.4, 0.6],
        }
    )
    features = pd.DataFrame(
        {"tournament_id": ["E1", "E1"], "player_id": [1, 2], "no_history": [0, 0]}
    )

    aligned = reporting.align_has_history(predictions, features)

    assert len(aligned) == 2
    assert aligned["p"].tolist() == [0.4, 0.6]
