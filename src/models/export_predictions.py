"""Builds one combined per-player prediction table for the API and dashboard (Phase 4).

Run with `uv run python -m src.models.export_predictions`. Reuses exactly the settings and
refitting each report already uses (validation: `run_model1`; holdout: `run_holdout`, the frozen
one-look; forward test: `run_forward_test`, re-run as the season progresses) - this recomputes the
same numbers those reports show rather than taking any new look at anything frozen. Output is
written to `data/processed/` (gitignored, regenerable from source data + config, like every other
file under `data/`), not `reports/` (which holds the public per-model write-ups, not a bulk
per-player export). Regenerate whenever the input data, a model setting, or a 2026 result changes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config
from src.features.ratings import build_features
from src.models import run_baseline1 as baseline_run
from src.models import run_forward_test as forward_run
from src.models import run_holdout as holdout_run
from src.models import run_model1 as model1_run
from src.models.model1 import FULL_COLUMNS, WIN_COLUMNS
from src.models.model1 import walk_forward_predictions as model1_predictions
from src.models.reporting import align_has_history
from src.sim.run_simulator import fold_sigmas, simulate_predictions

LABELS = baseline_run.LABELS
EVENTS_PATH = Path("data/processed/served_events.csv")
PREDICTIONS_PATH = Path("data/processed/served_predictions.csv")


def _long_predictions(
    frame: pd.DataFrame, label: str, model: str, era: str, include_naive: bool
) -> list[pd.DataFrame]:
    """One tidy frame of (era, season, tournament_id, player_id, label, model, p, y), plus a
    matching "Naive" frame from the same rows' `p_naive` column, when asked for."""
    base = frame[["season", "tournament_id", "player_id", "y"]].copy()
    base["label"] = label
    parts = [base.assign(model=model, p=frame["p"].to_numpy())]
    if include_naive:
        parts.append(base.assign(model="Naive", p=frame["p_naive"].to_numpy()))
    for part in parts:
        part["era"] = era
    return parts


def _events_table(features: pd.DataFrame, season, era: str) -> pd.DataFrame:
    in_season = features[features["season"] == season]
    columns = ["season", "tournament_id", "tournament_name", "start_date", "end_date", "field_size"]
    events = in_season[columns].drop_duplicates("tournament_id").copy()
    events["era"] = era
    return events


def _names(features: pd.DataFrame) -> pd.DataFrame:
    return features[["tournament_id", "player_id", "display_name"]].drop_duplicates(
        ["tournament_id", "player_id"]
    )


def validation_bundle(config: dict) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    events, rounds, folds = model1_run.load_inputs(config)
    half_life, shrink, _ = model1_run.tune_rating(
        events, rounds, folds, config["model1"], config["evaluation"]["seed"]
    )
    features = model1_run.features_for(
        events, rounds, config["model1"]["rating"], half_life, shrink
    )
    baseline = model1_run.baseline_predictions(features, folds, config)
    sigmas = fold_sigmas(features, rounds, folds)
    simulated = simulate_predictions(features, folds, sigmas, config["sim"], "fraction", 42)

    predictions, seasons = [], sorted({test for _, test in folds})
    for label in LABELS:
        kind, columns = ("softmax", WIN_COLUMNS) if label == "win" else ("logistic", FULL_COLUMNS)
        model = model1_predictions(
            features, label, kind, columns, folds, config["model1"], config["evaluation"]["seed"]
        )
        predictions += _long_predictions(baseline[label], label, "Baseline 1", "validation", True)
        predictions += _long_predictions(model, label, "Model 1", "validation", False)
        predictions += _long_predictions(simulated[label], label, "Simulator", "validation", False)
    events_tables = [_events_table(features, season, "validation") for season in seasons]
    return predictions, events_tables, _names(features)  # type: ignore[return-value]


def holdout_bundle(config: dict) -> tuple[list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame]:
    half_life, shrink, windows = holdout_run.choose_settings(config)
    (holdout_season,) = config["splits"]["holdout_seasons"]
    train_seasons = list(range(config["splits"]["first_train_season"], holdout_season))
    fold = [(train_seasons, holdout_season)]

    events = baseline_run.load_events(config["data"]["events_history"])
    rounds = pd.read_csv(config["data"]["rounds_history"])
    features = build_features(
        events, rounds, half_life, config["model1"]["rating"]["short_half_life_days"], shrink
    )
    sigmas = fold_sigmas(features, rounds, fold)
    simulated = simulate_predictions(features, fold, sigmas, config["sim"], "fraction", 42)

    predictions = []
    for label in LABELS:
        column = baseline_run.window_column(config["baseline1"]["feature"], windows[label])
        baseline = align_has_history(
            baseline_run.walk_forward_predictions(events, label, column, fold), features
        )
        kind, columns = ("softmax", WIN_COLUMNS) if label == "win" else ("logistic", FULL_COLUMNS)
        model = model1_predictions(
            features, label, kind, columns, fold, config["model1"], config["evaluation"]["seed"]
        )
        predictions += _long_predictions(baseline, label, "Baseline 1", "holdout", True)
        predictions += _long_predictions(model, label, "Model 1", "holdout", False)
        predictions += _long_predictions(simulated[label], label, "Simulator", "holdout", False)
    events_tables = [_events_table(features, holdout_season, "holdout")]
    return predictions, events_tables, _names(features)


def forward_test_bundle(
    config: dict,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame]:
    half_life, shrink, windows = holdout_run.choose_settings(config)
    events, rounds, fold = forward_run.load_forward_inputs(config)
    (forward_season,) = config["splits"]["forward_test_seasons"]
    features = build_features(
        events, rounds, half_life, config["model1"]["rating"]["short_half_life_days"], shrink
    )
    sigmas = fold_sigmas(features, rounds, fold)
    simulated = simulate_predictions(features, fold, sigmas, config["sim"], "fraction", 42)

    predictions = []
    for label in LABELS:
        column = baseline_run.window_column(config["baseline1"]["feature"], windows[label])
        baseline = align_has_history(
            baseline_run.walk_forward_predictions(events, label, column, fold), features
        )
        kind, columns = ("softmax", WIN_COLUMNS) if label == "win" else ("logistic", FULL_COLUMNS)
        model = model1_predictions(
            features, label, kind, columns, fold, config["model1"], config["evaluation"]["seed"]
        )
        predictions += _long_predictions(baseline, label, "Baseline 1", "forward_test", True)
        predictions += _long_predictions(model, label, "Model 1", "forward_test", False)
        predictions += _long_predictions(
            simulated[label], label, "Simulator", "forward_test", False
        )
    events_tables = [_events_table(features, forward_season, "forward_test")]
    return predictions, events_tables, _names(features)


def main() -> None:
    config = load_config()

    v_predictions, v_events, v_names = validation_bundle(config)
    h_predictions, h_events, h_names = holdout_bundle(config)
    f_predictions, f_events, f_names = forward_test_bundle(config)

    predictions = pd.concat(v_predictions + h_predictions + f_predictions, ignore_index=True)
    events = pd.concat(v_events + h_events + f_events, ignore_index=True)
    names = pd.concat([v_names, h_names, f_names], ignore_index=True).drop_duplicates(
        ["tournament_id", "player_id"]
    )
    predictions = predictions.merge(names, on=["tournament_id", "player_id"], how="left")

    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    events.sort_values(["season", "tournament_id"]).to_csv(EVENTS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    print(
        f"Wrote {len(events)} events and {len(predictions)} prediction rows "
        f"({predictions['era'].value_counts().to_dict()}) to {EVENTS_PATH} and {PREDICTIONS_PATH}."
    )


if __name__ == "__main__":
    main()
