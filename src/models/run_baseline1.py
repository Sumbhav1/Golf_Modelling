"""Baseline 1 on the validation seasons: walk-forward fit, score against the naive baseline, report.

Run with `uv run python -m src.models.run_baseline1`. The holdout and forward-test seasons are
removed from the data before anything is fitted, tuned or scored.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import assert_no_holdout, load_config
from src.models.baseline import fit_baseline
from src.models.evaluation import (
    calibration_table,
    log_loss_terms,
    plot_calibration,
    score_forecasts,
    walk_forward_folds,
)

LABELS = ["made_cut", "top10", "win"]
LABEL_NAMES = {"made_cut": "Make the cut", "top10": "Top 10", "win": "Win"}


def load_events(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, dtype=dict.fromkeys(LABELS, "boolean"))


def window_column(feature: str, window: int) -> str:
    return f"{feature}_rolling_{window}"


def walk_forward_predictions(
    events: pd.DataFrame, label: str, column: str, folds: list[tuple[list[int], int]]
) -> pd.DataFrame:
    """Predictions for each validation season from a model fitted on earlier seasons only."""
    parts = []
    for train_seasons, test_season in folds:
        train = events[events["season"].isin(train_seasons) & events[label].notna()]
        test = events[(events["season"] == test_season) & events[label].notna()]
        model = fit_baseline(train, label, column)
        part = test[["season", "tournament_id", "player_id"]].copy()
        part["event"] = test["tournament_id"].to_numpy()
        part["y"] = test[label].astype(int).to_numpy()
        part["p"] = model.predict(test)
        part["p_naive"] = train[label].astype(float).mean()
        part["has_history"] = test[column].notna().to_numpy()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def select_seasons(events: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list]:
    """Keep only the seasons the walk-forward folds use, and refuse reserved seasons.

    The holdout and forward-test seasons are dropped here, before anything is fitted or scored.
    """
    splits = config["splits"]
    folds = walk_forward_folds(
        events["season"].unique(), splits["validation_seasons"], splits["first_train_season"]
    )
    used = {season for train, test in folds for season in [*train, test]}
    assert_no_holdout(config, used)
    return events[events["season"].isin(used)], folds


def pooled_log_loss(predictions: pd.DataFrame) -> float:
    return float(log_loss_terms(predictions["y"], predictions["p"]).mean())


def score_slices(predictions: pd.DataFrame, samples: int, seed: int) -> dict[str, dict[str, float]]:
    """Metrics for the pooled validation seasons, each season, and players with history only."""
    slices = {"pooled": predictions}
    for season in sorted(predictions["season"].unique()):
        slices[f"season {season}"] = predictions[predictions["season"] == season]
    slices["pooled, players with history"] = predictions[predictions["has_history"]]
    return {name: score_forecasts(frame, samples, seed) for name, frame in slices.items()}


def _interval(value: float, lower: float, upper: float, digits: int = 4) -> str:
    return f"{value:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"


def metrics_table(rows: list[dict]) -> str:
    header = (
        "| Label | Slice | Rows | Events | Log loss (95% CI) | Naive log loss "
        "| Diff vs naive (95% CI) | Brier (95% CI) | Naive Brier | Diff vs naive (95% CI) |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['slice']} | {int(r['rows'])} | {int(r['events'])} "
            f"| {_interval(r['log_loss'], r['log_loss_lo'], r['log_loss_hi'])} "
            f"| {r['log_loss_naive']:.4f} "
            f"| {_interval(r['log_loss_delta'], r['log_loss_delta_lo'], r['log_loss_delta_hi'])} "
            f"| {_interval(r['brier'], r['brier_lo'], r['brier_hi'], 5)} | {r['brier_naive']:.5f} "
            f"| {_interval(r['brier_delta'], r['brier_delta_lo'], r['brier_delta_hi'], 5)} |"
        )
    return header + "\n".join(lines)


def build_report(
    config: dict, folds: list, window_scores: dict, chosen: dict, rows: list[dict]
) -> str:
    validation = config["splits"]["validation_seasons"]
    windows = config["baseline1"]["windows"]
    window_lines = (
        "| Label | " + " | ".join(f"{w}-season window" for w in windows) + " | Chosen |\n"
    )
    window_lines += "|---|" + "---|" * (len(windows) + 1) + "\n"
    for label in LABELS:
        cells = " | ".join(f"{window_scores[(label, w)]:.4f}" for w in windows)
        window_lines += f"| {LABEL_NAMES[label]} | {cells} | {chosen[label]} |\n"
    fold_lines = "\n".join(f"- validate {test}: fit on seasons {train}" for train, test in folds)
    main_rows = [r for r in rows if r["slice"] == "pooled"]
    return f"""# Baseline 1: validation report

Generated by `uv run python -m src.models.run_baseline1` from `config/config.yaml`.
Validation seasons {validation}. The holdout ({config["splits"]["holdout_seasons"]}) and the
forward-test ({config["splits"]["forward_test_seasons"]}) seasons were **not** loaded or looked at.

Baseline 1 turns a player's rolling strokes gained (from earlier seasons only) into a probability:
logistic regression for make-cut and top 10, a softmax over each event's field for win. Players with
no strokes-gained history get their own effect. The naive baseline gives every player the training
base rate. Differences are model minus naive, so **negative means better than naive**. Intervals
are 95% bootstrap intervals over events ({config["evaluation"]["bootstrap_samples"]} resamples,
seed {config["evaluation"]["seed"]}), not rows.

## Walk-forward folds

{fold_lines}

## Window choice (pooled validation log loss, lower is better)

{window_lines}
Chosen per label on the validation seasons only.

## Results

{metrics_table(main_rows)}

Calibration: `baseline1_calibration.png` (pooled validation predictions, equal-frequency bins).

## By season and for players with history

{metrics_table([r for r in rows if r["slice"] != "pooled"])}
"""


def main() -> None:
    config = load_config()
    settings = config["baseline1"]
    evaluation = config["evaluation"]
    splits = config["splits"]

    events, folds = select_seasons(load_events(config["data"]["events_history"]), config)

    predictions, window_scores, chosen = {}, {}, {}
    for label in settings["labels"]:
        for window in settings["windows"]:
            column = window_column(settings["feature"], window)
            predictions[(label, window)] = walk_forward_predictions(events, label, column, folds)
            window_scores[(label, window)] = pooled_log_loss(predictions[(label, window)])
        chosen[label] = min(settings["windows"], key=lambda w: window_scores[(label, w)])

    rows, tables = [], {}
    for label in settings["labels"]:
        best = predictions[(label, chosen[label])]
        for name, scores in score_slices(
            best, evaluation["bootstrap_samples"], evaluation["seed"]
        ).items():
            rows.append({"label": LABEL_NAMES[label], "slice": name, **scores})
        tables[LABEL_NAMES[label]] = {
            "Baseline 1": calibration_table(best["y"], best["p"], evaluation["calibration_bins"])
        }

    report_dir = Path(evaluation["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "baseline1_validation.md").write_text(
        build_report(
            config, folds, window_scores, {k: f"{v}-season" for k, v in chosen.items()}, rows
        )
    )
    pd.DataFrame(rows).to_csv(report_dir / "baseline1_validation_metrics.csv", index=False)
    plot_calibration(
        tables, report_dir / "baseline1_calibration.png",
        f"Baseline 1 calibration, validation seasons {splits['validation_seasons']}",
    )  # fmt: skip
    print((report_dir / "baseline1_validation.md").read_text())


if __name__ == "__main__":
    main()
