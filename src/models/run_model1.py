"""Model 1 on the validation seasons: walk-forward fit, scored against Baseline 1 and naive.

Run with `uv run python -m src.models.run_model1`. Only seasons up to the last validation season
are loaded, so the holdout and forward-test seasons are never read.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd

from src.config import assert_no_holdout, load_config
from src.features.ratings import build_features
from src.models import run_baseline1 as baseline_run
from src.models.evaluation import (
    calibration_table,
    log_loss_terms,
    plot_calibration,
    walk_forward_folds,
)
from src.models.model1 import (
    FIELD,
    FORM,
    FULL_COLUMNS,
    RATING,
    SG,
    WIN_COLUMNS,
    fit_model1,
    walk_forward_predictions,
)
from src.models.reporting import (
    LABEL_NAMES,
    align_has_history,
    markdown_table,
    results_table,
    score_model,
)

LABELS = baseline_run.LABELS
MODELS = {
    "made_cut": {
        "Model 1": ("logistic", FULL_COLUMNS),
        "Model 1 boosting": ("boosting", FULL_COLUMNS),
    },
    "top10": {
        "Model 1": ("logistic", FULL_COLUMNS),
        "Model 1 boosting": ("boosting", FULL_COLUMNS),
    },
    "win": {"Model 1": ("softmax", WIN_COLUMNS)},
}
ABLATION = {
    "rating only": RATING,
    "+ field features": RATING + FIELD,
    "+ form, rounds, rest": RATING + FIELD + FORM,
    "+ season SG categories (full model)": FULL_COLUMNS,
    "season SG categories only": SG,
}


def load_inputs(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Events and rounds up to the last validation season, and the walk-forward folds."""
    splits = config["splits"]
    events = baseline_run.load_events(config["data"]["events_history"])
    rounds = pd.read_csv(config["data"]["rounds_history"])
    folds = walk_forward_folds(
        events["season"].unique(), splits["validation_seasons"], splits["first_train_season"]
    )
    last = max(test for _, test in folds)
    assert_no_holdout(config, range(int(events["season"].min()), last + 1))
    return events[events["season"] <= last], rounds[rounds["season"] <= last], folds


def features_for(events: pd.DataFrame, rounds: pd.DataFrame, rating: dict, half_life, shrink):
    return build_features(events, rounds, half_life, rating["short_half_life_days"], shrink)


def tune_rating(events, rounds, folds, settings, seed) -> tuple[float, float, pd.DataFrame]:
    """Pick the long half-life and shrinkage by validation top-10 log loss (logistic model)."""
    rating = settings["rating"]
    rows = []
    for half_life in rating["long_half_life_days"]:
        for shrink in rating["shrinkage_rounds"]:
            features = features_for(events, rounds, rating, half_life, shrink)
            predictions = walk_forward_predictions(
                features, "top10", "logistic", FULL_COLUMNS, folds, settings, seed
            )
            loss = float(log_loss_terms(predictions["y"], predictions["p"]).mean())
            rows.append({"half_life_days": half_life, "shrinkage_rounds": shrink, "log_loss": loss})
    grid = pd.DataFrame(rows)
    best = grid.loc[grid["log_loss"].idxmin()]
    return float(best["half_life_days"]), float(best["shrinkage_rounds"]), grid


def chosen_baseline_windows(features: pd.DataFrame, folds: list, config: dict) -> dict[str, int]:
    """The rolling-strokes-gained window (3, 5 or 10 seasons) with the best pooled log loss."""
    settings = config["baseline1"]
    chosen = {}
    for label in LABELS:
        candidates = {
            window: baseline_run.walk_forward_predictions(
                features, label, baseline_run.window_column(settings["feature"], window), folds
            )
            for window in settings["windows"]
        }
        chosen[label] = min(candidates, key=lambda w: baseline_run.pooled_log_loss(candidates[w]))
    return chosen


def baseline_predictions(features: pd.DataFrame, folds: list, config: dict) -> dict:
    """Baseline 1 with its window chosen per label on the same folds, as in its own report.

    `has_history` is realigned to Model 1's rating-based definition (see
    `reporting.align_has_history`), since these predictions are always going into a table that
    also has Model 1 or the simulator in it.
    """
    settings = config["baseline1"]
    windows = chosen_baseline_windows(features, folds, config)
    return {
        label: align_has_history(
            baseline_run.walk_forward_predictions(
                features,
                label,
                baseline_run.window_column(settings["feature"], windows[label]),
                folds,
            ),
            features,
        )
        for label in LABELS
    }


def run_ablation(features, folds, settings, seed, baseline_top10) -> pd.DataFrame:
    rows = []
    reference = float(log_loss_terms(baseline_top10["y"], baseline_top10["p"]).mean())
    for name, columns in ABLATION.items():
        predictions = walk_forward_predictions(
            features, "top10", "logistic", columns, folds, settings, seed
        )
        loss = float(log_loss_terms(predictions["y"], predictions["p"]).mean())
        rows.append({"features": name, "log_loss": loss, "diff_vs_baseline_1": loss - reference})
    unnormalised = {**settings, "top10_normalise": False}
    predictions = walk_forward_predictions(
        features, "top10", "logistic", FULL_COLUMNS, folds, unnormalised, seed
    )
    loss = float(log_loss_terms(predictions["y"], predictions["p"]).mean())
    rows.append(
        {
            "features": "full model, no event normalisation",
            "log_loss": loss,
            "diff_vs_baseline_1": loss - reference,
        }
    )
    return pd.DataFrame(rows)


def _in_season(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    return frame[frame["season"] == season]


def calibration_in_the_large(
    features, folds, settings, seed, baseline_top10, model_top10
) -> pd.DataFrame:
    unnormalised = walk_forward_predictions(
        features,
        "top10",
        "logistic",
        FULL_COLUMNS,
        folds,
        {**settings, "top10_normalise": False},
        seed,
    )
    rows = []
    for season in sorted(model_top10["season"].unique()):
        pick = partial(_in_season, season=season)
        rows.append(
            {
                "season": season,
                "observed": pick(model_top10)["y"].mean(),
                "Baseline 1": pick(baseline_top10)["p"].mean(),
                "Model 1, no normalisation": pick(unnormalised)["p"].mean(),
                "Model 1": pick(model_top10)["p"].mean(),
            }
        )
    return pd.DataFrame(rows)


def coefficient_table(features, folds, settings, seed) -> pd.DataFrame:
    train = features[features["season"].isin(folds[-1][0])]
    table = {}
    for label, (kind, columns) in {k: v["Model 1"] for k, v in MODELS.items()}.items():
        defined = train[train[label].notna()]
        table[LABEL_NAMES[label]] = fit_model1(
            defined, label, kind, columns, settings, seed
        ).coefficients()
    frame = pd.DataFrame(table)
    return frame.reset_index(names="feature")


def build_report(config, folds, grid, best, rows, ablation, in_the_large, coefficients) -> str:
    validation = config["splits"]["validation_seasons"]
    samples, seed = config["evaluation"]["bootstrap_samples"], config["evaluation"]["seed"]
    fold_lines = "\n".join(f"- validate {test}: fit on seasons {train}" for train, test in folds)
    return f"""# Model 1: validation report

Generated by `uv run python -m src.models.run_model1` from `config/config.yaml`.
Validation seasons {validation}. The holdout ({config["splits"]["holdout_seasons"]}) and the
forward-test ({config["splits"]["forward_test_seasons"]}) seasons were **not** loaded or looked at.

Model 1 uses field-adjusted ratings built from round scores (each round measured against everyone
playing it, weighted toward recent rounds, shrunk when few rounds back it), features relative to the
field, recent form, rounds behind each rating, and the four season strokes-gained categories.
Logistic and gradient-boosting versions are shown for make-cut and top 10; win is a softmax over
the field. Top-10 probabilities are shifted within each event so they add up to the number of
top-10 places. Differences are model minus reference, so **negative means better**. Intervals are
95% bootstrap intervals over events ({samples} resamples, seed {seed}); differences against
Baseline 1 use the same resampled events for both, so they are paired.

## Walk-forward folds

{fold_lines}

## Rating settings (chosen on validation top-10 log loss, lower is better)

{markdown_table(grid)}

Chosen: long half-life {best[0]:.0f} days, shrinkage {best[1]:.0f} pseudo-rounds. Recent form uses a
{config["model1"]["rating"]["short_half_life_days"]}-day half-life. **The grid is essentially
flat** (every combination is within {grid["log_loss"].max() - grid["log_loss"].min():.4f} log loss
of each other), so this choice should not be read as informative on its own; treat any of these
settings as equivalent within noise.

## Results, pooled validation seasons

{results_table(rows, "pooled")}

## By season

{results_table(rows, "season 2023")}

{results_table(rows, "season 2024")}

## Players with rating history only

{results_table(rows, "pooled, players with rating history")}

## Ablation: top-10 log loss by feature set (logistic, pooled validation)

{markdown_table(ablation)}

## Top-10 calibration in the large (mean predicted probability, by validation season)

{markdown_table(in_the_large)}

## Fitted weights (standardised inputs; larger magnitude means more influence)

Fitted on seasons {folds[-1][0]}. Empty cells are inputs a model does not use.

{markdown_table(coefficients, 3)}

Calibration: `model1_calibration.png` (pooled validation predictions, equal-frequency bins).
"""


def main() -> None:
    config = load_config()
    settings, evaluation = config["model1"], config["evaluation"]
    seed, samples = evaluation["seed"], evaluation["bootstrap_samples"]

    events, rounds, folds = load_inputs(config)
    half_life, shrink, grid = tune_rating(events, rounds, folds, settings, seed)
    features = features_for(events, rounds, settings["rating"], half_life, shrink)

    baseline = baseline_predictions(features, folds, config)
    rows, kept = [], {}
    for label in LABELS:
        rows += score_model(label, "Baseline 1", baseline[label], None, samples, seed)
        for name, (kind, columns) in MODELS[label].items():
            predictions = walk_forward_predictions(
                features, label, kind, columns, folds, settings, seed
            )
            kept[(label, name)] = predictions
            rows += score_model(label, name, predictions, baseline[label], samples, seed)

    ablation = run_ablation(features, folds, settings, seed, baseline["top10"])
    in_the_large = calibration_in_the_large(
        features, folds, settings, seed, baseline["top10"], kept[("top10", "Model 1")]
    )
    coefficients = coefficient_table(features, folds, settings, seed)

    bins = evaluation["calibration_bins"]
    tables = {
        LABEL_NAMES[label]: {
            "Model 1": calibration_table(
                kept[(label, "Model 1")]["y"], kept[(label, "Model 1")]["p"], bins
            ),
            "Baseline 1": calibration_table(baseline[label]["y"], baseline[label]["p"], bins),
        }
        for label in LABELS
    }

    report_dir = Path(evaluation["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    text = build_report(
        config, folds, grid, (half_life, shrink), rows, ablation, in_the_large, coefficients
    )
    (report_dir / "model1_validation.md").write_text(text)
    pd.DataFrame(rows).to_csv(report_dir / "model1_validation_metrics.csv", index=False)
    validation = config["splits"]["validation_seasons"]
    plot_calibration(
        tables,
        report_dir / "model1_calibration.png",
        f"Model 1 vs Baseline 1 calibration, validation seasons {validation}",
    )
    print(text)


if __name__ == "__main__":
    main()
