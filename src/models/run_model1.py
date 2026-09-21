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
    score_forecasts,
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

LABELS = baseline_run.LABELS
LABEL_NAMES = baseline_run.LABEL_NAMES
KEYS = ["tournament_id", "player_id"]
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


def baseline_predictions(features: pd.DataFrame, folds: list, config: dict) -> dict:
    """Baseline 1 with its window chosen per label on the same folds, as in its own report."""
    settings = config["baseline1"]
    out = {}
    for label in LABELS:
        candidates = {
            window: baseline_run.walk_forward_predictions(
                features, label, baseline_run.window_column(settings["feature"], window), folds
            )
            for window in settings["windows"]
        }
        best = min(candidates, key=lambda w: baseline_run.pooled_log_loss(candidates[w]))
        out[label] = candidates[best]
    return out


def paired_scores(frame: pd.DataFrame, reference: pd.DataFrame, samples: int, seed: int) -> dict:
    """Scores of `frame` against another forecast on the same rows (difference: frame minus it)."""
    other = reference[[*KEYS, "p"]].rename(columns={"p": "p_naive"})
    merged = frame.drop(columns="p_naive").merge(other, on=KEYS, how="left", validate="one_to_one")
    return score_forecasts(merged, samples, seed)


def slices_of(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    slices = {"pooled": predictions}
    for season in sorted(predictions["season"].unique()):
        slices[f"season {season}"] = predictions[predictions["season"] == season]
    slices["pooled, players with rating history"] = predictions[predictions["has_history"]]
    return slices


def score_model(label, name, predictions, baseline, samples, seed) -> list[dict]:
    rows = []
    for slice_name, frame in slices_of(predictions).items():
        row = {"label": LABEL_NAMES[label], "model": name, "slice": slice_name}
        row.update(score_forecasts(frame, samples, seed))
        if baseline is not None:
            versus = paired_scores(frame, baseline, samples, seed)
            row.update(
                {
                    f"b1_{k}": v
                    for k, v in versus.items()
                    if k.endswith(("delta", "delta_lo", "delta_hi"))
                }
            )
        rows.append(row)
    return rows


def _interval(value: float, lower: float, upper: float, digits: int = 4) -> str:
    return f"{value:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"


def results_table(rows: list[dict], slice_name: str) -> str:
    header = (
        "| Label | Model | Log loss (95% CI) | Diff vs naive (95% CI) "
        "| Diff vs Baseline 1 (95% CI) | Brier | Brier diff vs Baseline 1 (95% CI) |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in (r for r in rows if r["slice"] == slice_name):
        vs_b1 = (
            _interval(r["b1_log_loss_delta"], r["b1_log_loss_delta_lo"], r["b1_log_loss_delta_hi"])
            if "b1_log_loss_delta" in r and pd.notna(r["b1_log_loss_delta"])
            else "-"
        )
        brier_b1 = (
            _interval(r["b1_brier_delta"], r["b1_brier_delta_lo"], r["b1_brier_delta_hi"], 5)
            if "b1_brier_delta" in r and pd.notna(r["b1_brier_delta"])
            else "-"
        )
        lines.append(
            f"| {r['label']} | {r['model']} "
            f"| {_interval(r['log_loss'], r['log_loss_lo'], r['log_loss_hi'])} "
            f"| {_interval(r['log_loss_delta'], r['log_loss_delta_lo'], r['log_loss_delta_hi'])} "
            f"| {vs_b1} | {r['brier']:.5f} | {brier_b1} |"
        )
    return header + "\n".join(lines)


def _cell(value, digits: int) -> str:
    if isinstance(value, float):
        return "-" if pd.isna(value) else f"{value:.{digits}f}"
    return str(value)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    header = "| " + " | ".join(frame.columns) + " |\n|" + "---|" * len(frame.columns) + "\n"
    body = "\n".join(
        "| " + " | ".join(_cell(v, digits) for v in row) + " |"
        for row in frame.itertuples(index=False)
    )
    return header + body


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
{config["model1"]["rating"]["short_half_life_days"]}-day half-life.

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
