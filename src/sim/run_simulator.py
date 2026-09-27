"""Simulator validation on the validation seasons: compare against realized frequencies, Baseline 1
and Model 1's direct classifier.

Run with `uv run python -m src.sim.run_simulator`. Uses the same walk-forward folds, rating
settings (re-derived, not re-tuned) and validation seasons as `src.models.run_model1`; the holdout
and forward-test seasons are never loaded.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.models import run_model1 as model1_run
from src.models.evaluation import calibration_table, plot_calibration
from src.models.model1 import FULL_COLUMNS, WIN_COLUMNS
from src.models.model1 import walk_forward_predictions as model1_predictions
from src.models.reporting import LABEL_NAMES, markdown_table, results_table, score_model
from src.sim.tournament import (
    cut_survivor_count,
    estimate_residual_sigma,
    event_rng,
    simulate_event,
    simulate_events,
)

LABELS = model1_run.LABELS


def fold_sigmas(features: pd.DataFrame, rounds: pd.DataFrame, folds: list) -> dict[int, float]:
    """Residual sigma per fold's training seasons, computed once and shared by every use of it.

    Sigma does not depend on the cut mode, so the same per-fold value is used whichever cutline
    assumption is being simulated.
    """
    return {
        test_season: estimate_residual_sigma(
            features[features["season"].isin(train_seasons)],
            rounds[rounds["season"].isin(train_seasons)],
        )
        for train_seasons, test_season in folds
    }


def simulate_predictions(
    features: pd.DataFrame,
    folds: list,
    sigmas: dict[int, float],
    sim_settings: dict,
    cut_mode: str,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Simulated win/top10/made_cut predictions per validation fold, shaped for `score_model`."""
    parts = {label: [] for label in LABELS}
    for train_seasons, test_season in folds:
        train = features[features["season"].isin(train_seasons)]
        test = features[features["season"] == test_season]
        simulated = simulate_events(
            test,
            sigmas[test_season],
            sim_settings["n_sims"],
            seed,
            cut_mode,
            sim_settings["fallback_cut_fraction"],
        )
        # simulate_events' probability columns (win, top10, made_cut) share their names with the
        # actual outcome columns being merged in, so rename before merging to avoid a collision.
        simulated = simulated.rename(columns={label: f"{label}_p" for label in LABELS})
        merged = simulated.merge(
            test[["tournament_id", "player_id", "season", "no_history", *LABELS]],
            on=["tournament_id", "player_id"],
            validate="one_to_one",
        )
        for label in LABELS:
            defined = merged[merged[label].notna()].copy()
            defined["event"] = defined["tournament_id"]
            defined["y"] = defined[label].astype(int)
            defined["p"] = defined[f"{label}_p"]
            defined["p_naive"] = float(train[label].astype(float).mean())
            defined["has_history"] = defined["no_history"] == 0
            keep = [
                "season",
                "tournament_id",
                "event",
                "player_id",
                "y",
                "p",
                "p_naive",
                "has_history",
            ]
            parts[label].append(defined[keep])
    return {label: pd.concat(frames, ignore_index=True) for label, frames in parts.items()}


def time_ten_thousand_sims(
    features: pd.DataFrame, sigma: float, fallback_cut_fraction: float
) -> tuple[int, float]:
    """Fastest of 3 runs of 10,000 simulations of the largest validation-season field."""
    sizes = features[features["is_standard_event"]].groupby("tournament_id").size()
    biggest = features[features["tournament_id"] == sizes.idxmax()]
    rating = biggest["rating"].fillna(0.0).to_numpy()
    cut_keep = cut_survivor_count(len(rating), True, None, "fraction", fallback_cut_fraction)
    elapsed = []
    for attempt in range(3):
        start = time.perf_counter()
        simulate_event(rating, sigma, 10000, event_rng(attempt, "timing"), cut_keep=cut_keep)
        elapsed.append(time.perf_counter() - start)
    return len(rating), min(elapsed)


def build_report(
    config, folds, half_life, shrink, sigmas, rows, actual_rows, field_size, seconds
) -> str:
    validation = config["splits"]["validation_seasons"]
    samples, seed = config["evaluation"]["bootstrap_samples"], config["evaluation"]["seed"]
    fold_lines = "\n".join(f"- validate {test}: fit on seasons {train}" for train, test in folds)
    sigma_table = markdown_table(
        pd.DataFrame([{"validate": test, "sigma": sigmas[test]} for _, test in folds])
    )
    actual_table = markdown_table(
        pd.DataFrame(
            [
                {
                    "label": LABEL_NAMES[r["label_key"]],
                    "log_loss": r["log_loss"],
                    "diff_vs_fair_cut_mode": r["log_loss"] - r["fair_log_loss"],
                }
                for r in actual_rows
            ]
        )
    )
    return f"""# Simulator: validation report

Generated by `uv run python -m src.sim.run_simulator` from `config/config.yaml`.
Validation seasons {validation}. The holdout ({config["splits"]["holdout_seasons"]}) and the
forward-test ({config["splits"]["forward_test_seasons"]}) seasons were **not** loaded or looked at.

The simulator draws each round's field-relative score as `-rating + Normal(0, sigma)`, where
`rating` is Model 1's as-of field-adjusted rating (half-life {half_life:.0f} days, shrinkage
{shrink:.0f} pseudo-rounds, as chosen on validation) and `sigma` is the residual round-to-round
standard deviation, estimated separately on each fold's training seasons:

{sigma_table}

A cut is applied after round 2. Win, top-10 and make-cut probabilities are the simulated frequency
over {config["sim"]["n_sims"]:,} tournaments. **The main comparison below approximates the cutline
as a fixed {config["sim"]["fallback_cut_fraction"]:.0%} of the field (`cut_mode="fraction"`)**, the
same information a live forecast would have; it is a fair comparison against Baseline 1 and Model 1,
neither of which knows the true cutline either. Differences are model minus reference, so
**negative means better**. Intervals are 95% bootstrap intervals over events ({samples} resamples,
seed {seed}); differences against Baseline 1 use the same resampled events for both, so they are
paired.

## Walk-forward folds

{fold_lines}

## Results, pooled validation seasons (`cut_mode="fraction"`)

{results_table(rows, "pooled")}

## Speed check

10,000 simulations of the largest validation-season field ({field_size} players) took
{seconds:.3f} seconds (fastest of 3 runs) — the Phase 2 target is 10,000+ in seconds, checked here
independently of `sim.n_sims` ({config["sim"]["n_sims"]:,}, used for the results above).

## Diagnostic: replaying the true cutline (`cut_mode="actual"`)

This is **not** a fair comparison against Baseline 1 or Model 1: it gives the simulator the real
number of players who made the cut, an outcome the make-cut label directly measures, so it should
only be read as an upper bound on the round-score model's own quality, isolated from the cut-size
assumption. The gap below is this diagnostic's log loss minus the fair (`"fraction"`) result above:

{actual_table}

Win and top 10 are barely affected by the cutline assumption (favourites are rarely near the cut
line), so their diagnostic and fair-comparison numbers are close; make-cut is the label the cutline
assumption actually matters for.

Calibration: `simulator_calibration.png` (pooled validation predictions, equal-frequency bins,
`cut_mode="fraction"`).
"""


def main() -> None:
    config = load_config()
    validation = config["splits"]["validation_seasons"]
    seed, samples = config["evaluation"]["seed"], config["evaluation"]["bootstrap_samples"]

    events, rounds, folds = model1_run.load_inputs(config)
    half_life, shrink, _ = model1_run.tune_rating(events, rounds, folds, config["model1"], seed)
    features = model1_run.features_for(
        events, rounds, config["model1"]["rating"], half_life, shrink
    )
    sigmas = fold_sigmas(features, rounds, folds)

    baseline = model1_run.baseline_predictions(features, folds, config)
    model1 = {}
    rows = []
    for label in LABELS:
        kind, columns = ("softmax", WIN_COLUMNS) if label == "win" else ("logistic", FULL_COLUMNS)
        model1[label] = model1_predictions(
            features, label, kind, columns, folds, config["model1"], seed
        )
        rows += score_model(label, "Baseline 1", baseline[label], None, samples, seed)
        rows += score_model(label, "Model 1", model1[label], baseline[label], samples, seed)

    # The fair comparison: the simulator does not get to know the true cutline.
    simulated = simulate_predictions(features, folds, sigmas, config["sim"], "fraction", seed)
    for label in LABELS:
        rows += score_model(label, "Simulator", simulated[label], baseline[label], samples, seed)

    # A diagnostic only: replays the real number of cut survivors, which leaks the make-cut
    # outcome's aggregate count. Scored against the fair result above, not against Baseline 1.
    actual_simulated = simulate_predictions(features, folds, sigmas, config["sim"], "actual", seed)
    actual_rows = []
    for label in LABELS:
        fair_loss = next(
            r
            for r in rows
            if r["label"] == LABEL_NAMES[label]
            and r["model"] == "Simulator"
            and r["slice"] == "pooled"
        )["log_loss"]
        scored = score_model(
            label, "Simulator (actual cut)", actual_simulated[label], None, samples, seed
        )
        pooled = next(r for r in scored if r["slice"] == "pooled")
        actual_rows.append({**pooled, "label_key": label, "fair_log_loss": fair_loss})

    field_size, seconds = time_ten_thousand_sims(
        features[features["season"] == folds[-1][1]], sigmas[folds[-1][1]],
        config["sim"]["fallback_cut_fraction"],
    )  # fmt: skip

    bins = config["evaluation"]["calibration_bins"]
    tables = {
        LABEL_NAMES[label]: {
            "Simulator": calibration_table(simulated[label]["y"], simulated[label]["p"], bins),
            "Baseline 1": calibration_table(baseline[label]["y"], baseline[label]["p"], bins),
            "Model 1": calibration_table(model1[label]["y"], model1[label]["p"], bins),
        }
        for label in LABELS
    }

    report_dir = Path(config["evaluation"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    text = build_report(
        config, folds, half_life, shrink, sigmas, rows, actual_rows, field_size, seconds
    )
    (report_dir / "simulator_validation.md").write_text(text)
    pd.DataFrame(rows).to_csv(report_dir / "simulator_validation_metrics.csv", index=False)
    plot_calibration(
        tables,
        report_dir / "simulator_calibration.png",
        f"Simulator vs Baseline 1 vs Model 1 calibration, validation seasons {validation}",
    )
    print(text)


if __name__ == "__main__":
    main()
