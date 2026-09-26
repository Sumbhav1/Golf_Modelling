"""Scoring for probability forecasts: metrics, walk-forward folds, event bootstrap, calibration.

Rows within an event are correlated, so confidence intervals resample whole events, not rows.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-15
CI_LEVEL = 0.95


def log_loss_terms(y: Iterable[float], p: Iterable[float]) -> np.ndarray:
    """Per-row log loss; probabilities are clipped away from 0 and 1."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier_terms(y: Iterable[float], p: Iterable[float]) -> np.ndarray:
    """Per-row squared error of the probability."""
    return (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2


def walk_forward_folds(
    seasons: Iterable[int], validation_seasons: Iterable[int], first_train_season: int
) -> list[tuple[list[int], int]]:
    """(train seasons, test season) pairs; each test season uses only earlier seasons to train."""
    available = sorted({int(season) for season in seasons})
    folds = []
    for test in sorted(validation_seasons):
        train = [s for s in available if first_train_season <= s < test]
        if not train:
            raise ValueError(f"No training seasons before {test}.")
        folds.append((train, test))
    return folds


def bootstrap_interval(
    sums: np.ndarray, counts: np.ndarray, samples: int, seed: int
) -> tuple[float, float]:
    """95% interval for sum(sums) / sum(counts), resampling events with replacement."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(samples, len(sums)))
    statistic = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    lower, upper = np.quantile(statistic, [(1 - CI_LEVEL) / 2, 1 - (1 - CI_LEVEL) / 2])
    return float(lower), float(upper)


def score_forecasts(frame: pd.DataFrame, samples: int, seed: int) -> dict[str, float]:
    """Log loss and Brier score of a forecast and of the naive one, with event-bootstrap intervals.

    `frame` needs the columns event, y, p and p_naive. Differences are model minus naive, so a
    negative value means the model is better. Intervals for differences are paired: the same
    resampled events are used for both forecasts.
    """
    terms = pd.DataFrame(
        {
            "event": frame["event"].to_numpy(),
            "log_loss": log_loss_terms(frame["y"], frame["p"]),
            "log_loss_naive": log_loss_terms(frame["y"], frame["p_naive"]),
            "brier": brier_terms(frame["y"], frame["p"]),
            "brier_naive": brier_terms(frame["y"], frame["p_naive"]),
        }
    )
    by_event = terms.groupby("event")
    sums = by_event.sum()
    counts = by_event.size().to_numpy(dtype=float)

    result: dict[str, float] = {"rows": float(len(terms)), "events": float(len(counts))}
    for name in ["log_loss", "brier"]:
        model, naive = sums[name].to_numpy(), sums[f"{name}_naive"].to_numpy()
        result[name] = model.sum() / counts.sum()
        result[f"{name}_naive"] = naive.sum() / counts.sum()
        result[f"{name}_lo"], result[f"{name}_hi"] = bootstrap_interval(
            model, counts, samples, seed
        )
        result[f"{name}_delta"] = result[name] - result[f"{name}_naive"]
        result[f"{name}_delta_lo"], result[f"{name}_delta_hi"] = bootstrap_interval(
            model - naive, counts, samples, seed
        )
    return result


def calibration_table(y: Iterable[float], p: Iterable[float], bins: int) -> pd.DataFrame:
    """Equal-frequency bins of the predicted probability: mean prediction, observed rate, rows."""
    frame = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    frame["bin"] = pd.qcut(frame["p"], q=bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True)
    return pd.DataFrame(
        {
            "mean_predicted": grouped["p"].mean(),
            "observed_rate": grouped["y"].mean(),
            "rows": grouped.size(),
        }
    ).reset_index(drop=True)


# Reference palette (validated default): categorical slots 1-3 on the light chart surface (the
# first three slots are the ones documented to validate against every pair, not just adjacent).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua


def plot_calibration(
    tables: dict[str, dict[str, pd.DataFrame]], path: Path | str, title: str
) -> None:
    """One reliability panel per label, one curve per model. On the dashed line is calibrated.

    `tables` maps label -> model name -> calibration table. The first model gets the first colour.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(tables), figsize=(4.2 * len(tables), 4.4), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for axis, (label, models) in zip(axes, tables.items(), strict=True):
        top = max(max(t["mean_predicted"].max(), t["observed_rate"].max()) for t in models.values())
        top *= 1.08
        axis.set_facecolor(SURFACE)
        axis.plot(
            [0, top],
            [0, top],
            color=INK_SECONDARY,
            linewidth=1,
            linestyle="--",
            label="Perfect calibration",
        )
        for color, (name, table) in zip(SERIES_COLORS, models.items(), strict=False):
            axis.plot(
                table["mean_predicted"],
                table["observed_rate"],
                color=color,
                linewidth=2,
                marker="o",
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=name,
            )
        axis.set_xlim(0, top)
        axis.set_ylim(0, top)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(label, color=INK, fontsize=12, loc="left")
        axis.set_xlabel("Predicted probability", color=INK_SECONDARY)
        axis.grid(color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=INK_SECONDARY)
        for spine in ["top", "right"]:
            axis.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            axis.spines[spine].set_color(GRID)
    axes[0].set_ylabel("Observed frequency", color=INK_SECONDARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(labels), frameon=False,
        labelcolor=INK_SECONDARY,
    )  # fmt: skip
    fig.suptitle(title, color=INK, fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
