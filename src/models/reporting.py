"""Report-building helpers shared by the Model 1 validation and holdout reports."""

from __future__ import annotations

import pandas as pd

from src.models import run_baseline1 as baseline_run
from src.models.evaluation import score_forecasts

LABEL_NAMES = baseline_run.LABEL_NAMES
KEYS = ["tournament_id", "player_id"]


def paired_scores(frame: pd.DataFrame, reference: pd.DataFrame, samples: int, seed: int) -> dict:
    """Scores of `frame` against another forecast on the same rows (difference: frame minus it)."""
    other = reference[[*KEYS, "p"]].rename(columns={"p": "p_naive"})
    merged = frame.drop(columns="p_naive").merge(other, on=KEYS, how="left", validate="one_to_one")
    return score_forecasts(merged, samples, seed)


def align_has_history(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Overwrite `has_history` with Model 1's rating-based definition (`no_history == 0`).

    Baseline 1 defines "has history" as having a rolling-strokes-gained value (a narrower,
    season-level criterion, since season SG needs ShotLink coverage); Model 1 and the simulator
    define it as having any earlier round at all, a much broader population. Left unaligned, a
    "players with history" slice silently compares two different populations between models, which
    made the reported numbers in that slice non-comparable (see docs/DECISIONS.md). Call this on
    Baseline 1's predictions before putting them in the same comparison table as Model 1 or the
    simulator, both of which already use the rating-based definition.
    """
    canonical = features[["tournament_id", "player_id", "no_history"]].drop_duplicates(
        ["tournament_id", "player_id"]
    )
    merged = predictions.drop(columns="has_history").merge(
        canonical, on=["tournament_id", "player_id"], how="left", validate="many_to_one"
    )
    merged["has_history"] = merged["no_history"] == 0
    return merged.drop(columns="no_history")


def slices_of(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    slices = {"pooled": predictions}
    for season in sorted(predictions["season"].unique()):
        slices[f"season {season}"] = predictions[predictions["season"] == season]
    slices["pooled, players with rating history"] = predictions[predictions["has_history"]]
    return slices


def score_model(label, name, predictions, baseline, samples, seed) -> list[dict]:
    """One row per slice: this model's scores, and (if `baseline` is given) paired diffs vs it."""
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
