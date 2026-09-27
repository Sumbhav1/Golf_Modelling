"""Streamlit dashboard over the exported predictions (`src.models.export_predictions`) and the
published reports in `reports/`.

Run with `uv run streamlit run src/dashboard/app.py`. Reads the same bundle the API serves through
`src.api.serving` - no HTTP calls between the two, and neither ever re-fits a model at request
time (see docs/DECISIONS.md). Research only; does not place bets.

Scope (v1): a viewer over the already-published validation/holdout/forward-test reports, and a
per-event, per-player prediction browser across all four forecasts (Baseline 1, Model 1, Simulator,
Naive). What this does *not* do, because the data does not support it yet: a market-vs-model panel
(no public odds source; see docs/DECISIONS.md) and a raw simulated-outcome distribution (the
exported bundle holds each player's simulated *probability*, not the underlying per-simulation
draws) - both dropped rather than built as placeholders, per the same call made in `docs/PLAN.md`.

Both pages lead with plain-language framing (a glossary, per-metric verdicts, a per-event "what
actually happened" callout) before the underlying numbers, rather than a raw dump of report/export
columns - the numbers alone don't say whether a difference is real or noise, or what "log loss"
even means, so the page says it instead of leaving that to the reader.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.api import serving
from src.models.run_baseline1 import LABEL_NAMES

REPORT_DIR = Path("reports")

REPORTS = {
    "Baseline 1 (validation)": ("baseline1_validation", "baseline1_calibration.png", "Baseline 1"),
    "Model 1 (validation)": ("model1_validation", "model1_calibration.png", None),
    "Simulator (validation)": ("simulator_validation", "simulator_calibration.png", None),
    "Frozen holdout, 2025 (one look)": (
        "holdout_2025",
        "holdout_2025_calibration.png",
        None,
    ),
    "Forward test, 2026 (ongoing)": (
        "forward_test_2026",
        "forward_test_2026_calibration.png",
        None,
    ),
}
LABELS = ["win", "top10", "made_cut"]
LABEL_HELP = {
    "win": "Won the tournament outright.",
    "top10": "Finished in the top 10.",
    "made_cut": "Played all four rounds (missing the cut ends the tournament after two).",
}
LABEL_VERB = {"top10": "finished in the top 10", "made_cut": "made the cut"}
MODEL_HELP = {
    "Naive": "The same probability for every player: the historical rate from the training "
    "seasons. The bar every model has to beat.",
    "Baseline 1": "Ranks players by their own recent strokes-gained - no ratings, just a rolling "
    "average.",
    "Model 1": "The project's main model: a field-adjusted skill rating plus recent form, rest "
    "and strokes-gained categories.",
    "Model 1 boosting": "The same features as Model 1, fit with gradient boosting instead of "
    "logistic regression, to check whether a more flexible model helps.",
    "Simulator": "Simulates all four rounds of the tournament (and the cut) thousands of times "
    "using Model 1's skill ratings, and reads off how often each outcome happens.",
}

st.set_page_config(page_title="Golf Pricing Engine", layout="wide")


def _metrics_glossary() -> None:
    with st.expander("What do these numbers mean?"):
        st.markdown(
            """
- **Log loss** - how good the predicted *probabilities* were, not just whether the favourite
  won. Being confidently wrong is punished hard; being right with a hedged, honest probability is
  rewarded. **Lower is always better**, and 0 would be a perfect, certain prediction.
- **Brier score** - a similar idea on a plainer 0-1 scale: the average squared gap between the
  predicted probability and what actually happened (0 or 1). Lower is better here too.
- **"vs guessing" / "vs naive"** - the naive forecast gives every player the same historical rate
  (e.g. "52% of players make the cut, on average"). A real model should do better than that; the
  numbers below say by how much, and whether that gap could just be noise.
- **95% interval** - given how few events we actually have, this number could plausibly have
  landed anywhere in this range by chance alone. If an interval for a *difference* (vs guessing, or
  vs another model) includes zero, we can't be confident there's a real difference at all.
- **"Pooled"** - every event in that report's seasons combined into one number.
            """
        )


def _verdict(delta_lo: float, delta_hi: float) -> str:
    """Plain-language read of a paired difference's 95% interval (negative = this is better)."""
    if pd.isna(delta_lo) or pd.isna(delta_hi):
        return ""
    if delta_hi < 0:
        return "clearly better"
    if delta_lo > 0:
        return "clearly worse"
    return "not clearly different - could be noise"


def _headline_metrics(metrics: pd.DataFrame, fallback_model_name: str | None) -> None:
    """One column of metrics per (label, model) in the pooled slice, read in plain language."""
    pooled = metrics[metrics["slice"] == "pooled"] if "slice" in metrics.columns else metrics
    has_model_column = "model" in pooled.columns

    for label_name in pooled["label"].unique():
        rows = pooled[pooled["label"] == label_name].to_dict("records")
        if not has_model_column:
            rows = [{**rows[0], "model": fallback_model_name}]
        st.markdown(f"#### {label_name}")
        columns = st.columns(len(rows))
        for column, row in zip(columns, rows, strict=True):
            with column:
                st.metric(
                    row["model"],
                    f"{row['log_loss']:.4f} log loss",
                    delta=f"{row['log_loss_delta']:+.4f} vs guessing",
                    delta_color="inverse",
                    help=MODEL_HELP.get(row["model"], ""),
                )
                verdict = _verdict(row["log_loss_delta_lo"], row["log_loss_delta_hi"])
                if verdict:
                    st.caption(f"{verdict} than guessing the historical rate for everyone.")
                b1_delta = row.get("b1_log_loss_delta")
                if pd.notna(b1_delta):
                    b1_verdict = _verdict(row["b1_log_loss_delta_lo"], row["b1_log_loss_delta_hi"])
                    sign = "better" if b1_delta < 0 else "worse"
                    st.caption(f"{abs(b1_delta):.4f} {sign} than Baseline 1 ({b1_verdict}).")


def results_page() -> None:
    st.header("Model results")
    st.caption("Full write-ups with methodology and limitations: the matching file in `reports/`.")
    _metrics_glossary()

    choice = st.selectbox("Report", list(REPORTS))
    stem, image, fallback_model_name = REPORTS[choice]
    metrics_path = REPORT_DIR / f"{stem}_metrics.csv"
    image_path = REPORT_DIR / image

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        _headline_metrics(metrics, fallback_model_name)
        with st.expander("Show the underlying table"):
            pooled = (
                metrics[metrics["slice"] == "pooled"] if "slice" in metrics.columns else metrics
            )
            st.dataframe(pooled.drop(columns=["slice"], errors="ignore"), width="stretch")
    else:
        st.info(f"No metrics yet at {metrics_path}; run the matching report script first.")

    if image_path.exists():
        st.markdown("#### Calibration")
        st.caption(
            "For each band of predicted probability, does that outcome actually happen about "
            "that often? Points on the dashed line are perfectly calibrated; above it means the "
            "model was too cautious for that band, below it means too confident."
        )
        st.image(str(image_path))
    else:
        st.info(f"No calibration plot yet at {image_path}; run the matching report script first.")


def _player_table(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Wide, per-player table for one outcome: one probability column per model, plus what
    actually happened. Probabilities stay as 0-1 floats here (percent formatting is applied only
    at display time, in `explorer_page`), so callers can still do arithmetic on them."""
    label_frame = frame[frame["label"] == label]
    wide = label_frame.pivot_table(
        index=["player_id", "display_name"], columns="model", values="p"
    ).reset_index()
    actual = (
        label_frame.groupby(["player_id", "display_name"])["y"].first().rename("actual")
    ).reset_index()
    wide = wide.merge(actual, on=["player_id", "display_name"])
    sort_column = "Model 1" if "Model 1" in wide.columns else wide.columns[2]
    return wide.sort_values(sort_column, ascending=False).drop(columns="player_id")


def _best_model_column(table: pd.DataFrame) -> str | None:
    for name in ("Model 1", "Simulator", "Baseline 1", "Naive"):
        if name in table.columns:
            return name
    return None


def _event_insight(table: pd.DataFrame, label: str) -> str:
    """One sentence tying the table to what actually happened in this specific event."""
    model = _best_model_column(table)
    if model is None:
        return ""
    happened = table[table["actual"] == 1]
    if happened.empty:
        return "Nobody in this field achieved this outcome."

    if label == "win":
        winner = happened.iloc[0]
        rank = int((table[model] > winner[model]).sum()) + 1
        return (
            f"The actual winner, **{winner['display_name']}**, was given a "
            f"**{winner[model]:.1%}** win probability by {model} - ranked #{rank} of "
            f"{len(table)} in the field."
        )

    not_happened = table[table["actual"] == 0]
    p_happened = happened[model].median()
    p_not = not_happened[model].median() if not not_happened.empty else float("nan")
    verb = LABEL_VERB.get(label, label)
    if pd.isna(p_not):
        return (
            f"Players who {verb} had a median predicted probability of "
            f"**{p_happened:.0%}** ({model})."
        )
    gap = p_happened - p_not
    read = (
        "separated them well"
        if gap > 0.15
        else "separated them a little"
        if gap > 0.03
        else "barely separated them"
    )
    return (
        f"Players who {verb} had a median predicted probability of **{p_happened:.0%}** ({model}), "
        f"vs **{p_not:.0%}** for those who didn't - {read}."
    )


def explorer_page() -> None:
    st.header("Event explorer")
    st.caption(
        "Every model's probability for every player in one event, next to what actually happened."
    )
    with st.expander("What am I looking at?"):
        st.markdown(
            "\n".join(f"- **{name}**: {help_text}" for name, help_text in MODEL_HELP.items())
        )
        st.caption(
            "Naive and Baseline 1 use only strokes-gained history; Model 1 and the Simulator "
            "also use a field-adjusted skill rating built from every round played (see "
            "`reports/model1_validation.md`)."
        )

    events = serving.list_events()

    eras = sorted(events["era"].unique())
    default_era = eras.index("forward_test") if "forward_test" in eras else 0
    era = st.selectbox("Era", eras, index=default_era)
    subset = events[events["era"] == era]

    seasons = sorted(subset["season"].unique())
    season = st.selectbox("Season", seasons, index=len(seasons) - 1)
    subset = subset[subset["season"] == season].sort_values("start_date")

    label_by_id = {
        row["tournament_id"]: f"{row['tournament_name']} ({row['start_date']})"
        for _, row in subset.iterrows()
    }
    tournament_id = st.selectbox(
        "Tournament", list(label_by_id), format_func=lambda tid: label_by_id[tid]
    )

    event_row, frame = serving.event_predictions(tournament_id)
    st.subheader(
        f"{event_row['tournament_name']} — {event_row['start_date']} to {event_row['end_date']}"
    )
    st.caption(f"Field size {int(event_row['field_size'])}. Era: {era}.")

    label = st.radio(
        "Outcome", LABELS, format_func=lambda label_key: LABEL_NAMES[label_key], horizontal=True
    )
    st.caption(LABEL_HELP[label])
    table = _player_table(frame, label)
    if table.empty:
        st.info("No predictions for this outcome in this event.")
        return

    insight = _event_insight(table, label)
    if insight:
        st.info(insight)

    probability_columns = [c for c in table.columns if c not in ("display_name", "actual")]
    column_config = {
        "display_name": st.column_config.TextColumn("Player"),
        "actual": st.column_config.CheckboxColumn(
            f"Actually {LABEL_NAMES[label].lower()}?", help="What really happened in this event."
        ),
        **{
            name: st.column_config.ProgressColumn(
                name, help=MODEL_HELP.get(name, ""), format="%.0f%%", min_value=0, max_value=100
            )
            for name in probability_columns
        },
    }
    display_table = table.copy()
    display_table["actual"] = display_table["actual"].astype(bool)
    for name in probability_columns:
        display_table[name] = display_table[name] * 100
    st.dataframe(
        display_table,
        width="stretch",
        hide_index=True,
        column_config=column_config,
        column_order=["display_name", *probability_columns, "actual"],
    )


def main() -> None:
    st.title("Golf Pricing Engine")
    st.caption(
        "A probabilistic model and Monte Carlo simulator pricing PGA Tour outcomes. "
        "Research only, does not place bets."
    )
    try:
        serving.load_bundle()
    except serving.BundleNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    page = st.sidebar.radio("Page", ["Model results", "Event explorer"])
    if page == "Model results":
        results_page()
    else:
        explorer_page()


main()
