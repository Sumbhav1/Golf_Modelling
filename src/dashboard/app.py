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
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.api import serving

REPORT_DIR = Path("reports")

REPORTS = {
    "Baseline 1 (validation)": ("baseline1_validation", "baseline1_calibration.png"),
    "Model 1 (validation)": ("model1_validation", "model1_calibration.png"),
    "Simulator (validation)": ("simulator_validation", "simulator_calibration.png"),
    "Frozen holdout, 2025 (one look)": ("holdout_2025_results", "holdout_2025_calibration.png"),
    "Forward test, 2026 (ongoing)": (
        "forward_test_2026_results",
        "forward_test_2026_calibration.png",
    ),
}
LABELS = ["win", "top10", "made_cut"]

st.set_page_config(page_title="Golf Pricing Engine", layout="wide")


def results_page() -> None:
    st.header("Model results")
    st.caption("Full write-ups with methodology and limitations: the matching file in `reports/`.")
    choice = st.selectbox("Report", list(REPORTS))
    stem, image = REPORTS[choice]
    metrics_path = REPORT_DIR / f"{stem}_metrics.csv"
    image_path = REPORT_DIR / image

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        pooled = metrics[metrics["slice"] == "pooled"] if "slice" in metrics.columns else metrics
        st.dataframe(pooled.drop(columns=["slice"], errors="ignore"), width="stretch")
    else:
        st.info(f"No metrics yet at {metrics_path}; run the matching report script first.")

    if image_path.exists():
        st.image(str(image_path))
    else:
        st.info(f"No calibration plot yet at {image_path}; run the matching report script first.")


def _player_table(frame: pd.DataFrame, label: str) -> pd.DataFrame:
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


def explorer_page() -> None:
    st.header("Event explorer")
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

    label = st.radio("Outcome", LABELS, horizontal=True)
    table = _player_table(frame, label)
    if table.empty:
        st.info("No predictions for this outcome in this event.")
        return

    st.dataframe(table.rename(columns={"display_name": "Player"}), width="stretch", hide_index=True)
    if "Model 1" in table.columns:
        top = table.head(20).set_index("display_name")["Model 1"]
        st.caption("Model 1 probability, top 20 by that model")
        st.bar_chart(top)


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
