# Golf Pricing Engine

A probabilistic model and Monte Carlo tournament simulator that prices PGA Tour outcomes (win, top-10, make-cut) and tests them honestly against bookmaker odds. Research only; it does not place bets.

**Status:** early and in progress. The pipeline runs end to end on 2018-2025 PGA Tour data (plus a separate 2026 feed for live prediction): raw results and strokes gained in, tidy player-round tables out, two probability models and a first Monte Carlo simulator on top. Every model is checked against a naive baseline, a walk-forward validation split, and a frozen holdout season, with confidence intervals and calibration plots throughout. So far the models beat the naive baseline; a market baseline (bookmaker odds) isn't wired into a full backtest yet - see [reports/](reports/) for why. A small API and dashboard serve the results so far. Full results are in [`reports/`](reports/).

## Setup

```
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## Results

- [reports/baseline1_validation.md](reports/baseline1_validation.md): strokes-gained baseline
- [reports/model1_validation.md](reports/model1_validation.md): field-adjusted rating model
- [reports/holdout_2025_results.md](reports/holdout_2025_results.md): frozen 2025 holdout, one look
- [reports/simulator_validation.md](reports/simulator_validation.md): Monte Carlo simulator
- [reports/forward_test_2026_results.md](reports/forward_test_2026_results.md): ongoing 2026 forward test, updated as events finish

## API and dashboard

Both serve precomputed predictions (see `src/models/export_predictions.py`) rather than re-fitting a model per request, so build the export once and re-run it whenever the input data or a model setting changes:

```
uv sync
uv run python -m src.models.export_predictions
uv run uvicorn src.api.main:app --reload      # API: http://localhost:8000/docs
uv run streamlit run src/dashboard/app.py     # dashboard, in a separate terminal
```

The API has `/health`, `/events` and `/events/{tournament_id}/predictions`. The dashboard has a results viewer over `reports/`, an event explorer across validation, the frozen holdout and the ongoing forward test, and a backtest-plan page (the methodology, why it isn't running yet, and a live vig-removal calculator over `src/backtest/odds.py`). Neither has a market-vs-model panel or a backtest equity curve - there is no public odds source or backtest to show (see `reports/`), so those were dropped rather than built as placeholders.
