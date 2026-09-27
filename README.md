# Golf Pricing Engine

A probabilistic model and Monte Carlo tournament simulator that prices PGA Tour outcomes (win, top-10, make-cut) and tests them honestly against bookmaker odds. Research only; it does not place bets.

**Status:** early and in progress. The pipeline runs end to end on 2018-2025 PGA Tour data (plus a separate 2026 feed for live prediction): raw results and strokes gained in, tidy player-round tables out, two probability models and a first Monte Carlo simulator on top. Every model is checked against a naive baseline, a walk-forward validation split, and a frozen holdout season, with confidence intervals and calibration plots throughout. So far the models beat the naive baseline; a market baseline (bookmaker odds) isn't wired in yet. Full results are in [`reports/`](reports/).

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
