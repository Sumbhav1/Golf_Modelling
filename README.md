# Golf Pricing Engine

A probabilistic model and Monte Carlo tournament simulator that prices PGA Tour outcomes (win, top-10, make-cut) and tests them honestly against bookmaker odds. Research only; it does not place bets.

**Status:** Phase 0 (foundations and data audit), nearly done: 2018-2025 schedules, results and season strokes gained are collected and documented. The 2026 season is kept in separate files for prediction. Tidy player-event and player-round tables with labels are built by `src/ingest/tidy.py`. Baseline 1 (rolling strokes gained to probabilities) beats the naive forecast on the 2023-2024 validation seasons ([report](reports/baseline1_validation.md)), and Model 1 (field-adjusted ratings from round scores) beats Baseline 1 on make-cut, top 10 and win ([report](reports/model1_validation.md)). The frozen 2025 holdout confirms make-cut and top 10; win's edge over Baseline 1 did not reproduce with significance ([holdout report](reports/holdout_2025_results.md)). A first Monte Carlo simulator (round-score model plus the cut) beats the naive forecast and Baseline 1 under a fair (no-outcome-information) cutline assumption, but not yet Model 1's classifier ([simulator report](reports/simulator_validation.md)). See [docs/PLAN.md](docs/PLAN.md) for the phases and checkboxes.

## Setup

```
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## Docs

- [docs/IDEA.md](docs/IDEA.md): what this is and why
- [docs/PLAN.md](docs/PLAN.md): phases and status
- [docs/DECISIONS.md](docs/DECISIONS.md): every modelling and design choice
- [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md): data sources, dates and terms
- [docs/DATA_STRUCTURE.md](docs/DATA_STRUCTURE.md): data files, columns, joins and quirks
