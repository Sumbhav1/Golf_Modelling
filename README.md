# Golf Pricing Engine

A probabilistic model and Monte Carlo tournament simulator that prices PGA Tour outcomes (win, top-10, make-cut) and tests them honestly against bookmaker odds. Research only; it does not place bets.

**Status:** Phase 0 (foundations and data audit). No model or results yet. See [docs/PLAN.md](docs/PLAN.md) for the phases and checkboxes.

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
