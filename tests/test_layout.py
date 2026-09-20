import importlib

import pytest

PACKAGES = ["ingest", "features", "models", "sim", "backtest", "api"]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    """Each area in the CLAUDE.md layout is an importable package."""
    importlib.import_module(f"src.{name}")
