"""Load config/config.yaml and guard the frozen holdout."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml

DEFAULT_PATH = Path("config/config.yaml")


def load_config(path: Path | str = DEFAULT_PATH) -> dict:
    with open(path) as file:
        return yaml.safe_load(file)


def assert_no_holdout(config: dict, seasons: Iterable[int]) -> None:
    """Raise if any season is a holdout or forward-test season.

    Call this with every season used for fitting, tuning or selection. The holdout is for final
    reporting only, and the forward-test season is kept out of tuning too.
    """
    reserved = set(config["splits"]["holdout_seasons"]) | set(
        config["splits"]["forward_test_seasons"]
    )
    used_reserved = sorted(reserved & set(seasons))
    if used_reserved:
        raise ValueError(f"Seasons {used_reserved} are reserved (holdout or forward test).")


def assert_no_forward_test(config: dict, seasons: Iterable[int]) -> None:
    """Raise if any season is the forward-test season.

    For the one deliberate holdout evaluation (`src/models/run_holdout.py`), which is allowed to
    use the holdout season but never the forward-test season.
    """
    used = sorted(set(config["splits"]["forward_test_seasons"]) & set(seasons))
    if used:
        raise ValueError(f"Seasons {used} are the forward-test season, not to be evaluated yet.")
