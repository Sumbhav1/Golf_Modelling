import pytest

from src.config import assert_no_forward_test, load_config


def test_forward_test_season_is_refused():
    config = load_config()

    assert_no_forward_test(config, [2019, 2024, 2025])  # the holdout season is allowed here
    with pytest.raises(ValueError, match="forward-test"):
        assert_no_forward_test(config, [2026])
