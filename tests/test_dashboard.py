import pandas as pd

from src.dashboard.app import _player_table

FRAME = pd.DataFrame(
    [
        {
            "player_id": 1,
            "display_name": "Player One",
            "label": "win",
            "model": "Model 1",
            "p": 0.3,
            "y": 1,
        },
        {
            "player_id": 1,
            "display_name": "Player One",
            "label": "win",
            "model": "Baseline 1",
            "p": 0.2,
            "y": 1,
        },
        {
            "player_id": 2,
            "display_name": "Player Two",
            "label": "win",
            "model": "Model 1",
            "p": 0.1,
            "y": 0,
        },
        {
            "player_id": 2,
            "display_name": "Player Two",
            "label": "win",
            "model": "Baseline 1",
            "p": 0.15,
            "y": 0,
        },
        {
            "player_id": 1,
            "display_name": "Player One",
            "label": "made_cut",
            "model": "Model 1",
            "p": 0.9,
            "y": 1,
        },
    ]
)


def test_player_table_sorts_by_model1_descending_and_keeps_the_actual_outcome():
    table = _player_table(FRAME, "win")

    assert list(table["display_name"]) == ["Player One", "Player Two"]
    assert list(table["actual"]) == [1, 0]
    assert set(table.columns) == {"display_name", "Model 1", "Baseline 1", "actual"}


def test_player_table_is_empty_for_a_label_with_no_rows():
    table = _player_table(FRAME, "top10")

    assert table.empty
