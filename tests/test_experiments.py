
import json

import pytest

from modelbench.experiments.tracker import (
    save_experiment,
    load_experiments,
)


def sample_metrics():
    return {
        "accuracy": 0.90,
        "precision": 0.89,
        "recall": 0.90,
        "f1_score": 0.89,
    }


def test_save_and_load_experiment(tmp_path):
    history_file = tmp_path / "history.json"

    saved = save_experiment(
        "Random Forest",
        sample_metrics(),
        history_file=history_file,
        notes="Initial experiment",
    )

    history = load_experiments(history_file)

    assert len(history) == 1
    assert history[0]["experiment_id"] == saved["experiment_id"]
    assert history[0]["model_name"] == "Random Forest"
    assert history[0]["metrics"]["accuracy"] == 0.90
    assert history[0]["notes"] == "Initial experiment"
    assert "timestamp" in history[0]


def test_missing_history_returns_empty_list(tmp_path):
    assert load_experiments(tmp_path / "missing.json") == []


def test_missing_metric_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Missing required metric"):
        save_experiment(
            "Random Forest",
            {"accuracy": 0.90},
            history_file=tmp_path / "history.json",
        )


def test_invalid_history_json_is_rejected(tmp_path):
    history_file = tmp_path / "history.json"
    history_file.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        load_experiments(history_file)
