
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from modelbench.evaluation.metrics import calculate_metrics
from modelbench.evaluation.cross_validation import run_cross_validation
from modelbench.evaluation.visualizations import (
    plot_confusion_matrix,
    plot_model_comparison,
)


# ---------- Metrics Tests ----------

def test_calculate_metrics():
    actual = [0, 1, 0, 1]
    predicted = [0, 1, 1, 1]

    result = calculate_metrics(actual, predicted)

    assert result["accuracy"] == 0.75
    assert 0 <= result["precision"] <= 1
    assert 0 <= result["recall"] <= 1
    assert 0 <= result["f1_score"] <= 1
    assert len(result["confusion_matrix"]) == 2


def test_metrics_reject_empty_data():
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_metrics([], [])


def test_metrics_reject_different_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        calculate_metrics([0, 1], [0])


# ---------- Cross-Validation Tests ----------

def test_cross_validation_returns_scores():
    df = pd.DataFrame({
        "age": list(range(40)),
        "city": ["Bengaluru", "Mysuru"] * 20,
        "target": [0, 1] * 20,
    })

    result = run_cross_validation(
        df,
        target_column="target",
        model_name="Logistic Regression",
        folds=5,
    )

    assert len(result["accuracy_scores"]) == 5
    assert 0 <= result["mean_accuracy"] <= 1
    assert 0 <= result["mean_f1_score"] <= 1


def test_cross_validation_rejects_too_many_folds():
    df = pd.DataFrame({
        "feature": list(range(6)),
        "target": [0, 0, 0, 1, 1, 1],
    })

    with pytest.raises(ValueError, match="at least 5 samples"):
        run_cross_validation(
            df,
            target_column="target",
            model_name="Logistic Regression",
            folds=5,
        )


# ---------- Visualization Tests ----------

def test_plot_confusion_matrix():
    fig = plot_confusion_matrix(
        [0, 1, 0, 1],
        [0, 1, 1, 1],
    )

    assert fig is not None
    plt.close(fig)


def test_plot_model_comparison():
    results = {
        "Logistic Regression": {
            "accuracy": 0.85,
            "precision": 0.84,
            "recall": 0.85,
            "f1_score": 0.84,
        },
        "Random Forest": {
            "accuracy": 0.92,
            "precision": 0.91,
            "recall": 0.92,
            "f1_score": 0.91,
        },
    }

    fig = plot_model_comparison(results)

    assert fig is not None
    plt.close(fig)


def test_plot_model_comparison_rejects_empty_results():
    with pytest.raises(ValueError, match="At least one"):
        plot_model_comparison({})
