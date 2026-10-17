
import pandas as pd
import pytest

from modelbench.models.trainer import train_model


def sample_dataset():
    return pd.DataFrame({
        "age": list(range(20, 40)),
        "city": ["Bengaluru", "Mysuru"] * 10,
        "purchased": [0, 1] * 10,
    })


def test_train_model_returns_predictions():
    result = train_model(
        sample_dataset(),
        target_column="purchased",
        model_name="Logistic Regression",
    )

    assert len(result["predictions"]) == len(result["y_test"])
    assert result["pipeline"] is not None


def test_train_model_rejects_missing_target():
    with pytest.raises(ValueError, match="was not found"):
        train_model(
            sample_dataset(),
            target_column="unknown",
            model_name="Logistic Regression",
        )


def test_train_model_rejects_invalid_test_size():
    with pytest.raises(ValueError, match="test_size"):
        train_model(
            sample_dataset(),
            target_column="purchased",
            model_name="Logistic Regression",
            test_size=1.0,
        )
