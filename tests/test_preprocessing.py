
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from modelbench.data.preprocessing import create_preprocessor


def test_preprocessor_handles_missing_and_categorical_values():
    X = pd.DataFrame({
        "age": [20, None, 30, 40],
        "city": ["Bengaluru", "Mysuru", None, "Bengaluru"],
    })

    y = [0, 1, 0, 1]

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor(X)),
        ("model", LogisticRegression()),
    ])

    pipeline.fit(X, y)
    predictions = pipeline.predict(
        pd.DataFrame({
            "age": [25],
            "city": ["Delhi"],
        })
    )

    assert len(predictions) == 1


def test_preprocessor_rejects_empty_features():
    import pytest

    X = pd.DataFrame()

    with pytest.raises(ValueError):
        create_preprocessor(X)
