
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from modelbench.data.preprocessing import create_preprocessor
from modelbench.models.registry import get_model


def train_model(
    df: pd.DataFrame,
    target_column: str,
    model_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train a classification model and return its test data and predictions."""

    if df.empty:
        raise ValueError("Dataset cannot be empty.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    data = df.dropna(subset=[target_column]).copy()

    if len(data) < 5:
        raise ValueError("At least 5 valid rows are required for training.")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    if X.shape[1] == 0:
        raise ValueError("Dataset must contain at least one feature column.")

    if y.nunique() < 2:
        raise ValueError("Target column must contain at least two classes.")

    # Stratification preserves class proportions when possible.
    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError as exc:
        raise ValueError(f"Unable to split dataset: {exc}") from exc

    pipeline = Pipeline(steps=[
        ("preprocessor", create_preprocessor(X_train)),
        ("model", get_model(model_name, random_state)),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
    }
