
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from modelbench.data.preprocessing import create_preprocessor
from modelbench.models.registry import get_model


def run_cross_validation(
    df: pd.DataFrame,
    target_column: str,
    model_name: str,
    folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Evaluate a classification model using stratified cross-validation."""

    if df.empty:
        raise ValueError("Dataset cannot be empty.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    if folds < 2:
        raise ValueError("Folds must be at least 2.")

    data = df.dropna(subset=[target_column]).copy()

    X = data.drop(columns=[target_column])
    y = data[target_column]

    if X.empty or X.shape[1] == 0:
        raise ValueError("Dataset must contain feature columns.")

    if y.nunique() < 2:
        raise ValueError("Target column must contain at least two classes.")

    class_counts = y.value_counts()

    if class_counts.min() < folds:
        raise ValueError(
            f"Each class needs at least {folds} samples for {folds}-fold "
            "stratified cross-validation."
        )

    pipeline = Pipeline(steps=[
        ("preprocessor", create_preprocessor(X)),
        ("model", get_model(model_name, random_state)),
    ])

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"],
        error_score="raise",
    )

    return {
        "model_name": model_name,
        "folds": folds,
        "accuracy_scores": scores["test_accuracy"].tolist(),
        "precision_scores": scores["test_precision_weighted"].tolist(),
        "recall_scores": scores["test_recall_weighted"].tolist(),
        "f1_scores": scores["test_f1_weighted"].tolist(),
        "mean_accuracy": float(scores["test_accuracy"].mean()),
        "mean_precision": float(scores["test_precision_weighted"].mean()),
        "mean_recall": float(scores["test_recall_weighted"].mean()),
        "mean_f1_score": float(scores["test_f1_weighted"].mean()),
    }
