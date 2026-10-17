
import pandas as pd


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Return validation errors. An empty list means basic checks passed."""
    errors = []

    if df.empty:
        errors.append("Dataset is empty.")

    if df.columns.empty:
        errors.append("Dataset has no columns.")

    if df.columns.duplicated().any():
        errors.append("Dataset contains duplicate column names.")

    if len(df.columns) < 2:
        errors.append("Dataset needs at least two columns.")

    if len(df) < 2:
        errors.append("Dataset needs at least two rows.")

    return errors
