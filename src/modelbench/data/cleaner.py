
import pandas as pd


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy without duplicate rows or columns."""
    cleaned = df.copy()

    cleaned = cleaned.loc[:, ~cleaned.columns.duplicated()]
    cleaned = cleaned.drop_duplicates()

    return cleaned
