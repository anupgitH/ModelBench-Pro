import pandas as pd
from io import BytesIO


def load_csv(file) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    if file is None:
        raise ValueError("No CSV file was provided.")

    try:
        if isinstance(file, bytes):
            df = pd.read_csv(BytesIO(file))
        else:
            df = pd.read_csv(file)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Unable to parse CSV file: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    return df
