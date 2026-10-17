
import pandas as pd
import pytest

from modelbench.data.loader import load_csv
from modelbench.data.validator import validate_dataset
from modelbench.data.cleaner import clean_dataset


def test_load_csv():
    df = load_csv(b"name,age\nAlice,21\nBob,22\n")
    assert df.shape == (2, 2)


def test_empty_csv_bytes():
    with pytest.raises(ValueError):
        load_csv(b"")


def test_validate_dataset():
    df = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
    assert validate_dataset(df) == []


def test_clean_dataset():
    df = pd.DataFrame({"feature": [1, 1], "target": [0, 0]})
    cleaned = clean_dataset(df)
    assert len(cleaned) == 1
