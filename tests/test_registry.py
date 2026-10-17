
from sklearn.base import ClassifierMixin

from modelbench.models.registry import (
    get_model,
    get_model_names,
    get_model_registry,
)


def test_registry_contains_four_models():
    assert len(get_model_names()) == 4


def test_models_are_classifiers():
    for model in get_model_registry().values():
        assert isinstance(model, ClassifierMixin)


def test_get_model_returns_selected_model():
    model = get_model("Random Forest")
    assert isinstance(model, ClassifierMixin)


def test_invalid_model_name():
    import pytest

    with pytest.raises(ValueError):
        get_model("Unknown Model")
