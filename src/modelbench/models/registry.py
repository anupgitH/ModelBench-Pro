
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def get_model_registry(random_state: int = 42) -> dict:
    """Return supported classification models."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Support Vector Machine": SVC(
            probability=True,
            random_state=random_state,
        ),
    }


def get_model_names() -> list[str]:
    """Return available model names."""
    return list(get_model_registry().keys())


def get_model(name: str, random_state: int = 42):
    """Return a fresh instance of the selected model."""

    registry = get_model_registry(random_state)

    if name not in registry:
        raise ValueError(f"Unsupported model: {name}")

    return registry[name]
