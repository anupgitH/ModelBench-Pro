
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred):
    """Plot a confusion matrix for classification predictions."""

    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks(range(len(matrix)))
    ax.set_yticks(range(len(matrix)))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center",
                color="black",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return fig


def plot_model_comparison(results: dict):
    """Compare model evaluation scores using a bar chart.

    Expected format:
    {
        "Logistic Regression": {"accuracy": 0.90, "f1_score": 0.88},
        "Random Forest": {"accuracy": 0.95, "f1_score": 0.93}
    }
    """

    if not results:
        raise ValueError("At least one model result is required.")

    model_names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1_score"]

    available_metrics = [
        metric
        for metric in metrics
        if all(metric in result for result in results.values())
    ]

    if not available_metrics:
        raise ValueError("No common evaluation metrics were provided.")

    values = np.array([
        [results[name][metric] for metric in available_metrics]
        for name in model_names
    ])

    x = np.arange(len(model_names))
    width = 0.8 / len(available_metrics)

    fig, ax = plt.subplots(figsize=(10, 5))

    for index, metric in enumerate(available_metrics):
        ax.bar(
            x + (index - (len(available_metrics) - 1) / 2) * width,
            values[:, index],
            width,
            label=metric.replace("_", " ").title(),
        )

    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    return fig
