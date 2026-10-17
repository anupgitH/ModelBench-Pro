
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


DEFAULT_HISTORY_FILE = Path("reports/experiment_history.json")


def save_experiment(
    model_name: str,
    metrics: dict,
    history_file: str | Path = DEFAULT_HISTORY_FILE,
    notes: str = "",
) -> dict:
    """Save an experiment and its evaluation metrics to a JSON history file."""

    if not model_name or not model_name.strip():
        raise ValueError("Model name cannot be empty.")

    required_metrics = ["accuracy", "precision", "recall", "f1_score"]

    for metric in required_metrics:
        if metric not in metrics:
            raise ValueError(f"Missing required metric: {metric}")

    record = {
        "experiment_id": str(uuid4()),
        "model_name": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "notes": notes,
    }

    path = Path(history_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Experiment history contains invalid JSON.") from exc

        if not isinstance(history, list):
            raise ValueError("Experiment history must contain a JSON list.")
    else:
        history = []

    history.append(record)

    # Write to a temporary file before replacing the history file.
    temp_path = path.with_name(path.name + ".tmp")

    try:
        temp_path.write_text(
            json.dumps(history, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return record


def load_experiments(
    history_file: str | Path = DEFAULT_HISTORY_FILE,
) -> list[dict]:
    """Load saved experiments, returning an empty list if no history exists."""

    path = Path(history_file)

    if not path.exists():
        return []

    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Experiment history contains invalid JSON.") from exc

    if not isinstance(history, list):
        raise ValueError("Experiment history must contain a JSON list.")

    return history
