"""MLflow tracking utilities using a local SQLite backend."""

import json
from pathlib import Path

import mlflow
import mlflow.sklearn


ROOT_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = ROOT_DIR / "mlflow.db"
TRACKING_URI = f"sqlite:///{DATABASE_PATH.as_posix()}"
EXPERIMENT_NAME = "Heart Disease Classification"


def configure_mlflow():
    """Configure local SQLite-based MLflow tracking."""

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def start_model_run(model_name, parameters):
    """Start an MLflow run and log its initial parameters and tags."""

    configure_mlflow()

    run = mlflow.start_run(run_name=model_name)

    mlflow.set_tags({
        "model_type": model_name,
        "dataset": "UCI Heart Disease",
        "task": "binary_classification",
    })

    log_parameters(parameters)

    return run


def log_parameters(parameters, prefix=""):
    """Log parameters, converting complex values to strings."""

    formatted = {}

    for key, value in parameters.items():
        name = f"{prefix}{key}"

        if isinstance(value, (dict, list, tuple)):
            formatted[name] = json.dumps(value)
        elif value is None:
            formatted[name] = "None"
        else:
            formatted[name] = value

    if formatted:
        mlflow.log_params(formatted)


def log_metrics(metrics):
    """Log numeric metrics to the active MLflow run."""

    valid_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if value is not None
    }

    if valid_metrics:
        mlflow.log_metrics(valid_metrics)


def log_artifact(path):
    """Log one file as an MLflow artifact."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"MLflow artifact not found: {path}"
        )

    mlflow.log_artifact(str(path))


def log_artifacts(paths):
    """Log multiple files as MLflow artifacts."""

    for path in paths:
        log_artifact(path)


def log_sklearn_model(model, artifact_name="model"):
    """Log a fitted scikit-learn model using cloudpickle."""

    mlflow.sklearn.log_model(
        sk_model=model,
        name=artifact_name,
        serialization_format="cloudpickle",
    )

def get_active_run_id():
    """Return the active run ID, if a run exists."""

    run = mlflow.active_run()

    return run.info.run_id if run else None