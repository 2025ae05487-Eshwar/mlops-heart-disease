"""Train, tune, evaluate, compare and track heart disease classifiers."""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.preprocessing import FEATURE_COLUMNS, build_preprocessor
from src.models.logistic_regression.model import (
    build_model as build_logistic_regression,
)
from src.models.random_forest.model import (
    build_model as build_random_forest,
)
from src.mlflow_tracking import (
    configure_mlflow,
    start_model_run,
    log_parameters,
    log_metrics,
    log_artifacts,
    log_sklearn_model,
    get_active_run_id,
)


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "processed" / "heart_disease_clean.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

# Metrics used during hyperparameter tuning.
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
}


# ---------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------

def load_data():
    """Load the cleaned dataset and validate its columns and target."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run scripts/download_data.py and "
            "scripts/preprocess_data.py first."
        )

    data = pd.read_csv(DATA_PATH)

    if "target" not in data.columns:
        raise ValueError(
            "Expected a binary target column named 'target'."
        )

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing expected feature columns: {missing_features}"
        )

    data = data.dropna(subset=["target"]).copy()

    X = data[FEATURE_COLUMNS].copy()
    y = pd.to_numeric(data["target"], errors="raise").astype(int)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    if y.nunique() != 2:
        raise ValueError(
            "The dataset must contain both target classes: 0 and 1."
        )

    print("\nDataset loaded successfully.")
    print(f"Number of rows: {len(data)}")
    print(f"Number of features before encoding: {X.shape[1]}")
    print("\nTarget distribution:")
    print(y.value_counts().sort_index().rename("count"))

    return X, y


# ---------------------------------------------------------
# Model and hyperparameter configuration
# ---------------------------------------------------------

def build_pipeline(classifier):
    """Combine feature preprocessing and classification."""

    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])


def get_model_configs():
    """Define classifiers and their hyperparameter search spaces."""

    return {
        "Logistic Regression": {
            "pipeline": build_pipeline(build_logistic_regression()),
            "parameters": {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        "Random Forest": {
            "pipeline": build_pipeline(build_random_forest()),
            "parameters": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_split": [2, 5],
                "classifier__class_weight": [None, "balanced"],
            },
        },
    }


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def calculate_test_metrics(model, X_test, y_test):
    """Calculate evaluation metrics on the held-out test set."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "test_accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "test_precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "test_recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "test_f1": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "test_roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
    }


def save_model_plots(name, model, X_test, y_test, metrics):
    """Save confusion matrix, ROC curve and test-metric chart."""

    safe_name = name.lower().replace(" ", "_")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # 1. Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=[0, 1],
        display_labels=["No disease (0)", "Disease (1)"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )

    ax.set_title(f"{name}: Confusion Matrix")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / f"{safe_name}_confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 2. ROC curve
    fig, ax = plt.subplots(figsize=(6, 5))

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
        name=name,
        ax=ax,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random baseline",
    )

    ax.set_title(f"{name}: ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / f"{safe_name}_roc_curve.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 3. Test-metric chart
    metric_labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    ]

    metric_keys = [
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
    ]

    values = [metrics[key] for key in metric_keys]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(metric_labels, values)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{name}: Held-Out Test Performance")
    ax.bar_label(bars, fmt="%.3f")
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / f"{safe_name}_metrics.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------
# Main training, evaluation and MLflow tracking
# ---------------------------------------------------------

def main():
    """Train, tune, evaluate, compare and track both classifiers."""

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Configure the local MLflow tracking directory.
    configure_mlflow()

    X, y = load_data()

    # Reserve a stratified test set. Do not use it for tuning.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTraining rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Cross-validation folds: {N_SPLITS}")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    best_cv_roc_auc = -1.0
    best_model = None
    best_model_name = None
    best_parameters = None

    # Train and track each candidate model in its own MLflow run.
    for model_name, config in get_model_configs().items():

        print("\n" + "=" * 70)
        print(f"MODEL: {model_name}")
        print("=" * 70)

        safe_name = model_name.lower().replace(" ", "_")

        with start_model_run(
            model_name,
            {
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "cv_folds": N_SPLITS,
                "selection_metric": "roc_auc",
                "search_space": config["parameters"],
            },
        ):

            # Hyperparameter tuning is performed on training data only.
            search = GridSearchCV(
                estimator=config["pipeline"],
                param_grid=config["parameters"],
                scoring=SCORING,
                refit="roc_auc",
                cv=cv,
                n_jobs=-1,
                return_train_score=False,
                error_score="raise",
            )

            search.fit(X_train, y_train)

            model = search.best_estimator_
            best_index = search.best_index_

            # Cross-validation metrics for the selected parameters.
            cv_metrics = {
                "cv_accuracy": float(
                    search.cv_results_[
                        "mean_test_accuracy"
                    ][best_index]
                ),
                "cv_precision": float(
                    search.cv_results_[
                        "mean_test_precision"
                    ][best_index]
                ),
                "cv_recall": float(
                    search.cv_results_[
                        "mean_test_recall"
                    ][best_index]
                ),
                "cv_roc_auc": float(
                    search.cv_results_[
                        "mean_test_roc_auc"
                    ][best_index]
                ),
            }

            # Evaluate the selected estimator on the held-out test set.
            test_metrics = calculate_test_metrics(
                model,
                X_test,
                y_test,
            )

            print(f"\nBest hyperparameters: {search.best_params_}")

            print("\nCross-validation metrics:")
            for metric, value in cv_metrics.items():
                print(f"  {metric}: {value:.4f}")

            print("\nHeld-out test metrics:")
            for metric, value in test_metrics.items():
                print(f"  {metric}: {value:.4f}")

            # Log the selected hyperparameters and evaluation metrics.
            log_parameters(
                search.best_params_,
                prefix="best_",
            )

            log_metrics({
                **cv_metrics,
                **test_metrics,
            })

            # Save and print the classification report.
            report = classification_report(
                y_test,
                model.predict(X_test),
                labels=[0, 1],
                target_names=[
                    "No heart disease",
                    "Heart disease",
                ],
                zero_division=0,
            )

            print("\nTest classification report:")
            print(report)

            report_path = (
                REPORTS_DIR
                / f"classification_report_{safe_name}.txt"
            )

            report_path.write_text(
                report,
                encoding="utf-8",
            )

            # Generate the three model-specific plots.
            save_model_plots(
                model_name,
                model,
                X_test,
                y_test,
                test_metrics,
            )

            # Log reports and plots as MLflow artifacts.
            log_artifacts([
                report_path,
                FIGURES_DIR / f"{safe_name}_confusion_matrix.png",
                FIGURES_DIR / f"{safe_name}_roc_curve.png",
                FIGURES_DIR / f"{safe_name}_metrics.png",
            ])

            # Log the fitted scikit-learn pipeline.
            log_sklearn_model(
                model,
                artifact_name="model",
            )

            run_id = get_active_run_id()
            print(f"MLflow run ID: {run_id}")

            # Retain metrics for the final comparison.
            results.append({
                "model": model_name,
                **cv_metrics,
                **test_metrics,
                "best_parameters": json.dumps(
                    search.best_params_
                ),
            })

        # Select the final model using cross-validation ROC-AUC.
        if cv_metrics["cv_roc_auc"] > best_cv_roc_auc:
            best_cv_roc_auc = cv_metrics["cv_roc_auc"]
            best_model = model
            best_model_name = model_name
            best_parameters = search.best_params_

    if best_model is None:
        raise RuntimeError("Training did not produce a valid model.")

    # -----------------------------------------------------
    # Final model comparison
    # -----------------------------------------------------

    comparison = pd.DataFrame(results).sort_values(
        "cv_roc_auc",
        ascending=False,
    )

    comparison_path = REPORTS_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    columns_to_show = [
        "model",
        "cv_accuracy",
        "cv_precision",
        "cv_recall",
        "cv_roc_auc",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
    ]

    print(
        comparison[columns_to_show].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # Plot comparison of test-set metrics.
    chart_columns = [
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
    ]

    chart_data = comparison.set_index("model")[chart_columns]
    chart_data.columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    ]

    ax = chart_data.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title("Logistic Regression vs Random Forest")
    ax.set_ylabel("Test score")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="lower right")

    plt.tight_layout()

    comparison_plot = FIGURES_DIR / "model_comparison.png"
    plt.savefig(
        comparison_plot,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Plot CV ROC-AUC used for model selection.
    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        comparison["model"],
        comparison["cv_roc_auc"],
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean five-fold ROC-AUC")
    ax.set_title("Model Selection: Cross-Validation ROC-AUC")
    ax.tick_params(axis="x", rotation=10)
    ax.bar_label(bars, fmt="%.3f")

    fig.tight_layout()

    cv_plot = FIGURES_DIR / "cv_roc_auc_comparison.png"
    fig.savefig(
        cv_plot,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # -----------------------------------------------------
    # Save selected model and metadata
    # -----------------------------------------------------

    model_path = ARTIFACTS_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)

    selected_result = next(
        result
        for result in results
        if result["model"] == best_model_name
    )

    metadata = {
        "selected_model": best_model_name,
        "selection_metric": (
            "mean five-fold cross-validation ROC-AUC"
        ),
        "best_cv_roc_auc": best_cv_roc_auc,
        "best_parameters": best_parameters,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": "target",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "cv_folds": N_SPLITS,
        "selected_model_metrics": selected_result,
    }

    metadata_path = ARTIFACTS_DIR / "model_info.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Track final model selection in a separate MLflow run
    # -----------------------------------------------------

    with start_model_run(
        "Final Model Selection",
        {
            "selected_model": best_model_name,
            "selection_metric": "cv_roc_auc",
            "cv_folds": N_SPLITS,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
        },
    ):

        log_parameters(
            best_parameters,
            prefix="selected_model_",
        )

        log_metrics({
            "best_cv_roc_auc": best_cv_roc_auc,
            **{
                key: value
                for key, value in selected_result.items()
                if key.startswith("test_")
                or key.startswith("cv_")
            },
        })

        # Log final comparison files, metadata and plots.
        log_artifacts([
            comparison_path,
            metadata_path,
            comparison_plot,
            cv_plot,
        ])

        # Log the selected fitted pipeline.
        log_sklearn_model(
            best_model,
            artifact_name="selected_model",
        )

        print(
            "Final selection MLflow run ID: "
            f"{get_active_run_id()}"
        )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL MODEL SELECTION")
    print("=" * 70)

    print(f"Selected model: {best_model_name}")
    print(f"Mean CV ROC-AUC: {best_cv_roc_auc:.4f}")
    print(f"Best parameters: {best_parameters}")

    print(f"\nSaved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved comparison: {comparison_path}")
    print(f"Saved plots: {FIGURES_DIR}")
    print(f"MLflow tracking directory: {ROOT_DIR / 'mlruns'}")


if __name__ == "__main__":
    main()