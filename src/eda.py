
"""Exploratory Data Analysis for the Heart Disease dataset."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")  # Allows plots to be saved in CI/headless environments.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_disease_clean.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET = "target"


def load_data() -> pd.DataFrame:
    """Load and validate the processed dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}\n"
            "Run download_data.py and preprocess_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("The processed dataset is empty.")

    if TARGET not in df.columns:
        raise ValueError(f"Required target column '{TARGET}' is missing.")

    if not set(df[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("The target must contain only 0 and 1.")

    return df


def save_basic_analysis(df: pd.DataFrame) -> None:
    """Save descriptive statistics, missing-value counts and summary."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df.describe().T.to_csv(REPORTS_DIR / "descriptive_statistics.csv")

    missing = df.isna().sum().rename("missing_count").to_frame()
    missing["missing_percentage"] = (
        missing["missing_count"] / len(df) * 100
    )
    missing.to_csv(REPORTS_DIR / "missing_values.csv")

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "features": int(df.shape[1] - 1),
        "missing_values_total": int(df.isna().sum().sum()),
        "target_distribution": {
            str(key): int(value)
            for key, value in df[TARGET].value_counts().sort_index().items()
        },
    }

    with open(REPORTS_DIR / "eda_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nDataset shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", missing)
    print("\nTarget distribution:\n", df[TARGET].value_counts().sort_index())
    print("\nSummary saved to reports/eda_summary.json")


def plot_class_balance(df: pd.DataFrame) -> None:
    """Visualize the distribution of the binary target."""
    counts = df[TARGET].value_counts().reindex([0, 1], fill_value=0)

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x=counts.index.astype(str), y=counts.values)
    ax.set_title("Heart Disease Class Distribution")
    ax.set_xlabel("Target (0 = absence, 1 = presence)")
    ax.set_ylabel("Number of records")

    for container in ax.containers:
        ax.bar_label(container)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_balance.png", dpi=300)
    plt.close()


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """Create histograms for all input features."""
    features = [column for column in df.columns if column != TARGET]

    df[features].hist(
        bins=20,
        figsize=(16, 12),
        edgecolor="black",
    )
    plt.suptitle("Heart Disease Feature Distributions", fontsize=16)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_distributions.png", dpi=300)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Visualize correlations between numeric columns."""
    plt.figure(figsize=(13, 10))
    correlation = df.corr(numeric_only=True)

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.4,
    )

    plt.title("Heart Disease Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()


def plot_missing_values(df: pd.DataFrame) -> None:
    """Plot the number of missing values in each column."""
    counts = df.isna().sum()
    counts = counts[counts > 0].sort_values(ascending=False)

    plt.figure(figsize=(9, 5))

    if counts.empty:
        plt.text(
            0.5, 0.5, "No missing values detected",
            ha="center", va="center",
        )
        plt.axis("off")
        plt.title("Missing Values by Column")
    else:
        sns.barplot(x=counts.values, y=counts.index)
        plt.xlabel("Number of missing values")
        plt.ylabel("Feature")
        plt.title("Missing Values by Column")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "missing_values.png", dpi=300)
    plt.close()


def run_eda() -> None:
    """Execute the full EDA workflow."""
    sns.set_theme(style="whitegrid")

    df = load_data()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    save_basic_analysis(df)
    plot_class_balance(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_missing_values(df)

    print("\nEDA completed successfully.")
    print(f"Figures saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    run_eda()
