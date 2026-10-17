
"""Download-independent preprocessing for the UCI Heart Disease dataset."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num",
]

# Nominal categorical variables in the original dataset.
CATEGORICAL_COLUMNS = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "thal",
]


def preprocess_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean raw data and produce clean and one-hot-encoded CSV files."""

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw data not found: {RAW_PATH}. "
            "Run python scripts/download_data.py first."
        )

    df = pd.read_csv(
        RAW_PATH,
        header=None,
        names=COLUMNS,
        na_values=["?"],
        skipinitialspace=True,
    )

    if df.empty or df.shape[1] != len(COLUMNS):
        raise ValueError(f"Unexpected dataset shape: {df.shape}")

    for column in COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Rows without a known target cannot be used for supervised learning.
    df = df.dropna(subset=["num"]).copy()

    if not df["num"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError("Unexpected diagnosis values.")

    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve interpretable feature codes for EDA.
    clean_path = PROCESSED_DIR / "heart_disease_clean.csv"
    df.to_csv(clean_path, index=False)

    # One-hot encode categorical features.
    # Missing feature values remain missing for later imputation.
    encoded = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLUMNS,
        dummy_na=False,
        dtype=int,
    )

    encoded_path = PROCESSED_DIR / "heart_disease_encoded.csv"
    encoded.to_csv(encoded_path, index=False)

    print(f"Clean data saved: {clean_path}")
    print(f"Encoded data saved: {encoded_path}")
    print(f"Clean shape: {df.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print("\nMissing values in clean data:")
    print(df.isna().sum())
    print("\nTarget distribution:")
    print(df["target"].value_counts().sort_index())

    return df, encoded


if __name__ == "__main__":
    preprocess_data()
