
"""Download the original Cleveland Heart Disease dataset from UCI."""

from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)

OUTPUT_PATH = RAW_DIR / "heart_disease.csv"


def download_data() -> None:
    """Download and save the raw dataset without modifying its contents."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from: {DATA_URL}")
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()

    if not response.content.strip():
        raise ValueError("The downloaded dataset is empty.")

    # Save the source response as-is; preprocessing happens elsewhere.
    OUTPUT_PATH.write_bytes(response.content)

    print(f"Raw dataset saved to: {OUTPUT_PATH}")
    print(f"Downloaded size: {len(response.content):,} bytes")


if __name__ == "__main__":
    download_data()
