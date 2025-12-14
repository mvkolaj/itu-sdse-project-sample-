# src/data_ingestion.py

import subprocess
import json
import datetime
from pathlib import Path

import pandas as pd
import mlflow

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_DATA_FILE = DATA_DIR / "raw_data.csv"
DATE_FILTERED_DATA_FILE = DATA_DIR / "date_filtered_data.csv"
DATE_LIMITS_FILE = ARTIFACTS_DIR / "date_limits.json"


def pull_data_from_dvc() -> None:
    """Pull raw dataset using DVC."""
    print("Pulling data from DVC")
    subprocess.run(
        ["dvc", "pull"],
        check=True,
        cwd=PROJECT_ROOT
    )


def filter_by_date(
    df: pd.DataFrame,
    min_date: datetime.date,
    max_date: datetime.date
) -> pd.DataFrame:
    """Filter dataframe by date range."""
    df = df.copy()
    df["date_part"] = pd.to_datetime(df["date_part"]).dt.date
    return df[
        (df["date_part"] >= min_date) &
        (df["date_part"] <= max_date)
    ]


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Pull data
    pull_data_from_dvc()

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            "raw_data.csv not found after dvc pull"
        )

    # 2. Load data
    print("Loading raw dataset")
    data = pd.read_csv(RAW_DATA_FILE)
    print(f"Total rows: {len(data)}")

    # 3. Date parameters (fixed defaults = deterministic)
    min_date = datetime.date(2024, 1, 1)
    max_date = datetime.date.today()

    # 4. Track in MLflow
    with mlflow.start_run():
        mlflow.log_param("min_date", str(min_date))
        mlflow.log_param("max_date", str(max_date))

        filtered = filter_by_date(data, min_date, max_date)
        filtered.to_csv(DATE_FILTERED_DATA_FILE, index=False)

        date_limits = {
            "requested_min_date": str(min_date),
            "requested_max_date": str(max_date),
            "actual_min_date": str(filtered["date_part"].min()),
            "actual_max_date": str(filtered["date_part"].max()),
        }

        with open(DATE_LIMITS_FILE, "w") as f:
            json.dump(date_limits, f)

        mlflow.log_artifact(DATE_LIMITS_FILE)

    print("Data ingestion complete.")


if __name__ == "__main__":
    main()
