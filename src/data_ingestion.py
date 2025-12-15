# src/data_ingestion.py

import os
import json
import datetime
import subprocess
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_DATA_FILE = DATA_DIR / "raw_data.csv"
FILTERED_DATA_FILE = ARTIFACTS_DIR / "date_filtered_data.csv"
DATE_METADATA_FILE = ARTIFACTS_DIR / "date_limits.json"

RAW_DATA_URL = (
    "https://raw.githubusercontent.com/"
    "Jeppe-T-K/itu-sdse-project-data/main/raw_data.csv"
)

def pull_with_dvc():
    subprocess.run(
        ["dvc", "update", "data/raw_data.csv.dvc"],
        check=True, cwd=PROJECT_ROOT)
    subprocess.run(
        ["dvc", "pull"], 
        check=True, cwd=PROJECT_ROOT)
    return True

def download_directly() -> None:
    print("Downloading raw dataset directly (CI fallback)")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    urlretrieve(RAW_DATA_URL, RAW_DATA_FILE)


def filter_dataset_by_date(
    df: pd.DataFrame,
    min_date: datetime.date,
    max_date: datetime.date
) -> pd.DataFrame:
    df = df.copy()
    df["date_part"] = pd.to_datetime(df["date_part"]).dt.date
    return df[
        (df["date_part"] >= min_date)
        & (df["date_part"] <= max_date)
    ]

def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Data ingestion started")

    pulled = pull_with_dvc()
    if not pulled:
        if "GITHUB_ACTIONS" in os.environ:
            download_directly()
        else:
            raise RuntimeError("DVC pull failed locally")

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError("raw_data.csv not available")

    data = pd.read_csv(RAW_DATA_FILE)
    print(f"Loaded dataset with {len(data)} rows")

    min_date = datetime.date(2024, 1, 1)
    max_date = datetime.date.today()

    mlflow.set_experiment("data-ingestion")

    with mlflow.start_run(run_name="date_filtering"):
        mlflow.log_param("min_date", min_date.isoformat())
        mlflow.log_param("max_date", max_date.isoformat())

        filtered = filter_dataset_by_date(data, min_date, max_date)

        filtered.to_csv(FILTERED_DATA_FILE, index=False)
        mlflow.log_artifact(FILTERED_DATA_FILE)

        metadata = {
            "requested_min_date": min_date.isoformat(),
            "requested_max_date": max_date.isoformat(),
            "actual_min_date": str(filtered["date_part"].min()),
            "actual_max_date": str(filtered["date_part"].max()),
            "rows_after_filtering": len(filtered),
        }

        with open(DATE_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        mlflow.log_artifact(DATE_METADATA_FILE)

    print("Data ingestion completed successfully")


if __name__ == "__main__":
    main()
# src/data_ingestion.py

import os
import json
import datetime
import subprocess
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_DATA_FILE = DATA_DIR / "raw_data.csv"
FILTERED_DATA_FILE = ARTIFACTS_DIR / "date_filtered_data.csv"
DATE_METADATA_FILE = ARTIFACTS_DIR / "date_limits.json"

RAW_DATA_URL = (
    "https://raw.githubusercontent.com/"
    "Jeppe-T-K/itu-sdse-project-data/main/raw_data.csv"
)


def pull_with_dvc() -> bool:
    try:
        subprocess.run(
            ["dvc", "pull"],
            check=True,
            cwd=PROJECT_ROOT
        )
        return True
    except subprocess.CalledProcessError:
        return False


def download_directly() -> None:
    print("Downloading raw dataset directly (CI fallback)")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    urlretrieve(RAW_DATA_URL, RAW_DATA_FILE)


def filter_dataset_by_date(
    df: pd.DataFrame,
    min_date: datetime.date,
    max_date: datetime.date
) -> pd.DataFrame:
    df = df.copy()
    df["date_part"] = pd.to_datetime(df["date_part"]).dt.date
    return df[
        (df["date_part"] >= min_date)
        & (df["date_part"] <= max_date)
    ]


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("▶ Data ingestion started")

    pulled = pull_with_dvc()
    if not pulled:
        if "GITHUB_ACTIONS" in os.environ:
            download_directly()
        else:
            raise RuntimeError("DVC pull failed locally")

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError("raw_data.csv not available")

    data = pd.read_csv(RAW_DATA_FILE)
    print(f"Loaded dataset with {len(data)} rows")

    min_date = datetime.date(2024, 1, 1)
    max_date = datetime.date.today()

    mlflow.set_experiment("data-ingestion")

    with mlflow.start_run(run_name="date_filtering"):
        mlflow.log_param("min_date", min_date.isoformat())
        mlflow.log_param("max_date", max_date.isoformat())

        filtered = filter_dataset_by_date(data, min_date, max_date)

        filtered.to_csv(FILTERED_DATA_FILE, index=False)
        mlflow.log_artifact(FILTERED_DATA_FILE)

        metadata = {
            "requested_min_date": min_date.isoformat(),
            "requested_max_date": max_date.isoformat(),
            "actual_min_date": str(filtered["date_part"].min()),
            "actual_max_date": str(filtered["date_part"].max()),
            "rows_after_filtering": len(filtered),
        }

        with open(DATE_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        mlflow.log_artifact(DATE_METADATA_FILE)

    print("Data ingestion completed successfully")


if __name__ == "__main__":
    main()
