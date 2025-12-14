import argparse
import datetime
import json
import subprocess
from pathlib import Path

import pandas as pd
import mlflow

from paths import (
    PROJECT_ROOT,
    RAW_DATA_FILE,
    DATE_FILTERED_DATA_FILE,
    DATE_LIMITS_FILE,
)

def pull_data_from_dvc():
    dvc_dir = PROJECT_ROOT / ".dvc"

    if not dvc_dir.exists():
        print("No .dvc directory found — skipping DVC pull.")
        return

    try:
        subprocess.run(
            ["dvc", "update", "data/raw_data.csv.dvc"],
            check=True,
            cwd=PROJECT_ROOT,
        )
        subprocess.run(
            ["dvc", "pull"],
            check=True,
            cwd=PROJECT_ROOT,
        )
        print("DVC pull completed.")
    except Exception as exc:
        print("DVC pull failed — continuing without it.")
        print(f"Reason: {exc}")


def parse_date_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_date", type=str, default=None)
    parser.add_argument("--max_date", type=str, default=None)
    args = parser.parse_args()

    min_date = (
        pd.to_datetime(args.min_date).date()
        if args.min_date
        else pd.to_datetime("2024-01-01").date()
    )

    max_date = (
        pd.to_datetime(args.max_date).date()
        if args.max_date
        else datetime.datetime.now().date()
    )

    return min_date, max_date


def filter_by_date(
    df: pd.DataFrame,
    min_date: datetime.date,
    max_date: datetime.date,
) -> pd.DataFrame:
    df = df.copy()
    df["date_part"] = pd.to_datetime(df["date_part"]).dt.date
    return df[
        (df["date_part"] >= min_date)
        & (df["date_part"] <= max_date)
    ]


def main():
    print("Pulling data from DVC")
    pull_data_from_dvc()

    print("Loading raw dataset")
    data = pd.read_csv(RAW_DATA_FILE)
    print("Total rows:", len(data))

    min_date, max_date = parse_date_args()

    DATE_FILTERED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATE_LIMITS_FILE.parent.mkdir(parents=True, exist_ok=True)

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
