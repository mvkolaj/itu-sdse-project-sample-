# data_processing.py

import os
import json
import datetime
from pathlib import Path
from pprint import pprint
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler



def describe_numeric_col(x: pd.Series) -> pd.Series:
    """
    Descriptive stats for a numeric column.
    """
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"],
    )


def impute_missing_values(x: pd.Series, method: str = "mean") -> pd.Series:
    """
    Impute missing values.
    - numeric: mean or median
    - non-numeric: mode
    """
    if (x.dtype == "float64") or (x.dtype == "int64"):
        if method == "mean":
            x = x.fillna(x.mean())
        else:
            x = x.fillna(x.median())
    else:
        # most frequent value
        x = x.fillna(x.mode()[0])
    return x


# -------------------------------------------------------------------
# Main pipeline function for data processing
# -------------------------------------------------------------------

def run_data_processing(
    raw_data_path: str = ".data/raw/raw_data.csv",
    artifacts_dir: str = "./artifacts",
    min_date: Optional[str] = "2024-01-01",
    max_date: Optional[str] = "2024-01-31",
) -> pd.DataFrame:
    """
    Full data processing pipeline.

    Steps:
    1. Ensure artifacts directory exists.
    2. Load raw data.
    3. Filter by date range.
    4. Feature selection (drop unused columns).
    5. Clean invalid rows (target/ID empty, filter source == 'signup').
    6. Cast selected columns to object.
    7. Split into continuous and categorical vars.
    8. Clip outliers in continuous vars.
    9. Impute missing values (cat + cont).
    10. Scale continuous vars and save scaler.
    11. Combine cat + cont into one dataframe.
    12. Save drift columns + training_data.csv.
    13. Bin 'source' into 'bin_source'.
    14. Save train_data_gold.csv.
    15. Return the final processed dataframe.
    """
    # 1. Ensure artifacts directory exists
    os.makedirs(artifacts_dir, exist_ok=True)
    print(f"Created artifacts directory at {artifacts_dir}")

    # 2. Load raw data
    print("Loading training data...")
    data = pd.read_csv(raw_data_path)
    print("Total rows:", len(data))

    # 3. Filter by date range
    if max_date:
        max_dt = pd.to_datetime(max_date).date()
    else:
        max_dt = pd.to_datetime(datetime.datetime.now().date()).date()

    if min_date:
        min_dt = pd.to_datetime(min_date).date()
    else:
        min_dt = pd.to_datetime("1900-01-01").date()

    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_dt) & (data["date_part"] <= max_dt)]

    min_dt_actual = data["date_part"].min()
    max_dt_actual = data["date_part"].max()
    date_limits = {"min_date": str(min_dt_actual), "max_date": str(max_dt_actual)}

    with open(os.path.join(artifacts_dir, "date_limits.json"), "w") as f:
        json.dump(date_limits, f)
    print("Saved date_limits.json")

    # 4. Feature selection (drop unused columns)
    data = data.drop(
        [
            "is_active",
            "marketing_consent",
            "first_booking",
            "existing_customer",
            "last_seen",
        ],
        axis=1,
    )

    data = data.drop(
        ["domain", "country", "visited_learn_more_before_booking", "visited_faq"],
        axis=1,
    )

    # 5. Data cleaning: remove rows with empty target / invalid IDs, filter source
    data["lead_indicator"].replace("", np.nan, inplace=True)
    data["lead_id"].replace("", np.nan, inplace=True)
    data["customer_code"].replace("", np.nan, inplace=True)

    data = data.dropna(axis=0, subset=["lead_indicator"])
    data = data.dropna(axis=0, subset=["lead_id"])

    # Filter: only signup as source
    data = data[data["source"] == "signup"]

    # Optional: print target distribution
    result = data["lead_indicator"].value_counts(normalize=True)
    print("Target value counter")
    for val, n in zip(result.index, result):
        print(val, ": ", n)

    # 6. Set some columns explicitly as object
    obj_cols = [
        "lead_id",
        "lead_indicator",
        "customer_group",
        "onboarding",
        "source",
        "customer_code",
    ]
    for col in obj_cols:
        if col in data.columns:
            data[col] = data[col].astype("object")
            print(f"Changed {col} to object type")

    # 7. Separate continuous and categorical columns
    cont_vars = data.loc[:, (data.dtypes == "float64") | (data.dtypes == "int64")]
    cat_vars = data.loc[:, (data.dtypes == "object")]

    print("\nContinuous columns:\n")
    pprint(list(cont_vars.columns), indent=4)
    print("\nCategorical columns:\n")
    pprint(list(cat_vars.columns), indent=4)

    # 8. Outliers: clip continuous vars at mean ± 2 std
    if not cont_vars.empty:
        cont_vars = cont_vars.apply(
            lambda x: x.clip(
                lower=x.mean() - 2 * x.std(),
                upper=x.mean() + 2 * x.std(),
            )
        )
        outlier_summary = cont_vars.apply(describe_numeric_col).T
        outlier_summary.to_csv(os.path.join(artifacts_dir, "outlier_summary.csv"))
        print("Saved outlier_summary.csv")

    # 9. Impute missing values

    # 9a. Categorical imputation: save mode values
    if not cat_vars.empty:
        cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
        cat_missing_impute.to_csv(
            os.path.join(artifacts_dir, "cat_missing_impute.csv")
        )
        print("Saved cat_missing_impute.csv")

        if "customer_code" in cat_vars.columns:
            cat_vars.loc[cat_vars["customer_code"].isna(), "customer_code"] = "None"

        cat_vars = cat_vars.apply(impute_missing_values)

    # 9b. Continuous vars missing values
    if not cont_vars.empty:
        cont_vars = cont_vars.apply(impute_missing_values)

    # 10. Data standardisation (MinMax scaling)
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    if not cont_vars.empty:
        scaler = MinMaxScaler()
        scaler.fit(cont_vars)
        joblib.dump(value=scaler, filename=scaler_path)
        print(f"Saved scaler in {scaler_path}")
        cont_vars = pd.DataFrame(
            scaler.transform(cont_vars), columns=cont_vars.columns
        )

    # 11. Combine categorical & continuous into single dataframe
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)
    print(f"Data cleansed and combined.\nRows: {len(data)}")

    # 12. Data drift artifact: save columns and training_data.csv
    data_columns = list(data.columns)
    with open(os.path.join(artifacts_dir, "columns_drift.json"), "w+") as f:
        json.dump(data_columns, f)
    print("Saved columns_drift.json")

    data.to_csv(os.path.join(artifacts_dir, "training_data.csv"), index=False)
    print("Saved training_data.csv")

    # 13. Binning object columns ('source' -> 'bin_source')
    if "source" in data.columns:
        mapping = {
            "li": "socials",
            "fb": "socials",
            "organic": "group1",
            "signup": "group1",
        }

        # default group
        data["bin_source"] = data["source"].map(mapping).fillna("Others")

    # 14. Save gold dataset (train_data_gold.csv)
    gold_path = os.path.join(artifacts_dir, "train_data_gold.csv")
    data.to_csv(gold_path, index=False)
    print(f"✔ Saved train_data_gold.csv at {gold_path}")

    # 15. Return final processed dataframe
    return data