import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data_imputation import impute_series
from paths import (
    DATE_FILTERED_DATA_FILE,
    OUTLIER_SUMMARY_FILE,
    CAT_MISSING_IMPUTE_FILE,
    SCALER_FILE,
    COLUMNS_DRIFT_FILE,
    TRAINING_DATA_FILE,
    TRAIN_GOLD_FILE,
)


def _numeric_summary(series: pd.Series) -> pd.Series:
    return pd.Series(
        [
            series.count(),
            series.isnull().sum(),
            series.mean(),
            series.min(),
            series.max(),
        ],
        index=["Count", "Missing", "Mean", "Min", "Max"],
    )


def clean_base_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [
        "is_active",
        "marketing_consent",
        "first_booking",
        "existing_customer",
        "last_seen",
        "domain",
        "country",
        "visited_learn_more_before_booking",
        "visited_faq",
    ]
    df = df.drop(columns=drop_cols)

    for col in ["lead_indicator", "lead_id", "customer_code"]:
        df[col].replace("", np.nan, inplace=True)

    df = df.dropna(subset=["lead_indicator", "lead_id"])
    df = df[df.source == "signup"]
    return df


def split_feature_types(df: pd.DataFrame):
    categorical_cols = [
        "lead_id",
        "lead_indicator",
        "customer_group",
        "onboarding",
        "source",
        "customer_code",
    ]
    df[categorical_cols] = df[categorical_cols].astype("object")

    continuous = df.select_dtypes(include=["float64", "int64"])
    categorical = df.select_dtypes(include=["object"])

    return categorical, continuous


def cap_outliers(continuous: pd.DataFrame) -> pd.DataFrame:
    capped = continuous.apply(
        lambda x: x.clip(
            lower=x.mean() - 2 * x.std(),
            upper=x.mean() + 2 * x.std(),
        )
    )
    capped.apply(_numeric_summary).T.to_csv(
        OUTLIER_SUMMARY_FILE, index=False
    )
    return capped


def impute_features(
    categorical: pd.DataFrame,
    continuous: pd.DataFrame,
):
    categorical.mode(dropna=True).to_csv(
        CAT_MISSING_IMPUTE_FILE, index=False
    )

    continuous = continuous.apply(impute_series)

    categorical.loc[
        categorical["customer_code"].isna(),
        "customer_code",
    ] = "None"
    categorical = categorical.apply(impute_series)

    return categorical, continuous


def scale_continuous_features(continuous: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    scaler.fit(continuous)
    joblib.dump(scaler, SCALER_FILE)

    return pd.DataFrame(
        scaler.transform(continuous),
        columns=continuous.columns,
    )


def combine_and_record_columns(
    categorical: pd.DataFrame,
    continuous: pd.DataFrame,
) -> pd.DataFrame:
    data = pd.concat(
        [
            categorical.reset_index(drop=True),
            continuous.reset_index(drop=True),
        ],
        axis=1,
    )

    with open(COLUMNS_DRIFT_FILE, "w") as f:
        json.dump(list(data.columns), f)

    data.to_csv(TRAINING_DATA_FILE, index=False)
    return data


def bin_source_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {
        "li": "socials",
        "fb": "socials",
        "organic": "group1",
        "signup": "group1",
    }
    df["bin_source"] = df["source"].map(mapping).fillna("Others")
    return df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_base_data(df)
    categorical, continuous = split_feature_types(df)
    continuous = cap_outliers(continuous)
    categorical, continuous = impute_features(categorical, continuous)
    continuous = scale_continuous_features(continuous)
    combined = combine_and_record_columns(categorical, continuous)
    final = bin_source_feature(combined)

    final.to_csv(TRAIN_GOLD_FILE, index=False)
    return final


if __name__ == "__main__":
    raw = pd.read_csv(DATE_FILTERED_DATA_FILE)
    run_feature_engineering(raw)
