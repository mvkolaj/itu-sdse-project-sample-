import pandas as pd


def impute_series(
    series: pd.Series,
    numeric_strategy: str = "mean",
) -> pd.Series:

    if series.dtype in ("float64", "int64"):
        if numeric_strategy == "median":
            return series.fillna(series.median())
        return series.fillna(series.mean())

    return series.fillna(series.mode().iloc[0])
