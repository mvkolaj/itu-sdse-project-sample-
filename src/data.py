# src/data.py

from pathlib import Path
import pandas as pd

# Path to your raw data file
RAW_DATA_PATH = "data/raw/raw_data.csv"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw training data for the pipeline.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw data as a DataFrame.
    """
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Raw data not found at: {file.resolve()}")

    df = pd.read_csv(file)
    return df
