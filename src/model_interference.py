# model_inference.py

import json
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRFClassifier


ARTIFACTS_DIR = Path("artifacts")
MODEL_LR_PATH = ARTIFACTS_DIR / "lead_model_lr.pkl"
MODEL_XGB_PATH = ARTIFACTS_DIR / "lead_model_xgboost.json"
COLUMNS_PATH = ARTIFACTS_DIR / "columns_list.json"


def load_column_order():
    if not COLUMNS_PATH.exists():
        raise FileNotFoundError("Missing artifacts/columns_list.json")

    with open(COLUMNS_PATH) as f:
        return json.load(f)["column_names"]


def load_model(model_type="logreg"):
    if model_type == "xgboost":
        if not MODEL_XGB_PATH.exists():
            raise FileNotFoundError("Missing XGBoost model .json file")

        model = XGBRFClassifier()
        model.load_model(str(MODEL_XGB_PATH))
        return model

    if not MODEL_LR_PATH.exists():
        raise FileNotFoundError("Missing Logistic Regression model .pkl file")

    return joblib.load(MODEL_LR_PATH)


def prepare_input(df: pd.DataFrame, required_cols):
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[required_cols]

    df = df.astype("float64")

    return df


def predict(input_csv, model_type="logreg", return_proba=False):
    
    model = load_model(model_type)

    required_cols = load_column_order()

    df = pd.read_csv(input_csv)

    df_prepared = prepare_input(df, required_cols)

    if return_proba:
        try:
            preds = model.predict_proba(df_prepared)[:, 1]
        except:
            preds = model.predict(df_prepared)
    else:
        preds = model.predict(df_prepared)

    print(f"\nPredictions ({model_type}):", preds)

    return preds
