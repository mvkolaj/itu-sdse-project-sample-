# inference_xgboost.py
import json
import pandas as pd
from xgboost import XGBRFClassifier


def load_model():
    model = XGBRFClassifier()
    model.load_model("artifacts/lead_model_xgboost.json")
    return model


def load_column_order():
    with open("artifacts/columns_list.json") as f:
        return json.load(f)["column_names"]


def predict(input_csv="new_data.csv"):
    model = load_model()
    required_cols = load_column_order()

    df = pd.read_csv(input_csv)

    # Ensure columns match training dataset
    df = df[required_cols]

    preds = model.predict(df)
    print("Predictions:", preds)

    return preds


# inference.py
import pandas as pd
import joblib
import json

def load_model():
    return joblib.load("artifacts/lead_model_lr.pkl")

def load_required_columns():
    with open("artifacts/columns_list.json") as f:
        return json.load(f)["column_names"]

def predict_new_data(input_csv="new_data.csv"):
    model = load_model()
    required_cols = load_required_columns()

    df = pd.read_csv(input_csv)

    # ensure correct column order
    df = df[required_cols]

    preds = model.predict(df)

    print("Predictions:", preds)
    return preds
