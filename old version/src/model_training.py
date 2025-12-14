# model_training.py

from pathlib import Path
from typing import Tuple, Literal

import json
import joblib
import pandas as pd
from scipy.stats import uniform, randint
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRFClassifier


ARTIFACTS_DIR = Path("artifacts")
DATA_GOLD_PATH = ARTIFACTS_DIR / "train_data_gold.csv"
COLUMNS_LIST_PATH = ARTIFACTS_DIR / "columns_list.json"



def create_dummy_cols(df: pd.DataFrame, col: str) -> pd.DataFrame:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df


def prepare_training_data(
    data_gold_path: Path = DATA_GOLD_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not data_gold_path.exists():
        raise FileNotFoundError(f"Gold data not found at {data_gold_path.resolve()}")

    data = pd.read_csv(data_gold_path)
    print(f"Training data length: {len(data)}")

    for col in ["lead_id", "customer_code", "date_part"]:
        if col in data.columns:

            data = data.drop(columns=[col])

    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
    cat_cols = [c for c in cat_cols if c in data.columns]

    cat_vars = data[cat_cols].copy()
    other_vars = data.drop(columns=cat_cols)

    for col in cat_vars.columns:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)

    data = pd.concat([other_vars, cat_vars], axis=1)

    for col in data.columns:
        try:
            data[col] = data[col].astype("float64")
        except Exception:
            pass

    if "lead_indicator" not in data.columns:
        raise ValueError(
            "Target column 'lead_indicator' not found in training data. "
            f"Available columns: {list(data.columns)}"
        )

    y = data["lead_indicator"]
    X = data.drop(columns=["lead_indicator"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.15, stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[XGBRFClassifier, dict]:
    
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    model = XGBRFClassifier(random_state=42)

    params = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
        "eval_metric": ["aucpr", "error"],
    }

    grid = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        cv=10,
        n_iter=10,
        n_jobs=-1,
        verbose=3,
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    xgb_path = ARTIFACTS_DIR / "lead_model_xgboost.json"
    best_model.save_model(str(xgb_path))

    with COLUMNS_LIST_PATH.open("w") as f:
        json.dump({"column_names": list(X_train.columns)}, f)

    print("Saved XGBoost model to:", xgb_path)
    print("Best params:", grid.best_params_)

    return best_model, grid.best_params_


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[LogisticRegression, dict]:
   
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    model = LogisticRegression()

    params = {
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "penalty": ["none", "l1", "l2", "elasticnet"],
        "C": [100, 10, 1.0, 0.1, 0.01],
    }

    grid = RandomizedSearchCV(model, params, cv=3, n_iter=10, verbose=3)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    lr_path = ARTIFACTS_DIR / "lead_model_lr.pkl"
    joblib.dump(best_model, lr_path)

    with COLUMNS_LIST_PATH.open("w") as f:
        json.dump({"column_names": list(X_train.columns)}, f)

    print("Saved Logistic Regression model to:", lr_path)
    print("Best params:", grid.best_params_)

    return best_model, grid.best_params_


def run_model_training(
    model_type: Literal["logreg", "xgboost"] = "logreg",
) -> Tuple[object, pd.DataFrame, pd.Series]:
    
    X_train, X_test, y_train, y_test = prepare_training_data()

    if model_type == "xgboost":
        model, best_params = train_xgboost(X_train, y_train)
    else:
        model, best_params = train_logistic_regression(X_train, y_train)

    print(f"Training complete. Best params: {best_params}")
    return model, X_test, y_test
