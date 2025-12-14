import datetime
from pathlib import Path
import pandas as pd
import joblib
import mlflow

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform, randint
from xgboost import XGBRFClassifier

from paths import (
    TRAIN_GOLD_FILE,
    XGBOOST_MODEL_FILE,
    LR_MODEL_FILE,
    X_TEST_FILE,
    Y_TEST_FILE,
)
from model_adapters import LogisticRegressionAdapter


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(["lead_id", "customer_code", "date_part"], axis=1)
    categorical = ["customer_group", "onboarding", "bin_source", "source"]

    for col in categorical:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

    return df.astype("float64")

def train_single_model(
    model_cls,
    params,
    X_train,
    y_train,
    model_path: Path,
    run_name: str,
    autolog_module: str,
):
    getattr(mlflow, autolog_module).autolog(log_models=False)

    model = model_cls(random_state=42)
    grid = RandomizedSearchCV(
        model, params, n_iter=10, cv=3, verbose=2
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    if isinstance(best, LogisticRegression):
        mlflow.pyfunc.log_model("model", python_model=lr_wrapper(best))
        joblib.dump(best, model_path)
    else:
        best.save_model(str(model_path))

def main():
    data = pd.read_csv(TRAIN_GOLD_FILE)
    data = encode_categoricals(data)

    y = data["lead_indicator"]
    X = data.drop(columns=["lead_indicator"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    X_test.to_csv(X_TEST_FILE, index=False)
    y_test.to_csv(Y_TEST_FILE, index=False)

    experiment_name = datetime.datetime.now().strftime("%Y_%m_%d")
    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(
        experiment_name
    ).experiment_id

    param_xgb = {
        "learning_rate": uniform(0.01, 0.3),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
    }

    param_lr = {
        "solver": ["lbfgs", "liblinear"],
        "C": [0.1, 1, 10],
    }

    with mlflow.start_run(experiment_id=experiment_id, run_name="xgb"):
        train_single_model(
            XGBRFClassifier, param_xgb,
            X_train, y_train, XGBOOST_MODEL_FILE,
            "xgb", "xgboost"
        )

    with mlflow.start_run(experiment_id=experiment_id, run_name="lr"):
        train_single_model(
            LogisticRegression, param_lr,
            X_train, y_train, LR_MODEL_FILE,
            "lr", "sklearn"
        )

if __name__ == "__main__":
    main()
