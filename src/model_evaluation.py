import json
import time
from typing import Tuple, Optional

import pandas as pd
import numpy as np
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBRFClassifier

from paths import (
    X_TEST_FILE,
    Y_TEST_FILE,
    LR_MODEL_FILE,
    XGBOOST_MODEL_FILE,
    COLUMNS_LIST_FILE,
    MODEL_RESULTS_FILE,
)


def load_test_data() -> Tuple[pd.DataFrame, pd.Series]:
    X_test = pd.read_csv(X_TEST_FILE)
    y_test = pd.read_csv(Y_TEST_FILE).iloc[:, 0]
    return X_test, y_test



def load_logistic_regression():
    return joblib.load(LR_MODEL_FILE)


def load_xgboost():
    model = XGBRFClassifier()
    model.load_model(str(XGBOOST_MODEL_FILE))
    return model


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict:
    return classification_report(
        y_true, y_pred, output_dict=True
    )


def save_columns_and_results(
    X: pd.DataFrame,
    model_results: dict,
) -> None:
    COLUMNS_LIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(COLUMNS_LIST_FILE, "w", encoding="utf-8") as f:
        json.dump({"columns": list(X.columns)}, f)

    with open(MODEL_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(model_results, f, indent=2)



def wait_until_ready(model_name: str, version: int) -> None:
    client = MlflowClient()
    for _ in range(10):
        details = client.get_model_version(
            name=model_name,
            version=version,
        )
        status = ModelVersionStatus.from_string(details.status)
        print(f"Model status: {ModelVersionStatus.to_string(status)}")
        if status == ModelVersionStatus.READY:
            return
        time.sleep(1)


def register_model(
    run_id: str,
    artifact_path: str,
    model_name: str,
) -> Optional[dict]:
    client = MlflowClient()
    model_uri = mlflow.get_artifact_uri(
        artifact_path=artifact_path,
        run_id=run_id,
    )
    details = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )
    wait_until_ready(details.name, details.version)
    return dict(details)

def evaluation_pipeline() -> None:
    X_test, y_test = load_test_data()

    lr_model = load_logistic_regression()
    xgb_model = load_xgboost()

    y_pred_lr = lr_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)


    lr_report = evaluate_predictions(y_test, y_pred_lr)
    xgb_report = evaluate_predictions(y_test, y_pred_xgb)

    model_results = {
        "logistic_regression": lr_report,
        "xgboost": xgb_report,
    }

  
    save_columns_and_results(X_test, model_results)

    print("Evaluation complete.")
    print(f"- {COLUMNS_LIST_FILE}")
    print(f"- {MODEL_RESULTS_FILE}")

if __name__ == "__main__":
    evaluation_pipeline()
