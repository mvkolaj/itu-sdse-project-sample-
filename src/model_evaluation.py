import json
from pathlib import Path

import pandas as pd
import joblib
from sklearn.metrics import classification_report



PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = PROJECT_ROOT / "model"

X_TEST_FILE = ARTIFACTS_DIR / "X_test.csv"
Y_TEST_FILE = ARTIFACTS_DIR / "y_test.csv"
LOGISTIC_REGRESSION_MODEL_FILE = ARTIFACTS_DIR / "logistic_regression_model.pkl"
XGBOOST_MODEL_FILE = ARTIFACTS_DIR / "xgboost_model.pkl"
FEATURE_COLUMN_LIST_FILE = ARTIFACTS_DIR / "feature_column_list.json"
MODEL_RESULTS_FILE = ARTIFACTS_DIR / "model_results.json"

def load_test_data():
    X_test = pd.read_csv(X_TEST_FILE)
    y_test = pd.read_csv(Y_TEST_FILE).iloc[:, 0]
    return X_test, y_test


def load_logistic_regression():
    return joblib.load(LOGISTIC_REGRESSION_MODEL_FILE)


def load_xgboost():
    return joblib.load(XGBOOST_MODEL_FILE)


def evaluate_predictions(y_true, y_pred):
    return classification_report(y_true, y_pred, output_dict=True)


def save_columns_and_results(X, model_results):
    FEATURE_COLUMN_LIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(FEATURE_COLUMN_LIST_FILE, "w", encoding="utf-8") as f:
        json.dump({"columns": list(X.columns)}, f)

    with open(MODEL_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(model_results, f, indent=2)


def select_best_model(model_results: dict) -> str:
    scores = {
        model: results["weighted avg"]["f1-score"]
        for model, results in model_results.items()
    }

    best_model = max(scores, key=scores.get)
    print(f"Best model selected: {best_model} (F1={scores[best_model]:.4f})")
    return best_model


def deploy_model(best_model: str):
    
    MODEL_DIR.mkdir(exist_ok=True)

    for f in MODEL_DIR.iterdir():
        f.unlink()

    if best_model == "logistic_regression":
        target = MODEL_DIR / "model.pkl"
        joblib.dump(joblib.load(LOGISTIC_REGRESSION_MODEL_FILE), target)

    elif best_model == "xgboost":
        target = MODEL_DIR / "model.pkl"
        joblib.dump(joblib.load(XGBOOST_MODEL_FILE), target)

    else:
        raise ValueError(f"Unknown model type: {best_model}")

    print(f"Model deployed to: {target.resolve()}")


def evaluation_pipeline():
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

    best_model = select_best_model(model_results)
    deploy_model(best_model)

    print("Evaluation and deployment completed successfully.")


if __name__ == "__main__":
    evaluation_pipeline()