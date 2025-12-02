# evaluation.py
import joblib
import json
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)


# evaluation_xgboost.py
import json
import pandas as pd
from xgboost import XGBRFClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def evaluate_xgboost():

    # ---- Load model ----
    model = XGBRFClassifier()
    model.load_model("artifacts/lead_model_xgboost.json")

    # ---- Load test data ----
    X_test = pd.read_csv("app/artifacts/X_test.csv")
    y_test = pd.read_csv("app/artifacts/y_test.csv")

    # ---- Predictions ----
    preds = model.predict(X_test)

    # ---- Metrics ----
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    conf = confusion_matrix(y_test, preds)

    print("Accuracy:", acc)
    print("Confusion matrix:\n", conf)
    print("Classification report:\n", classification_report(y_test, preds))

    # ---- Save results ----
    with open("artifacts/xgboost_results.json", "w") as f:
        json.dump(report, f)

    return report



def evaluate():
    model = joblib.load("artifacts/lead_model_lr.pkl")

    X_test = pd.read_csv("app/artifacts/X_test.csv")
    y_test = pd.read_csv("app/artifacts/y_test.csv")

    preds = model.predict(X_test)

    # Metrics
    report = classification_report(y_test, preds, output_dict=True)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    conf = confusion_matrix(y_test, preds)

    print("Accuracy:", acc)
    print("F1:", f1)
    print("Confusion matrix:\n", conf)

    # Save results for documentation
    with open("artifacts/model_results.json", "w") as f:
        json.dump(report, f)

    return report




