# model_evaluation.py

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def evaluate_model(model, X_test, y_test, model_type="logreg"):

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="binary")

    report_dict = classification_report(y_test, preds, output_dict=True)
    conf = confusion_matrix(y_test, preds).tolist()

    metrics = {
        "model_type": model_type,
        "accuracy": acc,
        "f1_score": f1,
        "confusion_matrix": conf,
        "classification_report": report_dict,
    }

    output_path = ARTIFACTS_DIR / f"{model_type}_evaluation.json"
    with output_path.open("w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n=== Evaluation Results ({model_type}) ===")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Confusion Matrix:", conf)
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    return metrics
