# model_selection.py

import json
from pathlib import Path


ARTIFACTS_DIR = Path("artifacts")
LOGREG_EVAL = ARTIFACTS_DIR / "logreg_evaluation.json"
XGB_EVAL = ARTIFACTS_DIR / "xgboost_evaluation.json"
SELECTED_MODEL_PATH = ARTIFACTS_DIR / "selected_model.json"


def load_metrics(file_path: Path) -> dict:
    if not file_path.exists():
        return None

    with file_path.open("r") as f:
        return json.load(f)


def select_best_model() -> str:

    logreg_metrics = load_metrics(LOGREG_EVAL)
    xgb_metrics = load_metrics(XGB_EVAL)

    if logreg_metrics is None and xgb_metrics is None:
        raise RuntimeError("No evaluation files found for model selection.")

    def get_scores(metrics):
        if metrics is None:
            return (0, 0)
        return (
            metrics.get("f1_score", 0),
            metrics.get("accuracy", 0),
        )

    logreg_f1, logreg_acc = get_scores(logreg_metrics)
    xgb_f1, xgb_acc = get_scores(xgb_metrics)

    print("\n=== MODEL SELECTION ===")
    print(f"LogReg → F1: {logreg_f1:.4f}, Acc: {logreg_acc:.4f}")
    print(f"XGBoost → F1: {xgb_f1:.4f}, Acc: {xgb_acc:.4f}")

    if xgb_f1 > logreg_f1:
        best = "xgboost"
    elif logreg_f1 > xgb_f1:
        best = "logreg"
    else:
        best = "xgboost" if xgb_acc >= logreg_acc else "logreg"

    with SELECTED_MODEL_PATH.open("w") as f:
        json.dump({"selected_model": best}, f, indent=4)

    print(f"Selected model: {best.upper()} (saved to selected_model.json)")
    return best
