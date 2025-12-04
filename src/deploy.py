# deploy.py

import json
import shutil
from pathlib import Path
import os
from datetime import datetime


ARTIFACTS_DIR = Path("artifacts")
DEPLOY_DIR = Path("deployment")

# Model artifact paths
LR_MODEL = ARTIFACTS_DIR / "lead_model_lr.pkl"
XGB_MODEL = ARTIFACTS_DIR / "lead_model_xgboost.json"
SELECTED_MODEL_FILE = ARTIFACTS_DIR / "selected_model.json"


def load_selected_model_type() -> str:
    """
    Reads artifacts/selected_model.json and retrieves the model type.
    """
    if not SELECTED_MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model selection file not found. Run model_selection.select_best_model() first."
        )

    with SELECTED_MODEL_FILE.open("r") as f:
        data = json.load(f)

    return data["selected_model"]


def deploy_model(best_model_type):
    """
    Copies the selected model to a deployment folder and saves metadata.
    """
    os.makedirs("deployment", exist_ok=True)

    # Determine which model artifact to deploy
    if best_model_type.lower() == "logreg":
        source_path = "artifacts/lead_model_lr.pkl"
        deployed_path = "deployment/production_model.pkl"
    else:
        source_path = "artifacts/lead_model_xgboost.json"
        deployed_path = "deployment/production_model.json"

    # Copy model file
    shutil.copy(source_path, deployed_path)

    # Save metadata
    metadata = {
        "model_type": best_model_type,
        "source_path": source_path,
        "deployed_path": deployed_path,
        "timestamp": str(datetime.now())
    }

    with open("deployment/deployment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✔ Deployed {best_model_type} model to: {deployed_path}")
    return metadata


if __name__ == "__main__":
    deploy_model()
