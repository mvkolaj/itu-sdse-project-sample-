# deploy.py

import json
import shutil
from pathlib import Path
import os
from datetime import datetime

ARTIFACTS_DIR = Path("artifacts")
DEPLOY_DIR = Path("deployment")

LR_MODEL = ARTIFACTS_DIR / "lead_model_lr.pkl"
XGB_MODEL = ARTIFACTS_DIR / "lead_model_xgboost.json"
SELECTED_MODEL_FILE = ARTIFACTS_DIR / "selected_model.json"


def load_selected_model_type() -> str:
    if not SELECTED_MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model selection file not found. Run model_selection.select_best_model() first."
        )
    with SELECTED_MODEL_FILE.open("r") as f:
        data = json.load(f)
    return data["selected_model"]


def deploy_model(best_model_type: str):
    os.makedirs(DEPLOY_DIR, exist_ok=True)

    # Determine which model artifact to deploy
    if best_model_type.lower() == "logreg":
        source_path = LR_MODEL
        deployed_path = DEPLOY_DIR / "production_model.pkl"
    else:
        source_path = XGB_MODEL
        deployed_path = DEPLOY_DIR / "production_model.json"

    # Copy model file into deployment folder
    shutil.copy(source_path, deployed_path)

    # Save metadata
    metadata = {
        "model_type": best_model_type,
        "source_path": str(source_path),
        "deployed_path": str(deployed_path),
        "timestamp": str(datetime.now()),
    }
    (DEPLOY_DIR / "deployment_metadata.json").write_text(
        json.dumps(metadata, indent=4)
    )

    print(f"✔ Deployed {best_model_type} model to: {deployed_path}")


    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)

    # Decide canonical filename inside model/
    if deployed_path.suffix == ".pkl":
        canonical_name = "model.pkl"
    else:
        canonical_name = "model.json"

    model_target = model_dir / canonical_name

    # Remove old canonical file if present
    if model_target.exists():
        model_target.unlink()

    shutil.copy2(deployed_path, model_target)
    print(f"✔ Canonical model artifact saved as: {model_target.resolve()}")
    return metadata
