# main.py
"""
Main pipeline entrypoint for the SDSE MLOps project.

This script performs the FULL ML pipeline:
1. Data Processing
2. Model Training
3. Model Evaluation
4. Model Selection
5. Deployment

All modular logic lives inside /src/, and this file orchestrates the full workflow.
"""
import os
os.environ["NO_AT_BRIDGE"] = "1"
os.environ["PYTHONMALLOC"] = "malloc"
os.environ["FORCE_IDLE_SHUTDOWN"] = "1"

# Disable parallelism globally
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

from src.data_processing import run_data_processing
from src.model_training import run_model_training
from src.model_evaluation import evaluate_model
from src.model_selection import select_best_model
from src.deploy import deploy_model




def main():
    print("\n====================================")
    print("STEP 1 — DATA PROCESSING")
    print("====================================")

    df = run_data_processing()
    print(f"✔ Data processing complete. Shape: {df.shape}")


    print("\n====================================")
    print("STEP 2 — MODEL TRAINING")
    print("====================================")

    print("→ Training Logistic Regression...")
    lr_model, X_test_lr, y_test_lr = run_model_training(model_type="logreg")

    print("\n→ Training XGBoost...")
    xgb_model, X_test_xgb, y_test_xgb = run_model_training(model_type="xgboost")


    print("\n====================================")
    print("STEP 3 — MODEL EVALUATION")
    print("====================================")

    print("\n→ Evaluating Logistic Regression...")
    lr_metrics = evaluate_model(
        model=lr_model,
        X_test=X_test_lr,
        y_test=y_test_lr,
        model_type="logreg"
    )

    print("\n→ Evaluating XGBoost...")
    xgb_metrics = evaluate_model(
        model=xgb_model,
        X_test=X_test_xgb,
        y_test=y_test_xgb,
        model_type="xgboost"
    )


    print("\n====================================")
    print("STEP 4 — MODEL SELECTION")
    print("====================================")

    best_model_type = select_best_model()
    print(f"✔ Best model selected: {best_model_type.upper()}")


    print("\n====================================")
    print("STEP 5 — DEPLOYMENT")
    print("====================================")

    metadata = deploy_model(best_model_type=best_model_type)
    print("✔ Deployment complete.")
    print("Deployment metadata:", metadata)


    print("\n====================================")
    print("PIPELINE FINISHED SUCCESSFULLY ")
    print("====================================\n")


if __name__ == "__main__":
    main()
