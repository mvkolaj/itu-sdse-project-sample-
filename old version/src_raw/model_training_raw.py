# %% [markdown]
# # MODEL TRAINING
# 
# Training the model uses a training dataset for training an ML algorithm. It has sample output data and the matching input data that affects the output.

# %%
import datetime

# Constants used:
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_gold_path = "./artifacts/train_data_gold.csv"
data_version = "00000"
experiment_name = current_date

# %% [markdown]
# # Create paths
# 
# Maybe the artifacts path has not been created during data cleaning

# %%
import os
import shutil

os.makedirs("artifacts", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)
os.makedirs("mlruns/.trash", exist_ok=True)

# %%
import mlflow

mlflow.set_experiment(experiment_name)

# %% [markdown]
# # Helper functions
# 
# * *create_dummies*: Create one-hot encoding columns in the data.

# %%
def create_dummy_cols(df, col):
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df

# %% [markdown]
# # Load training data
# We use the training data we cleaned earlier

# %%
data = pd.read_csv(data_gold_path)
print(f"Training data length: {len(data)}")
data.head(5)

# %% [markdown]
# # Data type split

# %%
data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)

cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
cat_vars = data[cat_cols]

other_vars = data.drop(cat_cols, axis=1)

# %% [markdown]
# # Dummy variable for categorical vars
# 
# 1. Create one-hot encoded cols for cat vars
# 2. Change to floats

# %%
import pandas as pd

for col in cat_vars:
    cat_vars[col] = cat_vars[col].astype("category")
    cat_vars = create_dummy_cols(cat_vars, col)

data = pd.concat([other_vars, cat_vars], axis=1)

for col in data:
    data[col] = data[col].astype("float64")
    print(f"Changed column {col} to float")

# %% [markdown]
# # Splitting data

# %%
y = data["lead_indicator"]
X = data.drop(["lead_indicator"], axis=1)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.15, stratify=y
)
y_train

# training_xgboost.py
import json
import joblib
import pandas as pd
from pathlib import Path
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRFClassifier


def train_xgboost(X_train, y_train):

    Path("artifacts").mkdir(exist_ok=True)

    # ---- Define model and hyperparam search ----
    model = XGBRFClassifier(random_state=42)

    params = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
        "eval_metric": ["aucpr", "error"]
    }

    grid = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        cv=10,
        n_iter=10,
        n_jobs=-1,
        verbose=3
    )

    # ---- Train ----
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # ---- Save model ----
    xgb_path = "artifacts/lead_model_xgboost.json"
    best_model.save_model(xgb_path)

    # ---- Save column names for inference ----
    with open("artifacts/columns_list.json", "w") as f:
        json.dump({"column_names": list(X_train.columns)}, f)

    print("Saved XGBoost model to:", xgb_path)
    print("Best params:", grid.best_params_)

    return best_model, grid.best_params_


# training.py
import mlflow
import mlflow.pyfunc
import joblib
import json
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

from pathlib import Path

class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]


def train(X_train, y_train, experiment_name="lr_experiment"):
    mlflow.sklearn.autolog(log_input_examples=True, log_models=False)

    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    with mlflow.start_run(experiment_id=experiment_id):

        # --- Train & Tune ---
        model = LogisticRegression()
        params = {
            'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            'penalty':  ["none", "l1", "l2", "elasticnet"],
            'C': [100, 10, 1.0, 0.1, 0.01]
        }

        grid = RandomizedSearchCV(model, params, cv=3, n_iter=10, verbose=3)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        # --- Save artifacts ---
        Path("artifacts").mkdir(exist_ok=True)

        joblib.dump(best_model, "artifacts/lead_model_lr.pkl")

        # Custom probability wrapper
        mlflow.pyfunc.log_model("prob_model", python_model=lr_wrapper(best_model))

        # Save column names for inference
        with open("artifacts/columns_list.json", "w") as f:
            json.dump({"column_names": list(X_train.columns)}, f)

        return best_model, grid.best_params_


