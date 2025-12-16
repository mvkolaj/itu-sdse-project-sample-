# ITU BDS MLOPS'25 - Project
This repository contains a project completed for the Data Science in Production: MLOps and Software Engineering course (Autumn 2025) at the IT University of Copenhagen.

It implements a full MLOps pipeline for training, evaluating, and deploying a machine learning pipeline using Python, Go (Dagger), GitHub Actions, and DVC. The pipeline is used to identify **potential new customers** on a website, using **user behavior data**.


## Project Overview
This main goal of this project was to refactor a Python notebook into a **machine learning pipeline**.

The pipeline handles **data preprocessing, feature engineering, model training and evaluation, model selection and deployment**.  
The original notebook has been refactored into **Python scripts**, and the pipeline is containerized with **Dagger** and automated using **GitHub Actions**. 

---

## Repository Structure
The repository is organized as follows:
```text
.
├── .dvc/                                       <- DVC internal files
├── .github/workflows/
│   └── test_action.yml                         <- GitHub Actions CI workflow
│
├── data/
│   └── raw_data.csv.dvc                        <- DVC pointer to raw dataset
│
├── docs/                                       <- Architecture diagrams and documentation assets
│
├── go/
│   ├── pipeline.go                             <- Dagger-based pipeline 
│   └── go.mod                                  <- Go module definition
│
├── notebooks/                                  <- Exploratory notebooks and experiments
│
├── src/                                        <- Core Python source code
│   ├── __init__.py
│   ├── data_preprocessing.py                   <- Data loading and preprocessing
│   ├── data_features.py                        <- Feature engineering
│   ├── model_training.py                       <- Model training
│   └── model_evaluation_and_deployment.py      <- Evaluation, model selection and deployment
│
├── README.md                                   <- Project documentation
├── requirements.txt                            <- Python dependencies




---
## How to Run Locally 

Prerequisites

Make sure the following tools are installed on your machine:

**Docker Desktop:** Must be installed and running (required by Dagger).
**Go (≥ 1.21):** Used to run the pipeline orchestrator (go/pipeline.go).
**DVC:** Used to pull the dataset from a remote source.

**1. Clone the Repository**
```bash
git clone <repository-url>
cd <repository-root>
```
**2. Run the MLOps Pipeline**
From the repository root, run:
```bash
go run go/pipeline.go
```

This command will:
1. Start the Dagger engine
2. Pull the dataset using DVC
3. Run the following pipeline stages:
-Data preprocessing
-Feature engineering
-Model trainin
-Model evaluation and deployment
4. Save all generated outputs to the local filesystem

**4. Verify the Output** 
After a successful run, the following directories will be created:
* **`mlruns/`:** Contains the local MLflow Tracking data and the Model Registry.
* **`artifacts/`:** Contains the final generated files, including `train_data_gold.csv`, and `model_results.json` etc.
* **`model/`:** Contains the final deployed model artifact, named `model.json`
---
## Using GitHub Actions Workflow


---
## Notes 


---

## Authors
Zofia Brodewicz (zobr@itu.dk) and Mikolaj Andrzejewski (mikoa@itu.dk)
