
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

# -----------------------------------
# Step 1: Define paths
# -----------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR / "data" /
    "Telco-Customer-Churn-Cleaned.csv"
)

MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# Step 2: Load data
# -----------------------------------

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["customerID", "Churn"])

y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected target labels.")

y = y.astype(int)

# -----------------------------------
# Step 3: Preserve original test split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Reserved test records:", len(X_test))

# -----------------------------------
# Step 4: Identify column types
# -----------------------------------

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

# -----------------------------------
# Step 5: Build preprocessing pipeline
# -----------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]
)

# -----------------------------------
# Step 6: Configure GridSearchCV
# -----------------------------------

param_grid = {
    "classifier__C": [
        0.01, 0.1, 1.0, 10.0, 100.0
    ],
    "classifier__class_weight": [
        None, "balanced"
    ]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    refit=True,
    n_jobs=-1
)

print("\nTuning final model...")

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

print("\n========== SELECTED PARAMETERS ==========")
print(best_params)

print("\nBest CV ROC-AUC:")
print(round(grid_search.best_score_, 4))

# -----------------------------------
# Step 7: Select classification threshold
# -----------------------------------

selected_pipeline = grid_search.best_estimator_

print("\nGenerating training OOF probabilities...")

oof_prob = cross_val_predict(
    selected_pipeline,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

thresholds = np.round(
    np.arange(0.20, 0.701, 0.05),
    2
)

threshold_results = []

for threshold in thresholds:

    predictions = (oof_prob >= threshold).astype(int)

    threshold_results.append({
        "Threshold": float(threshold),
        "Precision": precision_score(
            y_train, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_train, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_train, predictions, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

best_row = threshold_df.loc[
    threshold_df["F1"].idxmax()
]

best_threshold = float(best_row["Threshold"])

print("\n========== SELECTED THRESHOLD ==========")
print("Threshold:", best_threshold)
print("Training OOF F1:", round(float(best_row["F1"]), 4))
print("Training OOF Recall:", round(float(best_row["Recall"]), 4))

# -----------------------------------
# Step 8: Fit final training pipeline
# -----------------------------------

final_model = grid_search.best_estimator_

# GridSearchCV with refit=True has already fitted
# this pipeline on all training records.

# -----------------------------------
# Step 9: Save model and configuration
# -----------------------------------

MODEL_PATH = MODEL_DIR / "final_churn_model.joblib"

CONFIG_PATH = MODEL_DIR / "final_model_config.json"

joblib.dump(final_model, MODEL_PATH)

config = {
    "model": "LogisticRegression",
    "threshold": best_threshold,
    "best_params": {
        "C": float(best_params["classifier__C"]),
        "class_weight": best_params["classifier__class_weight"]
    },
    "cv_roc_auc": float(grid_search.best_score_),
    "target_positive_class": 1,
    "features": X_train.columns.tolist(),
    "training_rows": len(X_train),
    "random_state": 42
}

with open(CONFIG_PATH, "w", encoding="utf-8") as file:
    json.dump(config, file, indent=4)

threshold_df.to_csv(
    REPORT_DIR / "final_threshold_selection.csv",
    index=False
)

print("\n========== SAVED FILES ==========")
print(MODEL_PATH)
print(CONFIG_PATH)

print("\nFinal model training completed successfully!")
