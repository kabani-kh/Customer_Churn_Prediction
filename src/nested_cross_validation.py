
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ---------------------------------------
# Step 1: Define project paths
# ---------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "Telco-Customer-Churn-Cleaned.csv"
)

REPORT_DIR = PROJECT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# Step 2: Load and prepare dataset
# ---------------------------------------

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["customerID", "Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

if y.isna().any():
    raise ValueError("Unexpected target labels found.")

y = y.astype(int)

# Keep the same original training split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Original training shape:", X_train.shape)
print("Original test shape:", X_test.shape)

print("\nNested CV uses ONLY the original training set.")

# ---------------------------------------
# Step 3: Identify feature types
# ---------------------------------------

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

# ---------------------------------------
# Step 4: Define a model-building function
# ---------------------------------------

def build_pipeline():

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

    return pipeline

# ---------------------------------------
# Step 5: Hyperparameter search space
# ---------------------------------------

param_grid = {
    "classifier__C": [
        0.01, 0.1, 1.0, 10.0, 100.0
    ],
    "classifier__class_weight": [
        None, "balanced"
    ]
}

# ---------------------------------------
# Step 6: Configure nested validation
# ---------------------------------------

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

inner_cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

threshold_values = np.arange(0.20, 0.71, 0.05)

results = []

# ---------------------------------------
# Step 7: Outer CV loop
# ---------------------------------------

for fold_number, (train_idx, valid_idx) in enumerate(
    outer_cv.split(X_train, y_train),
    start=1
):

    print(f"\n========== OUTER FOLD {fold_number} ==========")

    X_inner_train = X_train.iloc[train_idx]
    X_outer_valid = X_train.iloc[valid_idx]

    y_inner_train = y_train.iloc[train_idx]
    y_outer_valid = y_train.iloc[valid_idx]

    # -----------------------------------
    # Step 8: Tune hyperparameters
    # -----------------------------------

    grid_search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=param_grid,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=1,
        refit=True
    )

    grid_search.fit(X_inner_train, y_inner_train)

    best_params = grid_search.best_params_

    print("Best parameters:", best_params)
    print("Inner CV ROC-AUC:", round(grid_search.best_score_, 4))

    # -----------------------------------
    # Step 9: Select threshold
    # -----------------------------------

    tuned_model = build_pipeline()

    tuned_model.set_params(**best_params)

    oof_prob = cross_val_predict(
        tuned_model,
        X_inner_train,
        y_inner_train,
        cv=inner_cv,
        method="predict_proba",
        n_jobs=1
    )[:, 1]

    best_threshold = None
    best_f1 = -1.0

    for threshold in threshold_values:

        predictions = (
            oof_prob >= threshold
        ).astype(int)

        current_f1 = f1_score(
            y_inner_train,
            predictions,
            zero_division=0
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)

    print("Selected threshold:", round(best_threshold, 2))

    # -----------------------------------
    # Step 10: Train and evaluate
    # -----------------------------------

    tuned_model.fit(X_inner_train, y_inner_train)

    valid_prob = tuned_model.predict_proba(
        X_outer_valid
    )[:, 1]

    valid_pred = (
        valid_prob >= best_threshold
    ).astype(int)

    fold_result = {
        "Fold": fold_number,
        "Best_C": best_params["classifier__C"],
        "Class_Weight": str(
            best_params["classifier__class_weight"]
        ),
        "Threshold": round(best_threshold, 2),
        "Accuracy": accuracy_score(
            y_outer_valid, valid_pred
        ),
        "Precision": precision_score(
            y_outer_valid, valid_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_outer_valid, valid_pred,
            zero_division=0
        ),
        "F1": f1_score(
            y_outer_valid, valid_pred,
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_outer_valid, valid_prob
        )
    }

    results.append(fold_result)

    print(
        "Outer F1:",
        round(fold_result["F1"], 4)
    )
    print(
        "Outer ROC-AUC:",
        round(fold_result["ROC_AUC"], 4)
    )

# ---------------------------------------
# Step 11: Results table
# ---------------------------------------

results_df = pd.DataFrame(results)

print("\n========== NESTED CV RESULTS ==========")

print(results_df.round(4).to_string(index=False))

# ---------------------------------------
# Step 12: Summary statistics
# ---------------------------------------

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC"
]

summary_df = results_df[metric_columns].agg(
    ["mean", "std"]
)

print("\n========== NESTED CV SUMMARY ==========")

print(summary_df.round(4).to_string())

# ---------------------------------------
# Step 13: Save reports
# ---------------------------------------

results_df.to_csv(
    REPORT_DIR / "nested_cv_fold_results.csv",
    index=False
)

summary_df.to_csv(
    REPORT_DIR / "nested_cv_summary.csv"
)

print("\nNested CV completed successfully!")
print("Reports saved to:", REPORT_DIR)
