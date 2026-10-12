
from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# -----------------------------------
# Step 1: Project paths
# -----------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "Telco-Customer-Churn-Cleaned.csv"
)

REPORT_DIR = PROJECT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# Step 2: Load data
# -----------------------------------

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["customerID", "Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

if y.isna().any():
    raise ValueError("Unexpected target labels.")

y = y.astype(int)

# -----------------------------------
# Step 3: Train/Test split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)

# The test set is not used for tuning.

# -----------------------------------
# Step 4: Identify feature types
# -----------------------------------

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

# -----------------------------------
# Step 5: Preprocessing
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
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

# -----------------------------------
# Step 6: Create pipeline
# -----------------------------------

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
# Step 7: Hyperparameter grid
# -----------------------------------

param_grid = {
    "classifier__C": [
        0.01,
        0.1,
        1.0,
        10.0,
        100.0
    ],
    "classifier__class_weight": [
        None,
        "balanced"
    ]
}

# -----------------------------------
# Step 8: Cross-validation
# -----------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# -----------------------------------
# Step 9: Grid Search
# -----------------------------------

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

print("\nRunning GridSearchCV...")

grid_search.fit(X_train, y_train)

# -----------------------------------
# Step 10: Best parameters
# -----------------------------------

print("\n========== BEST PARAMETERS ==========")
print(grid_search.best_params_)

print("\n========== BEST CV ROC-AUC ==========")
print(round(grid_search.best_score_, 4))

# -----------------------------------
# Step 11: All results
# -----------------------------------

results = pd.DataFrame(grid_search.cv_results_)

columns_to_show = [
    "param_classifier__C",
    "param_classifier__class_weight",
    "mean_train_score",
    "mean_test_score",
    "std_test_score",
    "rank_test_score"
]

results = results[columns_to_show]

results = results.sort_values(
    by="rank_test_score"
)

print("\n========== GRID SEARCH RESULTS ==========")
print(results.round(4).to_string(index=False))

# -----------------------------------
# Step 12: Save results
# -----------------------------------

results.to_csv(
    REPORT_DIR / "hyperparameter_tuning_results.csv",
    index=False
)

print("\nHyperparameter tuning completed!")
print("Results saved successfully.")
