
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# Step 1: Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR / "data" /
    "Telco-Customer-Churn-Cleaned.csv"
)

FIGURE_DIR = PROJECT_DIR / "reports" / "figures"
REPORT_DIR = PROJECT_DIR / "reports"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Step 2: Load dataset
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected target labels.")

y = y.astype(int)

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 4: Identify feature types
categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

# Step 5: Define preprocessing
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

# Step 6: Define model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

# Step 7: Cross-validation configuration
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Step 8: Get out-of-fold probabilities
print("Generating cross-validated probabilities...")

oof_prob = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

# Step 9: Evaluate different thresholds
thresholds = np.arange(0.20, 0.71, 0.05)

results = []

for threshold in thresholds:
    predictions = (oof_prob >= threshold).astype(int)

    precision = precision_score(
        y_train, predictions, zero_division=0
    )

    recall = recall_score(
        y_train, predictions, zero_division=0
    )

    f1 = f1_score(
        y_train, predictions, zero_division=0
    )

    results.append({
        "Threshold": round(float(threshold), 2),
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

# Step 10: Create results table
results_df = pd.DataFrame(results)

print("\n========== THRESHOLD COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

# Step 11: Select threshold using F1
best_row = results_df.loc[results_df["F1"].idxmax()]
best_threshold = float(best_row["Threshold"])

print("\n========== BEST CV THRESHOLD ==========")
print(f"Threshold : {best_threshold:.2f}")
print(f"Precision : {best_row['Precision']:.4f}")
print(f"Recall    : {best_row['Recall']:.4f}")
print(f"F1-score  : {best_row['F1']:.4f}")

# Step 12: Save results
results_df.to_csv(
    REPORT_DIR / "threshold_comparison.csv",
    index=False
)

# Step 13: Plot metrics
plt.figure(figsize=(10, 6))

plt.plot(
    results_df["Threshold"],
    results_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    results_df["Threshold"],
    results_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    results_df["Threshold"],
    results_df["F1"],
    marker="o",
    label="F1-score"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    color="gray",
    label=f"Selected threshold: {best_threshold:.2f}"
)

plt.title("Threshold Optimization - Logistic Regression")
plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "threshold_optimization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nThreshold comparison saved successfully!")
