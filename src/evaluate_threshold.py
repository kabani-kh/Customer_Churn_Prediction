
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ---------------------------------
# Step 1: Define paths
# ---------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "Telco-Customer-Churn-Cleaned.csv"
)

FIGURE_DIR = PROJECT_DIR / "reports" / "figures"
REPORT_DIR = PROJECT_DIR / "reports"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------
# Step 2: Load dataset
# ---------------------------------

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["customerID", "Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

if y.isna().any():
    raise ValueError("Unexpected target labels.")

y = y.astype(int)

# ---------------------------------
# Step 3: Train/Test split
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------------------------------
# Step 4: Feature types
# ---------------------------------

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

# ---------------------------------
# Step 5: Build pipeline
# ---------------------------------

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

# ---------------------------------
# Step 6: Train on training data
# ---------------------------------

model.fit(X_train, y_train)

# ---------------------------------
# Step 7: Predict probabilities
# ---------------------------------

y_prob = model.predict_proba(X_test)[:, 1]

# ---------------------------------
# Step 8: Compare thresholds
# ---------------------------------

thresholds = [0.50, 0.30]

results = []

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

results_df = pd.DataFrame(results)

print("\n========== THRESHOLD TEST COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

print("\nROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

# ---------------------------------
# Step 9: Save comparison
# ---------------------------------

results_df.to_csv(
    REPORT_DIR / "threshold_test_comparison.csv",
    index=False
)

# ---------------------------------
# Step 10: Plot confusion matrices
# ---------------------------------

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"]
    )

    plt.title(
        f"Logistic Regression - Threshold {threshold:.2f}"
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    filename = f"confusion_matrix_threshold_{threshold:.2f}.png"

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("\nThreshold evaluation completed successfully!")
