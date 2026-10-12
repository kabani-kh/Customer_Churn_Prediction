
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

# -----------------------------------
# Step 1: Define project paths
# -----------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "Telco-Customer-Churn-Cleaned.csv"
)

FIGURE_DIR = PROJECT_DIR / "reports" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# Step 2: Load cleaned dataset
# -----------------------------------

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["customerID", "Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

if y.isna().any():
    raise ValueError("Unexpected Churn labels.")

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
# Step 5: Define model pipelines
# -----------------------------------

logistic_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

tree_preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", logistic_preprocessor),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Decision Tree": Pipeline([
        ("preprocessor", tree_preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", tree_preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ))
    ])
}

# -----------------------------------
# Step 6: Plot ROC curves
# -----------------------------------

prediction_scores = {}
plt.figure(figsize=(9, 7))

for model_name, model in models.items():

    print(f"Training {model_name}...")

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    prediction_scores[model_name] = y_prob

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_prob
    )

    auc_score = roc_auc_score(
        y_test,
        y_prob
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} (AUC = {auc_score:.4f})"
    )

    print(f"{model_name} ROC-AUC: {auc_score:.4f}")

# Random classifier reference line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random Classifier (AUC = 0.50)"
)

plt.title("ROC Curve Comparison - Customer Churn Prediction")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.tight_layout()

# -----------------------------------
# Step 7: Save figure
# -----------------------------------

OUTPUT_PATH = FIGURE_DIR / "roc_curve_comparison.png"

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nROC comparison chart saved successfully!")
print(OUTPUT_PATH)


# -----------------------------------
# Step 8: Precision-Recall Comparison
# -----------------------------------

plt.figure(figsize=(9, 7))

for model_name, y_prob in prediction_scores.items():

    precision, recall, thresholds = precision_recall_curve(
        y_test,
        y_prob
    )

    ap_score = average_precision_score(
        y_test,
        y_prob
    )

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"{model_name} (AP = {ap_score:.4f})"
    )

    print(f"{model_name} Average Precision: {ap_score:.4f}")

# Positive class prevalence
baseline_precision = y_test.mean()

plt.axhline(
    y=baseline_precision,
    linestyle="--",
    color="gray",
    label=f"Class Prevalence = {baseline_precision:.4f}"
)

plt.title("Precision-Recall Curve Comparison")
plt.xlabel("Recall")
plt.ylabel("Precision")

plt.xlim(0, 1)
plt.ylim(0, 1.05)

plt.legend(loc="lower left")
plt.grid(alpha=0.3)

plt.tight_layout()

# Save figure
PR_OUTPUT_PATH = FIGURE_DIR / "precision_recall_comparison.png"

plt.savefig(
    PR_OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nPrecision-Recall comparison saved successfully!")
print(PR_OUTPUT_PATH)