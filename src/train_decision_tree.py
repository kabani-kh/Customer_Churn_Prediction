
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------
# Step 1: Define paths
# ---------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR / "data" /
    "Telco-Customer-Churn-Cleaned.csv"
)

FIGURE_DIR = PROJECT_DIR / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------
# Step 2: Load data
# ---------------------------------

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["customerID", "Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

if y.isna().any():
    raise ValueError("Unexpected Churn labels.")

y = y.astype(int)

# ---------------------------------
# Step 3: Feature types
# ---------------------------------

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

# ---------------------------------
# Step 4: Train/Test Split
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# ---------------------------------
# Step 5: Preprocessing
# ---------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
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

# ---------------------------------
# Step 6: Decision Tree Pipeline
# ---------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=20,
                random_state=42
            )
        )
    ]
)

# ---------------------------------
# Step 7: Cross-Validation
# ---------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    },
    n_jobs=-1
)

print("\n========== CROSS-VALIDATION ==========")

for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    scores = cv_results[f"test_{metric}"]
    print(
        f"{metric:10}: "
        f"{scores.mean():.4f} "
        f"(std: {scores.std():.4f})"
    )

# ---------------------------------
# Step 8: Train on all training data
# ---------------------------------

model.fit(X_train, y_train)

print("\nDecision Tree training completed!")

# ---------------------------------
# Step 9: Predict test data
# ---------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ---------------------------------
# Step 10: Evaluation
# ---------------------------------

print("\n========== TEST EVALUATION ==========")

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ---------------------------------
# Step 11: Confusion Matrix
# ---------------------------------

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

plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "decision_tree_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------
# Step 12: Visualize decision tree
# ---------------------------------

feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

tree_model = model.named_steps["classifier"]

plt.figure(figsize=(24, 12))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["No Churn", "Churn"],
    filled=True,
    rounded=True,
    fontsize=7,
    max_depth=3
)

plt.title("Decision Tree - First Four Levels")

plt.savefig(
    FIGURE_DIR / "decision_tree_visualization.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\nDecision Tree charts saved successfully!")
