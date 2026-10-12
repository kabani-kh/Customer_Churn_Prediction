
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------
# Step 1: Project paths
# ---------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "Telco-Customer-Churn-Cleaned.csv"
)

FIGURE_DIR = PROJECT_DIR / "reports" / "figures"
MODEL_DIR = PROJECT_DIR / "models"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# Step 2: Load cleaned dataset
# ---------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# ---------------------------------------
# Step 3: Define X and y
# ---------------------------------------

X = df.drop(columns=["customerID", "Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

if y.isna().any():
    raise ValueError("Unexpected Churn labels.")

y = y.astype(int)

# ---------------------------------------
# Step 4: Identify feature types
# ---------------------------------------

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

# ---------------------------------------
# Step 5: Train/Test Split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# ---------------------------------------
# Step 6: Preprocessing
# ---------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# ---------------------------------------
# Step 7: Random Forest Pipeline
# ---------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=1
            )
        )
    ]
)

# ---------------------------------------
# Step 8: Cross-Validation
# ---------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nRunning 5-fold Cross-Validation...")

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

for metric in [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]:
    scores = cv_results[f"test_{metric}"]

    print(
        f"{metric:10}: {scores.mean():.4f} "
        f"(std: {scores.std():.4f})"
    )

# ---------------------------------------
# Step 9: Train model
# ---------------------------------------

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Random Forest training completed!")

# ---------------------------------------
# Step 10: Predictions
# ---------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ---------------------------------------
# Step 11: Model evaluation
# ---------------------------------------

print("\n========== TEST EVALUATION ==========")

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ---------------------------------------
# Step 12: Confusion Matrix
# ---------------------------------------

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

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "random_forest_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------
# Step 13: Feature Importance
# ---------------------------------------

feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = (
    model.named_steps["classifier"]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== TOP 10 FEATURES ==========")
print(importance_df.head(10).to_string(index=False))

top_features = importance_df.head(15)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature",
    color="steelblue"
)

plt.title("Random Forest - Top 15 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "random_forest_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------
# Step 14: Save model
# ---------------------------------------

MODEL_PATH = MODEL_DIR / "random_forest.joblib"

joblib.dump(model, MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nRandom Forest workflow completed!")
