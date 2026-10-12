
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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
    confusion_matrix,
    classification_report
)

# -----------------------------------
# Step 1: Project paths
# -----------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "Telco-Customer-Churn-Cleaned.csv"
)

MODEL_DIR = PROJECT_DIR / "models"
FIGURE_DIR = PROJECT_DIR / "reports" / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# Step 2: Load dataset
# -----------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset shape:")
print(df.shape)

# -----------------------------------
# Step 3: Define X and y
# -----------------------------------

X = df.drop(columns=["customerID", "Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

if y.isna().any():
    raise ValueError("Unexpected Churn labels found.")

y = y.astype(int)

# -----------------------------------
# Step 4: Identify feature types
# -----------------------------------

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

# -----------------------------------
# Step 5: Train/Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# -----------------------------------
# Step 6: Preprocessing
# -----------------------------------

numerical_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -----------------------------------
# Step 7: Create model pipeline
# -----------------------------------

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

# -----------------------------------
# Step 8: Train the model
# -----------------------------------

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Model training completed!")

# -----------------------------------
# Step 9: Make predictions
# -----------------------------------

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# -----------------------------------
# Step 10: Evaluate model
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n========== MODEL EVALUATION ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# -----------------------------------
# Step 11: Confusion Matrix
# -----------------------------------

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "logistic_regression_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# -----------------------------------
# Step 12: Save trained pipeline
# -----------------------------------

MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"

joblib.dump(model, MODEL_PATH)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nLogistic Regression workflow completed!")
