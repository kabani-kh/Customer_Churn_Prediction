
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# ---------------------------------
# Step 1: Project paths
# ---------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "Telco-Customer-Churn-Cleaned.csv"
)

# ---------------------------------
# Step 2: Load dataset
# ---------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset shape:")
print(df.shape)

# ---------------------------------
# Step 3: Separate features and target
# ---------------------------------

X = df.drop(columns=["customerID", "Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

if y.isna().any():
    raise ValueError("Unexpected Churn labels found.")

y = y.astype(int)

print("\nFeatures shape:")
print(X.shape)

print("\nTarget distribution:")
print(y.value_counts())

# ---------------------------------
# Step 4: Identify column types
# ---------------------------------

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)

# ---------------------------------
# Step 5: Split data
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining features shape:")
print(X_train.shape)

print("\nTesting features shape:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(4))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(4))

# ---------------------------------
# Step 6: Define preprocessing
# ---------------------------------

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
        (
            "num",
            numerical_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

print("\nPreprocessing pipeline created successfully.")

print("\nNumerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))
