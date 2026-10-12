
from pathlib import Path
import json

import pandas as pd
import joblib

# -----------------------------------
# Step 1: Define project paths
# -----------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "final_churn_model.joblib"
)

CONFIG_PATH = (
    PROJECT_DIR
    / "models"
    / "final_model_config.json"
)

# -----------------------------------
# Step 2: Check required files
# -----------------------------------

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

if not CONFIG_PATH.is_file():
    raise FileNotFoundError(
        f"Config file not found: {CONFIG_PATH}"
    )

# -----------------------------------
# Step 3: Load trained model
# -----------------------------------

model = joblib.load(MODEL_PATH)

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = json.load(file)

threshold = float(config["threshold"])
required_features = config["features"]

print("Model loaded successfully!")
print("Classification threshold:", threshold)

# -----------------------------------
# Step 4: Create an example customer
# -----------------------------------

customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.00,
    "TotalCharges": 285.00
}

# -----------------------------------
# Step 5: Convert customer to DataFrame
# -----------------------------------

customer_df = pd.DataFrame([customer])

# Check feature names
missing_features = [
    col for col in required_features
    if col not in customer_df.columns
]

extra_features = [
    col for col in customer_df.columns
    if col not in required_features
]

if missing_features or extra_features:
    raise ValueError(
        f"Missing features: {missing_features}\n"
        f"Extra features: {extra_features}"
    )

# Match original feature order
customer_df = customer_df[required_features]

# -----------------------------------
# Input Validation
# -----------------------------------

# Check missing values
if customer_df.isnull().any().any():
    raise ValueError("Customer data contains missing values.")

# Check numerical values
numeric_columns = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

for column in numeric_columns:
    customer_df[column] = pd.to_numeric(
        customer_df[column],
        errors="raise"
    )

    if not customer_df[column].between(
        0, float("inf")
    ).all():
        raise ValueError(
            f"{column} cannot be negative."
        )

# Check SeniorCitizen
if not customer_df["SeniorCitizen"].isin([0, 1]).all():
    raise ValueError(
        "SeniorCitizen must be 0 or 1."
    )

# Check contract types
allowed_contracts = [
    "Month-to-month",
    "One year",
    "Two year"
]

if not customer_df["Contract"].isin(
    allowed_contracts
).all():
    raise ValueError(
        "Invalid contract type."
    )

print("\nInput validation passed!")


# -----------------------------------
# Step 6: Predict churn probability
# -----------------------------------

churn_probability = model.predict_proba(
    customer_df
)[0, 1]

# -----------------------------------
# Step 7: Apply selected threshold
# -----------------------------------

predicted_class = int(
    churn_probability >= threshold
)

predicted_label = (
    "Churn"
    if predicted_class == 1
    else "No Churn"
)

# -----------------------------------
# Step 8: Display results
# -----------------------------------

print("\n========== CUSTOMER INFORMATION ==========")

print(customer_df.to_string(index=False))

print("\n========== PREDICTION RESULTS ==========")

print(
    f"Churn Probability: {churn_probability:.2%}"
)

print(
    f"Classification Threshold: {threshold:.2f}"
)

print(
    f"Predicted Class: {predicted_class}"
)

print(
    f"Prediction: {predicted_label}"
)

print("\nPrediction completed successfully!")
