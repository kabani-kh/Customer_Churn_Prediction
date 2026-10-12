
from pathlib import Path
import pandas as pd

# 1. Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_DIR / "data" / "Telco-Customer-Churn.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / "Telco-Customer-Churn-Cleaned.csv"

# 2. Read original dataset
df = pd.read_csv(INPUT_PATH)

print("Original dataset shape:")
print(df.shape)

# 3. Check duplicate customer IDs
duplicate_ids = df["customerID"].duplicated().sum()

print("\nDuplicate customer IDs:")
print(duplicate_ids)

# 4. Inspect TotalCharges data type
print("\nTotalCharges type before cleaning:")
print(df["TotalCharges"].dtype)

# 5. Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# 6. Check missing values after conversion
print("\nMissing values after conversion:")
print(df.isnull().sum())

# 7. Inspect rows with missing TotalCharges
missing_total = df["TotalCharges"].isna()

print("\nRows with missing TotalCharges:")
print(df.loc[missing_total, ["customerID", "tenure", "TotalCharges"]])

# 8. Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"]).copy()

# 9. Check remaining missing values
print("\nRemaining missing values:")
print(df.isnull().sum().sum())

# 10. Check cleaned data types
print("\nCleaned data types:")
print(df.dtypes)

# 11. Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset shape:")
print(df.shape)

print("\nCleaned dataset saved successfully!")
print(OUTPUT_PATH)
