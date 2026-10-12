
from pathlib import Path
import pandas as pd

# Step 1: Define the main project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Step 2: Define the dataset path
DATA_PATH = PROJECT_DIR / "data" / "Telco-Customer-Churn.csv"

# Step 3: Check whether the dataset exists
if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# Step 4: Load the dataset
df = pd.read_csv(DATA_PATH)

# Step 5: Display dataset shape
print("\n========== DATASET SHAPE ==========")
print(df.shape)

# Step 6: Display first 5 rows
print("\n========== FIRST 5 ROWS ==========")
print(df.head().to_string())

# Step 7: Display column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# Step 8: Display data types and non-null counts
print("\n========== DATASET INFO ==========")
df.info()

# Step 9: Display target distribution
print("\n========== CHURN DISTRIBUTION ==========")
print(df["Churn"].value_counts())

# Step 10: Display missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\nDataset inspection completed successfully!")
