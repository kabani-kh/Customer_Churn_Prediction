
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------
# Step 1: Define project paths
# ---------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "Telco-Customer-Churn-Cleaned.csv"

FIGURE_DIR = PROJECT_DIR / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------
# Step 2: Load cleaned dataset
# ---------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset shape:")
print(df.shape)

# Set a consistent chart style
sns.set_theme(style="whitegrid")

# ---------------------------------
# Step 3: Churn Distribution
# ---------------------------------

churn_counts = df["Churn"].value_counts()

print("\nChurn counts:")
print(churn_counts)

churn_percentages = df["Churn"].value_counts(normalize=True) * 100

print("\nChurn percentages:")
print(churn_percentages.round(2))

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Churn",
    order=["No", "Yes"],
    hue="Churn",
    hue_order=["No", "Yes"],
    palette={"No": "steelblue", "Yes": "tomato"},
    legend=False
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "churn_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ---------------------------------
# Step 4: Contract vs Churn
# ---------------------------------

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\nContract vs Churn (%):")
print(contract_churn.round(2))

contract_order = [
    "Month-to-month",
    "One year",
    "Two year"
]

contract_churn = contract_churn.reindex(contract_order)

ax = contract_churn.plot(
    kind="bar",
    figsize=(9, 5),
    color=["steelblue", "tomato"]
)

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Percentage of Customers (%)")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "contract_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nEDA charts saved successfully!")


# ---------------------------------
# Step 5: Tenure vs Churn
# ---------------------------------

print("\n========== TENURE ANALYSIS ==========")

print("\nAverage tenure by Churn:")
print(df.groupby("Churn")["tenure"].mean().round(2))

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    hue_order=["No", "Yes"],
    bins=24,
    stat="density",
    common_norm=False,
    element="step",
    alpha=0.35,
    palette={"No": "steelblue", "Yes": "tomato"}
)

plt.title("Customer Tenure Distribution by Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Density")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "tenure_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ---------------------------------
# Step 6: Monthly Charges vs Churn
# ---------------------------------

print("\n========== MONTHLY CHARGES ANALYSIS ==========")

print("\nMonthly charges statistics by Churn:")
print(
    df.groupby("Churn")["MonthlyCharges"]
    .agg(["mean", "median", "min", "max"])
    .round(2)
)

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges",
    order=["No", "Yes"],
    hue="Churn",
    palette={"No": "steelblue", "Yes": "tomato"},
    legend=False
)

plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn Status")
plt.ylabel("Monthly Charges ($)")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "monthly_charges_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("\nAll four EDA charts saved successfully!")
