# Exploratory Data Analysis — Customer Churn Prediction

## Overview

This analysis examines customer churn patterns in the IBM Telco Customer Churn dataset, focusing on contract type, customer tenure, and monthly charges.

The objective is to understand how these characteristics differ between customers who stayed and those who left, before developing a churn prediction model.

## Dataset and Cleaning

The original dataset contains 7,043 customer records and 21 columns.

During data cleaning:
- No duplicate customer IDs were found.
- The `TotalCharges` column was converted from text to a numeric format.
- Eleven records contained missing `TotalCharges` values. All eleven had a tenure of zero months.
- These records were excluded from the analysis.

The cleaned dataset contains **7,032 customers**, with no remaining missing values.

## Exploratory Findings

### 1. Customer Churn Distribution

Of the 7,032 customers:

- 5,163 (73.42%) did not churn.
- 1,869 (26.58%) churned.

The target classes are imbalanced. This will be taken into account when evaluating the classification models, particularly when interpreting precision, recall, and F1-score.

### 2. Contract Type

Churn rates differ considerably across contract types.

| Contract Type | Churn Rate |
|---|---:|
| Month-to-month | 42.71% |
| One year | 11.28% |
| Two year | 2.85% |

Customers on month-to-month contracts have the highest observed churn rate.

Contract type appears to be a useful variable to investigate during predictive modeling.

### 3. Customer Tenure

The tenure distributions show a clear difference between customers who churned and those who stayed.

Customers who left are more concentrated in the earlier months of their subscriptions, while longer-tenure customers are more common among those who remained.

This suggests that customer tenure may help distinguish between the two groups.

### 4. Monthly Charges

Monthly charges also differ between the two groups.

| Metric | Stayed | Churned |
|---|---:|---:|
| Average monthly charge | $61.31 | $74.44 |
| Median monthly charge | $64.45 | $79.65 |

Customers who churned paid approximately $13.13 more per month on average.

The boxplots show substantial overlap between the groups, however, so monthly charges alone are unlikely to provide a reliable basis for classification.

## Key Takeaways

The exploratory analysis identified three characteristics worth examining further:

- **Contract type:** Month-to-month customers have substantially higher observed churn rates.
- **Tenure:** Churned customers tend to have shorter subscription histories.
- **Monthly charges:** Higher charges are associated with churn, although the distributions overlap.

These relationships are observational and should not be interpreted as evidence of causation.

The next stage of the project will involve encoding categorical features, scaling numerical variables where appropriate, and training classification models using separate training and test datasets.

## Dataset Source

IBM Telco Customer Churn dataset.

Source: https://github.com/IBM/telco-customer-churn-on-icp4d

This project uses a publicly available sample dataset for educational and portfolio purposes.