# Customer Churn Prediction — Model Evaluation Report

## 1. Project Overview

This project examines customer churn using the IBM Telco Customer Churn dataset. The objective is to explore patterns associated with customer attrition and develop classification models that can identify customers with a higher estimated risk of leaving.

The workflow includes data cleaning, exploratory data analysis, feature preprocessing, baseline evaluation, model comparison, hyperparameter tuning, threshold selection, and nested cross-validation.

Three classification algorithms were investigated:

- Logistic Regression
- Decision Tree
- Random Forest

A majority-class DummyClassifier was used as a baseline.

The analysis focuses on model performance as well as the trade-offs involved in identifying customers who may churn.

## 2. Dataset Description

The original dataset contains 7,043 customer records and 21 columns, including the target variable, `Churn`.

The available information covers customer demographics, subscription history, contracted services, payment methods, and billing amounts.

### Data Cleaning

The following checks and transformations were performed:

- Checked customer IDs for duplicates; none were found.
- Converted `TotalCharges` from a text field to a numeric field.
- Identified 11 records with missing `TotalCharges` values.
- Confirmed that all 11 records had a tenure of zero months.
- Excluded these 11 records from the modeling dataset.
- Verified that no missing values remained after cleaning.

The resulting dataset contains **7,032 records and 21 columns**.

### Target Variable

The target variable was encoded as:

| Original Value | Encoded Value |
|---|---:|
| No | 0 |
| Yes | 1 |

The `customerID` column was excluded from the predictive features because it serves as an identifier rather than a meaningful customer characteristic.

The remaining dataset contains 19 input features.

## 3. Exploratory Data Analysis

Exploratory analysis was conducted to examine customer characteristics associated with churn before model training.

### 3.1 Churn Distribution

The cleaned dataset contains:

| Churn Status | Customers | Percentage |
|---|---:|---:|
| No | 5,163 | 73.42% |
| Yes | 1,869 | 26.58% |
| Total | 7,032 | 100% |

The target distribution is imbalanced, with considerably fewer churned customers than retained customers.

For this reason, model evaluation considers precision, recall, F1-score, ROC-AUC, and Average Precision in addition to accuracy.

### 3.2 Contract Type

Churn rates vary substantially across contract categories.

| Contract Type | Churn Rate |
|---|---:|
| Month-to-month | 42.71% |
| One year | 11.28% |
| Two year | 2.85% |

Month-to-month customers had the highest observed churn rate.

This suggests that contract type may be useful for distinguishing customer groups with different churn patterns.

### 3.3 Customer Tenure

The tenure distributions differed between customers who stayed and those who left.

Churned customers were more concentrated among shorter-tenure subscribers, while customers who remained were more frequently represented at longer subscription durations.

Tenure was therefore retained as a potential predictive feature.

### 3.4 Monthly Charges

Customers who churned had higher monthly charges on average.

| Metric | Stayed | Churned |
|---|---:|---:|
| Mean monthly charges | $61.31 | $74.44 |
| Median monthly charges | $64.45 | $79.65 |

The mean difference was approximately $13.13 per month.

However, the distributions overlap, indicating that monthly charges alone are unlikely to distinguish the two customer groups reliably.

These exploratory findings describe associations within the dataset and do not establish causal relationships.

## 4. Data Preprocessing

The cleaned dataset was divided into training and testing subsets using an 80/20 stratified split.

| Subset | Records |
|---|---:|
| Training | 5,625 |
| Testing | 1,407 |

The split used `random_state=42` to support reproducibility.

Stratification preserved approximately the same churn proportion in both subsets.

### Feature Transformation

Categorical features were transformed using One-Hot Encoding.

Numerical features were standardized for Logistic Regression using StandardScaler. The tree-based models used their numerical features without scaling.

Preprocessing was included within scikit-learn Pipelines so that transformations were fitted on the appropriate training partitions.

This approach helps prevent information leakage from validation and test data during model training.

## 5. Baseline Model

A DummyClassifier using the most frequent class was evaluated as a baseline.

Because most customers did not churn, the baseline predicted `Churn = 0` for every record.

Its test-set results were:

| Metric | Result |
|---|---:|
| Accuracy | 0.7342 |
| Precision (Churn) | 0.0000 |
| Recall (Churn) | 0.0000 |
| F1-score (Churn) | 0.0000 |

The model correctly classified the 1,033 non-churned customers but failed to identify all 374 churned customers in the test set.

This provides a useful reference: a model can achieve relatively high accuracy while failing completely at the main task of identifying churn.

## 6. Initial Classification Model Comparison

Logistic Regression, Decision Tree, and Random Forest were trained using the same original training and test split.

The following table summarizes their initial test-set performance at a classification threshold of 0.50.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8038 | 0.6485 | 0.5722 | 0.6080 | 0.8359 |
| Decision Tree | 0.7903 | 0.6037 | 0.6150 | 0.6093 | 0.8264 |
| Random Forest | 0.7939 | 0.6409 | 0.5107 | 0.5685 | 0.8338 |

Precision, recall, and F1-score in this table refer to the positive class (`Churn = 1`).

### Observations

Logistic Regression achieved the highest test accuracy and ROC-AUC among the three initial models.

Decision Tree achieved the highest recall and marginally the highest F1-score at the default threshold.

Random Forest achieved a similar ROC-AUC to Logistic Regression but identified fewer churned customers at the default threshold.

The differences in ROC-AUC were relatively small, so these results do not establish a statistically significant performance advantage for any individual algorithm.

## 7. Cross-Validation Results

Five-fold stratified cross-validation was performed for Decision Tree and Random Forest using the training data.

| Metric | Decision Tree | Random Forest |
|---|---:|---:|
| Mean Accuracy | 0.7860 | 0.8025 |
| Mean Precision | 0.6073 | 0.6712 |
| Mean Recall | 0.5572 | 0.5037 |
| Mean F1-score | 0.5795 | 0.5752 |
| Mean ROC-AUC | 0.8296 | 0.8480 |

Random Forest achieved the higher mean cross-validated ROC-AUC.

Decision Tree achieved slightly higher mean recall and F1-score.

These findings indicate that model comparisons depend on the evaluation metric and the operating threshold.

## 8. ROC and Precision–Recall Analysis

ROC curves were used to compare the models' ability to separate churned and retained customers across classification thresholds.

The test ROC-AUC results were:

| Model | ROC-AUC |
|---|---:|
| Logistic Regression | 0.8359 |
| Decision Tree | 0.8264 |
| Random Forest | 0.8338 |

Because the churn class represents approximately 26.58% of the dataset, Precision–Recall curves were also examined.

### Average Precision

| Model | Average Precision |
|---|---:|
| Logistic Regression | 0.6228 |
| Decision Tree | 0.6091 |
| Random Forest | 0.6367 |

Random Forest achieved the highest Average Precision, while Logistic Regression achieved the highest ROC-AUC.

The two metrics evaluate different aspects of model performance. Both provide useful information when assessing classifiers on imbalanced data.

## 9. Classification Threshold Analysis

The initial Logistic Regression model used a classification threshold of 0.50.

Alternative thresholds were examined using out-of-fold predictions from five-fold cross-validation on the original training set.

Among the thresholds tested, 0.30 produced the highest training out-of-fold F1-score.

### Exploratory Test-Set Comparison

| Metric | Threshold 0.50 | Threshold 0.30 |
|---|---:|---:|
| Accuracy | 0.8038 | 0.7427 |
| Precision | 0.6485 | 0.5108 |
| Recall | 0.5722 | 0.7594 |
| F1-score | 0.6080 | 0.6108 |
| True Positives | 214 | 284 |
| False Positives | 116 | 272 |
| False Negatives | 160 | 90 |
| True Negatives | 917 | 761 |

Reducing the threshold from 0.50 to 0.30 identified 70 additional churned customers.

However, it also resulted in 156 additional false-positive predictions.

Recall improved considerably, while F1-score increased only slightly.

The appropriate threshold would depend on business priorities, including the costs of missed churn cases and unnecessary retention interventions.

This comparison used the original, untuned Logistic Regression model. It should not be interpreted as an independent test of the subsequently tuned final configuration.

## 10. Hyperparameter Tuning

GridSearchCV was used to tune Logistic Regression on the training set.

The search examined five regularization strengths and two class-weight configurations.

Five-fold stratified cross-validation was used, with ROC-AUC as the selection metric.

### Selected Hyperparameters

| Parameter | Selected Value |
|---|---|
| Regularization strength (`C`) | 10.0 |
| Class weight | None |
| Best mean CV ROC-AUC | 0.8463 |

Several parameter combinations produced similar cross-validation scores.

For example, `C=100.0` with no class weighting achieved a mean ROC-AUC of 0.8462, compared with 0.8463 for the selected configuration.

This difference is too small to support a claim of meaningful superiority.

## 11. Nested Cross-Validation

Nested cross-validation was conducted on the original training subset of 5,625 records to assess the Logistic Regression tuning and threshold-selection procedure.

The evaluation used:

- Five outer folds for performance assessment.
- Three inner folds for hyperparameter tuning.
- ROC-AUC to select regularization strength and class weighting.
- Out-of-fold predictions to select the classification threshold based on F1-score.

The original 1,407-record test set was not used during nested cross-validation.

### Results

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Accuracy | 0.7685 | 0.0109 |
| Precision | 0.5479 | 0.0161 |
| Recall | 0.7445 | 0.0453 |
| F1-score | 0.6306 | 0.0197 |
| ROC-AUC | 0.8463 | 0.0056 |

The nested procedure achieved a mean recall of 74.45% and a mean precision of 54.79%.

ROC-AUC varied relatively little across the outer validation folds.

### Parameter and Threshold Stability

The selected regularization strength varied between `C=10` and `C=100`.

Class weighting was set to `None` in four of the five outer folds and to `balanced` in one fold.

Selected thresholds ranged from 0.30 to 0.55.

These variations indicate that the preferred operating threshold is sensitive to the training sample.

Nested cross-validation provides an estimate of the performance of the Logistic Regression tuning procedure. It does not independently assess the initial choice of Logistic Regression over every other model family examined in the project.

## 12. Random Forest Feature Importance

Impurity-based feature importance scores were extracted from the trained Random Forest model.

The highest-ranked features were:

| Feature | Importance |
|---|---:|
| Tenure | 0.1572 |
| Month-to-month contract | 0.1148 |
| Total charges | 0.1107 |
| Monthly charges | 0.0734 |
| No online security | 0.0583 |
| Fiber optic internet service | 0.0527 |
| No technical support | 0.0473 |
| Electronic check payment method | 0.0430 |
| Two-year contract | 0.0375 |
| No online backup | 0.0225 |

Tenure and contract type ranked prominently, consistent with the exploratory findings.

However, these importance values have limitations.

They do not indicate whether a feature increases or decreases churn probability, and correlated variables may influence one another's importance.

The scores should therefore be treated as model-specific indicators rather than causal explanations.

## 13. Final Model Configuration

A Logistic Regression pipeline was prepared as the final saved model artifact.

The configuration was selected using five-fold cross-validation on the original training subset.

| Setting | Value |
|---|---|
| Algorithm | Logistic Regression |
| Training records | 5,625 |
| Input features | 19 |
| Regularization parameter (`C`) | 10.0 |
| Class weight | None |
| Classification threshold | 0.30 |
| Best tuning CV ROC-AUC | 0.8463 |
| Training out-of-fold F1-score | 0.6344 |
| Training out-of-fold recall | 0.7666 |

The pipeline contains the fitted preprocessing transformations and classification model.

The trained model was saved as:

`models/final_churn_model.joblib`

Its configuration, including the selected threshold, was saved as:

`models/final_model_config.json`

The threshold-selection results were also exported as a CSV report.

The out-of-fold scores used to select the final threshold are not independent estimates of the complete model-selection process. The nested cross-validation results provide a more appropriate assessment of that procedure.

## 14. Example Customer Prediction

The saved model was tested using a synthetic customer profile.

The example customer had the following characteristics:

| Feature | Value |
|---|---|
| Tenure | 3 months |
| Contract | Month-to-month |
| Internet service | Fiber optic |
| Monthly charges | $95.00 |
| Total charges | $285.00 |

The inference script loaded the trained model and its configuration without retraining.

### Prediction Output

| Output | Value |
|---|---|
| Estimated churn probability | 71.45% |
| Classification threshold | 30% |
| Predicted class | Churn (1) |

The example confirms that the saved preprocessing pipeline and classifier can be used together for inference.

Initial input validation was added to check required features, missing values, selected numerical constraints, the senior citizen indicator, and contract type.

The prediction is a model-generated estimate for a synthetic example. It is not evidence that the customer will actually churn, and probability calibration has not yet been assessed.

Additional input validation would be necessary for a production application.

## 15. Limitations and Further Work

The results should be interpreted with several limitations in mind.

### Dataset

The project uses a publicly available sample dataset. The findings should not automatically be generalized to a real telecommunications company's current customers.

### Evaluation

The original test set was examined during several exploratory model comparisons. As a result, those test-set metrics should not be presented as a fully independent assessment of the final selected modeling workflow.

Nested cross-validation was subsequently performed on the original training subset to provide a more reliable assessment of the Logistic Regression tuning procedure.

### Threshold Selection

The classification threshold was selected to improve F1-score in the evaluated training data.

A production threshold would require explicit business requirements and estimates of intervention costs.

### Model Interpretation

Feature importance and exploratory associations do not demonstrate causal effects.

A model identifying customers at risk of churn does not establish which customers would respond to retention campaigns.

### Further Development

Potential next steps include:

- Model calibration and calibration assessment.
- Cost-sensitive threshold selection.
- Additional evaluation of feature importance using permutation-based methods.
- Validation on newly collected, independent customer data.
- More comprehensive input validation and automated testing.
- Monitoring data drift and model performance if the system is deployed.

## 16. Conclusion

This project investigated customer churn using exploratory analysis, several classification algorithms, and a structured evaluation process.

The initial model comparison showed that Logistic Regression, Decision Tree, and Random Forest achieved broadly similar ROC-AUC results, although their precision and recall differed.

Logistic Regression was retained as the primary interpretable benchmark. Random Forest achieved the highest Average Precision in the exploratory test-set comparison.

Adjusting the classification threshold demonstrated how recall can be increased at the cost of additional false-positive predictions.

Nested cross-validation was used to assess the tuning and threshold-selection procedure without relying on the previously examined test set.

The resulting project includes a saved prediction pipeline, reproducible training scripts, evaluation reports, and visualizations.

The analysis illustrates why selecting a classification model requires consideration of several performance measures and the intended application rather than accuracy alone.

## Dataset Source

**IBM Telco Customer Churn**

Source repository: https://github.com/IBM/telco-customer-churn-on-icp4d

The dataset is used for educational analysis and portfolio development. The project is an independent learning exercise and does not imply affiliation with or endorsement by IBM.