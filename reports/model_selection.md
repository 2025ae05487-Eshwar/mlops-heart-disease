# Model Development, Hyperparameter Tuning and Selection

## 1. Objective

The objective is to develop and compare two classification models for binary heart disease prediction using the UCI Heart Disease dataset. Logistic Regression and Random Forest are evaluated using stratified cross-validation and a separate held-out test set.

## 2. Dataset and Experimental Setup

The cleaned dataset contains 303 records and 13 input features. The target variable is binary:

- `0`: No heart disease
- `1`: Heart disease

The dataset contains 164 negative cases and 139 positive cases.

The data was divided using a stratified 80:20 train-test split with `random_state=42`. This produced 242 training records and 61 test records. Stratification preserves the approximate class distribution in both subsets.

Five-fold stratified cross-validation was applied to the training subset during hyperparameter tuning. The held-out test set was kept separate from model selection.

## 3. Feature Preparation

The model pipeline performs preprocessing and classification together to maintain consistent transformations during training and inference.

### Numerical features

The numerical features are:

- Age
- Resting blood pressure (`trestbps`)
- Serum cholesterol (`chol`)
- Maximum heart rate achieved (`thalach`)
- ST depression (`oldpeak`)
- Number of major vessels (`ca`)

Missing numerical values are imputed using the median, and the features are scaled using `StandardScaler`.

### Categorical features

The categorical features are:

- Sex (`sex`)
- Chest pain type (`cp`)
- Fasting blood sugar (`fbs`)
- Resting electrocardiographic results (`restecg`)
- Exercise-induced angina (`exang`)
- ST-segment slope (`slope`)
- Thalassemia-related result (`thal`)

Missing categorical values are imputed using the most frequent value. `OneHotEncoder(handle_unknown="ignore")` converts categorical variables into numerical features and supports previously unseen categories during inference.

The preprocessing transformations are fitted within the scikit-learn pipeline, so cross-validation fits them independently on each training fold and avoids fitting transformations on the held-out test set.

## 4. Models and Hyperparameter Tuning

Two classification algorithms were evaluated.

### 4.1 Logistic Regression

Logistic Regression provides a linear classification baseline. The regularization parameter `C` and class-weight setting were tuned using GridSearchCV.

Search space:

- `C`: 0.1, 1.0, 10.0
- `class_weight`: None, balanced

**Selected hyperparameters:**

- `C`: 0.1
- `class_weight`: None

The selected configuration uses stronger regularization than configurations with larger `C` values.

### 4.2 Random Forest

Random Forest combines multiple decision trees and can capture nonlinear relationships between input features.

Search space:

- `n_estimators`: 100, 200
- `max_depth`: None, 5, 10
- `min_samples_split`: 2, 5
- `class_weight`: None, balanced

**Selected hyperparameters:**

- `n_estimators`: 100
- `max_depth`: 5
- `min_samples_split`: 2
- `class_weight`: None

The selected maximum depth limits the complexity of individual trees relative to unrestricted trees.

### 4.3 Tuning and selection procedure

GridSearchCV evaluates the specified parameter combinations using five-fold stratified cross-validation. Mean cross-validation ROC-AUC is the primary selection metric. The best configuration is refitted on the complete training subset before evaluation on the held-out test set.

## 5. Evaluation Metrics

The following metrics are used:

- **Accuracy:** Proportion of all predictions that are correct.
- **Precision:** Proportion of positive predictions that are correct.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well the model distinguishes the two classes across classification thresholds.

Cross-validation accuracy, precision, recall and ROC-AUC are reported for each model. Held-out test accuracy, precision, recall, F1-score and ROC-AUC are also reported.

## 6. Experimental Results

### 6.1 Cross-validation results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.8511 | 0.8139 |
| Precision | 0.8776 | 0.8210 |
| Recall | 0.7921 | 0.7644 |
| ROC-AUC | 0.9113 | 0.8985 |

Logistic Regression achieved the higher mean cross-validation ROC-AUC of 0.9113, compared with 0.8985 for Random Forest. It also achieved higher cross-validation accuracy, precision and recall.

### 6.2 Held-out test results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.8852 | 0.8689 |
| Precision | 0.8387 | 0.8333 |
| Recall | 0.9286 | 0.8929 |
| F1-score | 0.8814 | 0.8621 |
| ROC-AUC | 0.9610 | 0.9470 |

Logistic Regression achieved 88.52% test accuracy and 0.9610 test ROC-AUC. Its test recall of 92.86% indicates that it correctly identified 26 of the 28 positive cases in this test set. Random Forest achieved 86.89% accuracy, 0.9470 ROC-AUC and 89.29% recall.

The test results favor Logistic Regression across all five reported metrics.

## 7. Final Model Selection

**Selected model: Logistic Regression**

The selection was based on the highest mean five-fold cross-validation ROC-AUC, rather than on held-out test performance.

The selected Logistic Regression configuration achieved a cross-validation ROC-AUC of 0.9113, compared with 0.8985 for Random Forest. It also achieved higher cross-validation accuracy, precision and recall. Its held-out test results were stronger across all reported test metrics.

For this experiment, Logistic Regression therefore offers the strongest overall performance of the two evaluated models. Its relatively high test recall is also a useful characteristic for a task in which false-negative predictions matter.

This selection is based on one train-test split and the configured hyperparameter search. Further validation on independent data would be necessary before considering the model for real-world clinical use.

## 8. Generated Artifacts and Visualizations

The training workflow generates the following outputs:

- `artifacts/best_model.joblib`: Selected fitted preprocessing-and-classification pipeline.
- `artifacts/model_info.json`: Selected model, hyperparameters and metrics.
- `reports/model_comparison.csv`: Comparison results for both models.
- `reports/figures/model_performance/logistic_regression_confusion_matrix.png`: Logistic Regression confusion matrix.
- `reports/figures/model_performance/logistic_regression_roc_curve.png`: Logistic Regression ROC curve.
- `reports/figures/model_performance/logistic_regression_metrics.png`: Logistic Regression test-metric chart.
- `reports/figures/model_performance/random_forest_confusion_matrix.png`: Random Forest confusion matrix.
- `reports/figures/model_performance/random_forest_roc_curve.png`: Random Forest ROC curve.
- `reports/figures/model_performance/random_forest_metrics.png`: Random Forest test-metric chart.
- `reports/figures/model_performance/model_comparison.png`: Test-metric comparison chart.
- `reports/figures/model_performance/cv_roc_auc_comparison.png`: Cross-validation ROC-AUC comparison chart.

## 9. Limitations

The results are based on a single stratified train-test split of 303 records. Performance may vary with a different split or an independent dataset. The reported metrics should not be interpreted as evidence of clinical readiness. Further external validation, assessment of calibration, threshold selection, fairness and clinical safety would be required before any real-world application.

The results document the current experiment and should be updated if the dataset, feature-processing pipeline, hyperparameter search space or training configuration changes.
