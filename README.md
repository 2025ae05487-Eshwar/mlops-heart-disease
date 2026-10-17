# Heart Disease Prediction — End-to-End MLOps Project

An end-to-end Machine Learning Operations (MLOps) project for binary heart disease classification using the UCI Cleveland Heart Disease dataset. The project covers data collection, preprocessing, exploratory data analysis, model training, hyperparameter tuning, model evaluation, experiment tracking with MLflow, and model deployment through a FastAPI REST API. Docker and Docker Compose configuration are included for containerization.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objectives](#2-objectives)
3. [Key Features](#3-key-features)
4. [Technology Stack](#4-technology-stack)
5. [System Architecture](#5-system-architecture)
6. [Dataset Description](#6-dataset-description)
7. [Project Structure](#7-project-structure)
8. [Installation and Setup](#8-installation-and-setup)
9. [Data Collection and Preprocessing](#9-data-collection-and-preprocessing)
10. [Exploratory Data Analysis](#10-exploratory-data-analysis)
11. [Model Development](#11-model-development)
12. [Model Evaluation and Results](#12-model-evaluation-and-results)
13. [MLflow Experiment Tracking](#13-mlflow-experiment-tracking)
14. [FastAPI Model Serving](#14-fastapi-model-serving)
15. [API Documentation and Testing](#15-api-documentation-and-testing)
16. [Docker and Docker Compose](#16-docker-and-docker-compose)
17. [Git and Version Control](#17-git-and-version-control)
18. [Reproducibility](#18-reproducibility)
19. [Current Project Status](#19-current-project-status)
20. [Future Enhancements](#20-future-enhancements)
21. [Limitations](#21-limitations)
22. [Disclaimer](#22-disclaimer)
23. [Author](#23-author)

---

## 1. Project Overview

Heart disease is a major global health concern. Machine learning can help analyze clinical measurements and identify patterns associated with the presence of heart disease.

This project develops a binary classification pipeline that predicts whether a patient belongs to the heart disease positive or negative class based on selected clinical attributes.

The project follows an MLOps-oriented workflow, connecting the model development lifecycle with experiment tracking, model persistence, API-based inference, and containerization configuration.

The original dataset contains 303 observations. The original target variable, `num`, represents multiple target values from 0 to 4. For this project, it is transformed into a binary target:

- `0`: No heart disease indicated by the dataset.
- `1`: Heart disease indicated by the dataset.

Two classification algorithms are trained and compared:

- Logistic Regression
- Random Forest Classifier

Hyperparameter tuning and stratified five-fold cross-validation are used to compare model performance. Logistic Regression is selected as the final model based on its mean cross-validation ROC-AUC score.

The selected model is saved as a serialized artifact and integrated into a FastAPI application that exposes prediction, health, and readiness endpoints.

**Project goal:** Build a reproducible, maintainable, and deployable machine learning application rather than stopping at model training alone.

## 2. Objectives

The main objectives are:

1. Collect the heart disease dataset and organize raw and processed data.
2. Clean the dataset and handle missing values.
3. Perform exploratory data analysis to understand the dataset and feature relationships.
4. Build a preprocessing pipeline for numerical and categorical features.
5. Train and compare Logistic Regression and Random Forest models.
6. Apply hyperparameter tuning and stratified cross-validation.
7. Evaluate classification performance using multiple metrics.
8. Track model experiments using MLflow with a local SQLite backend.
9. Persist the selected model and associated metadata.
10. Develop a REST API using FastAPI for model inference.
11. Provide health and readiness endpoints for service checks.
12. Prepare Docker and Docker Compose configuration for reproducible execution.
13. Establish a foundation for automated testing, continuous integration, deployment, and model monitoring.

## 3. Key Features

### Data pipeline
- Raw and processed dataset separation.
- Data cleaning and missing-value handling.
- Binary target transformation.
- Reusable preprocessing components.

### Model development
- Logistic Regression and Random Forest comparison.
- Stratified five-fold cross-validation.
- Hyperparameter tuning.
- Accuracy, precision, recall, F1-score, and ROC-AUC evaluation.
- Model comparison report and evaluation visualizations.

### Experiment tracking
- MLflow experiment organization.
- SQLite-backed tracking storage.
- Parameter and metric logging.
- Model and artifact logging.

### Model serving
- FastAPI REST API.
- Input validation using Pydantic.
- Prediction and positive-class probability output.
- Health and readiness checks.
- Interactive Swagger documentation.

### Deployment preparation
- Python dependency management.
- Dockerfile.
- Docker Compose configuration.
- Git ignore rules for environments, caches, and generated files.

## 4. Technology Stack

| Category | Technology |
|---|---|
| Programming language | Python 3.10 |
| Data manipulation | Pandas, NumPy |
| Machine learning | scikit-learn |
| Model serialization | Joblib |
| Experiment tracking | MLflow |
| Tracking database | SQLite |
| API framework | FastAPI |
| API server | Uvicorn |
| Input validation | Pydantic |
| Data visualization | Matplotlib and Seaborn, where used in EDA |
| Containerization | Docker |
| Multi-container configuration | Docker Compose |
| Version control | Git |
| Development environment | Python virtual environment (`.venv`) |
| Dataset | UCI Cleveland Heart Disease dataset |

Package versions should be maintained in `requirements.txt`. Docker should use Python 3.10 to match the development environment, subject to compatibility with the pinned dependencies.

## 5. System Architecture

The project is organized into five major stages.

### Stage 1: Data ingestion and preprocessing

The raw dataset is collected and stored under `data/raw/`. A preprocessing script cleans the data, handles missing values, and generates the processed dataset under `data/processed/`.

### Stage 2: Model development

Exploratory data analysis is performed to understand the data. A reusable preprocessing pipeline handles numerical and categorical features. Logistic Regression and Random Forest models are trained and evaluated using cross-validation and a held-out test set.

### Stage 3: Experiment tracking

Training parameters, metrics, and relevant artifacts are recorded using MLflow. A local SQLite database stores experiment metadata.

### Stage 4: Model serving

The selected model is saved to `artifacts/best_model.joblib`. The FastAPI application loads this model and exposes REST endpoints for prediction and service health checks.

### Stage 5: Containerization and deployment

A Dockerfile defines how to package the API, dependencies, source code, and model artifact. Docker Compose provides configuration for running the application. Successful container execution should be verified separately before considering this deployment stage complete.

### Architecture flow

```text
                 UCI Heart Disease Dataset
                            |
                            v
                     Data Ingestion
                            |
                            v
                    Raw Dataset (CSV)
                            |
                            v
                  Data Cleaning and
                    Preprocessing
                            |
                            v
                  Processed Dataset
                            |
                            v
                 Exploratory Data Analysis
                            |
                            v
                  Feature Preprocessing
                            |
                            v
              +-------------+-------------+
              |                           |
              v                           v
      Logistic Regression          Random Forest
              |                           |
              +-------------+-------------+
                            |
                            v
                Cross-Validation and
                  Hyperparameter Tuning
                            |
                            v
                  Model Comparison
                            |
                            v
                  Best Model Selection
                            |
                +-----------+-----------+
                |                       |
                v                       v
          MLflow Tracking         Model Artifact
                |                 best_model.joblib
                |                       |
                +-----------+-----------+
                            |
                            v
                     FastAPI Service
                            |
                            v
                  REST Prediction API
                            |
                            v
                  Client / Swagger UI
```

## 6. Dataset Description

### Dataset source

The project uses the Cleveland subset of the UCI Heart Disease dataset.

Dataset information and original documentation:

- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/45/heart+disease

The dataset contains 303 observations and clinical attributes used for heart disease classification.

### Input features

The following 13 input features are used by the prediction API.

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Encoded sex value |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar indicator |
| `restecg` | Resting electrocardiographic result |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina indicator |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels visualized by fluoroscopy |
| `thal` | Thalassemia-related categorical encoding |

The categorical values use the encodings expected by the dataset and trained model. They should not be interpreted as arbitrary integers or replaced with different encodings without updating the preprocessing pipeline.

### Target variable

The original target variable is `num`.

It is transformed as follows:

```python
target = 0 if num == 0 else 1
```

This creates a binary classification problem.

- Class `0`: No heart disease indicated by the dataset.
- Class `1`: Heart disease indicated by the dataset.

### Dataset handling

The project separates raw and processed data:

- `data/raw/heart_disease.csv`: downloaded raw dataset.
- `data/processed/heart_disease_clean.csv`: cleaned dataset prepared for analysis and training.

The raw dataset should be retained as the source of truth. Cleaning and target transformation should be repeatable through the preprocessing script.

## 7. Project Structure

The project is organized into scripts, source modules, model artifacts, reports, and deployment configuration.

```text
mlops_heart_disease/
│
├── data/
│   ├── raw/
│   │   └── heart_disease.csv
│   └── processed/
│       └── heart_disease_clean.csv
│
├── scripts/
│   ├── download_data.py
│   └── preprocess_data.py
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── mlflow_tracking.py
│   ├── train.py
│   │
│   └── models/
│       ├── logistic_regression/
│       │   └── model.py
│       └── random_forest/
│           └── model.py
│
├── artifacts/
│   ├── best_model.joblib
│   └── model_info.json
│
├── reports/
│   ├── model_comparison.csv
│   └── figures/
│
├── tests/                     # Automated tests to be added
│
├── Dockerfile
├── docker-compose.yml         # Or the Compose filename actually used
├── .dockerignore
├── .gitignore
├── requirements.txt
├── mlflow.db                  # Generated local tracking database
└── README.md
```

Some files and directories are generated only after the corresponding scripts run. The `tests/` directory is a planned location for automated tests.

## 8. Installation and Setup

### Prerequisites

Install the following tools:

- Python 3.10
- Git
- A code editor such as Visual Studio Code
- Docker Desktop for Windows, with Docker Compose, for container execution

Docker Desktop must be installed and running before Docker commands can be executed.

### Step 1: Clone the repository

If the project is hosted on GitHub:

```bash
git clone <YOUR_REPOSITORY_URL>
cd mlops_heart_disease
```

Replace `<YOUR_REPOSITORY_URL>` with the actual repository URL.

If you are working with the existing local project, navigate to the project root instead.

### Step 2: Create a virtual environment

On Windows Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Verify the Python version:

```bash
python --version
```

Expected major and minor version:

```text
Python 3.10.x
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `requirements.txt` has not been generated yet, install the packages required by the source code and generate the dependency file:

```bash
python -m pip freeze > requirements.txt
```

For reproducibility, review and maintain the dependency versions rather than relying indefinitely on an unreviewed environment snapshot.

### Step 4: Verify the environment

```bash
python -c "import sys; print(sys.executable)"
```

The output should point to the project's `.venv` environment.

## 9. Data Collection and Preprocessing

### Download the dataset

Run the data download script from the project root:

```bash
python scripts/download_data.py
```

The script downloads the raw dataset to:

```text
data/raw/heart_disease.csv
```

Verify that the file exists before running preprocessing.

### Preprocess the data

Run:

```bash
python scripts/preprocess_data.py
```

The preprocessing step creates the cleaned dataset:

```text
data/processed/heart_disease_clean.csv
```

The workflow handles missing values and prepares the target for binary classification.

### Model preprocessing pipeline

The reusable preprocessing implementation is in `src/preprocessing.py`.

It uses a `ColumnTransformer` with separate transformations:

**Numerical features**
- Median imputation for missing values.
- Standard scaling.

**Categorical features**
- Most-frequent-value imputation.
- One-hot encoding with `handle_unknown="ignore"`.

The preprocessing transformations are incorporated into the model pipeline so that training and inference use consistent feature processing.

Using a pipeline also reduces the risk of applying different transformations to training data and prediction requests.

## 10. Exploratory Data Analysis

Exploratory data analysis is implemented in:

```text
src/eda.py
```

EDA helps investigate the dataset before model development.

Areas to examine include:

- Dataset dimensions and feature types.
- Missing-value distribution.
- Numerical feature distributions.
- Target class distribution.
- Relationships between input features and the target.
- Correlations among numerical features.
- Potential outliers and unusual values.

Visualizations generated by the project are stored under:

```text
reports/figures/
```

The exact figures depend on the plots generated by the EDA and training scripts.

EDA is useful for understanding the data, but observed correlations should not be interpreted as evidence of causation.

## 11. Model Development

The training workflow is implemented in:

```text
src/train.py
```

Model definitions are separated into their respective modules.

### 11.1 Logistic Regression

Implementation:

```text
src/models/logistic_regression/model.py
```

The model uses scikit-learn's `LogisticRegression` with a maximum iteration limit of 2,000 and a random state of 42 in its default configuration.

Logistic Regression is a useful baseline for binary classification because it is relatively simple, efficient, and easier to interpret than many more complex models.

The tuning workflow evaluates different hyperparameter settings, including the regularization parameter `C` and class weighting.

### 11.2 Random Forest

Implementation:

```text
src/models/random_forest/model.py
```

The default configuration uses:

- 200 decision trees.
- Balanced class weights.
- Random state 42.
- Parallel processing through `n_jobs=-1`.

Random Forest can capture nonlinear relationships and interactions between features.

### 11.3 Training and validation strategy

The training workflow uses:

- A held-out test set.
- Stratified five-fold cross-validation.
- Grid search for hyperparameter tuning.
- A common preprocessing approach.
- Multiple performance metrics for model comparison.

Stratification helps preserve class proportions across the validation folds.

The held-out test set is used to assess the selected model's performance on data not used to fit that model. Cross-validation provides an additional basis for comparing model configurations.

### 11.4 Model selection

The selection criterion is the mean cross-validation ROC-AUC score.

Logistic Regression was selected because it achieved a mean cross-validation ROC-AUC of **0.9113**, higher than the Random Forest result recorded in the training run.

The selected model is saved to:

```text
artifacts/best_model.joblib
```

Model metadata is saved to:

```text
artifacts/model_info.json
```

The model comparison report is saved to:

```text
reports/model_comparison.csv
```

## 12. Model Evaluation and Results

The following results are from the recorded successful training run. They should be treated as experiment results for this particular dataset split and configuration, not as guaranteed performance on other populations.

### 12.1 Cross-validation results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Mean CV accuracy | 0.8511 | 0.8139 |
| Mean CV precision | 0.8776 | 0.8210 |
| Mean CV recall | 0.7921 | 0.7644 |
| Mean CV ROC-AUC | 0.9113 | 0.8985 |

### 12.2 Held-out test results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.8852 | 0.8689 |
| Precision | 0.8387 | 0.8333 |
| Recall | 0.9286 | 0.8929 |
| F1-score | 0.8814 | 0.8621 |
| ROC-AUC | 0.9610 | 0.9470 |

### 12.3 Interpretation

**Accuracy** measures the proportion of correct predictions across both classes.

**Precision** measures the proportion of positive predictions that are actually positive in the evaluation data.

**Recall** measures the proportion of actual positive cases correctly identified by the model.

**F1-score** balances precision and recall using their harmonic mean.

**ROC-AUC** measures how well the model ranks positive cases above negative cases across classification thresholds.

Logistic Regression achieved better recorded cross-validation ROC-AUC, accuracy, precision, recall, and ROC-AUC on the held-out test set than Random Forest.

The test recall of 0.9286 indicates that the selected model identified a high proportion of positive cases in this test split. However, the precision of 0.8387 also indicates that some positive predictions were false positives.

Because this is a health-related classification problem, threshold selection and the relative costs of false negatives and false positives require careful consideration before any real-world application.

The held-out test metrics should be interpreted alongside the dataset size, split strategy, and limitations of the underlying data.

## 13. MLflow Experiment Tracking

MLflow is used to track model training experiments.

The tracking utilities are implemented in:

```text
src/mlflow_tracking.py
```

The experiment name is:

```text
Heart Disease Classification
```

### 13.1 Tracking backend

The project uses a local SQLite tracking database:

```text
mlflow.db
```

The tracking URI is configured using a SQLite connection string.

This approach avoids relying on MLflow's legacy filesystem-based tracking backend.

### 13.2 Information tracked

The tracking utilities support logging:

- Model and experiment tags.
- Training parameters.
- Evaluation metrics.
- Model artifacts.
- Other selected files generated during training.

The training script integrates with these utilities to record model runs.

### 13.3 Start the MLflow UI

From the project root, with the virtual environment activated:

```bash
mlflow ui --backend-store-uri "sqlite:///mlflow.db" --host 127.0.0.1 --port 5000 --workers 1
```

Open:

http://127.0.0.1:5000

The single-worker setting was used to avoid a Windows socket error encountered with the default worker configuration.

The UI can be used to inspect recorded runs, compare metrics, and review logged parameters and artifacts.

Stop the UI with `Ctrl+C` when finished.

### 13.4 Model serialization

The model is saved locally with Joblib. MLflow model logging uses cloudpickle serialization in the configured helper.

Serialized model files should only be loaded from trusted sources because deserialization can execute unsafe code in some circumstances.

### 13.5 Tracking persistence

The SQLite database contains local experiment metadata. It should be handled separately from the model artifact and source code.

For a shared or production environment, a managed tracking server and appropriate artifact storage can be considered later.

## 14. FastAPI Model Serving

The API implementation is:

```text
src/api.py
```

The application loads the saved model from:

```text
artifacts/best_model.joblib
```

The API provides prediction functionality and service checks.

### 14.1 Start the API locally

Activate the virtual environment and run:

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Keep the terminal open while testing the service.

### 14.2 Available endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Returns basic API information |
| GET | `/health` | Checks whether the API process responds |
| GET | `/ready` | Checks whether the model is loaded |
| POST | `/predict` | Returns a model prediction |

### 14.3 Root endpoint

Request:

```http
GET /
```

Returns basic service information, including links to the API documentation and health endpoints.

### 14.4 Health endpoint

Request:

```http
GET /health
```

Expected response:

```json
{
  "status": "ok"
}
```

This endpoint checks whether the API process is responding. It does not independently establish that the model is ready for predictions.

### 14.5 Readiness endpoint

Request:

```http
GET /ready
```

Expected response when the model is loaded:

```json
{
  "status": "ready",
  "model": "best_model.joblib"
}
```

If the model is unavailable, the endpoint returns HTTP 503.

### 14.6 Prediction endpoint

Endpoint:

```http
POST /predict
```

The request body contains the 13 model input features.

Example request:

```json
{
  "age": 63,
  "sex": 1,
  "cp": 1,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 2,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 3,
  "ca": 0,
  "thal": 6
}
```

This is a sample software-testing payload, not real patient data.

### 14.7 Prediction response

The response contains fields similar to:

```json
{
  "prediction": 1,
  "label": "Heart disease detected by model",
  "positive_class_probability": 0.85,
  "disclaimer": "This is a machine-learning prediction, not a medical diagnosis."
}
```

The prediction and probability shown above are illustrative only. Actual values depend on the model and supplied input.

Response fields:

- `prediction`: predicted binary class, either `0` or `1`.
- `label`: human-readable representation of the predicted class.
- `positive_class_probability`: model-estimated probability for class `1`.
- `disclaimer`: reminder that the output is not a medical diagnosis.

The probability is model-dependent and is not necessarily a calibrated estimate of an individual's clinical risk.

### 14.8 Input validation

The request schema uses Pydantic field constraints and encoded categories to validate the request.

Examples of validation include:

- Age must be within the configured range.
- Binary features must use the expected binary values.
- Numerical measurements must satisfy their configured constraints.
- All required features must be provided.

Invalid requests should return a validation error rather than being silently accepted.

## 15. API Documentation and Testing

FastAPI automatically generates an interactive OpenAPI documentation interface.

Once the API is running, open:

http://127.0.0.1:8000/docs

The Swagger UI provides interactive access to the four implemented endpoints.

### Manual verification checklist

- [x] Open the Swagger UI.
- [x] Confirm `/`, `/health`, `/ready`, and `/predict` are listed.
- [x] Confirm the readiness endpoint reports the model as loaded.
- [x] Execute a prediction request.
- [x] Inspect the prediction response.

### Recommended automated tests

The next testing stage should include:

1. Health endpoint returns HTTP 200.
2. Readiness endpoint returns HTTP 200 when the model is loaded.
3. A valid prediction request returns HTTP 200.
4. The prediction is either `0` or `1`.
5. The positive-class probability is between `0` and `1`.
6. Missing required fields are rejected.
7. Invalid categorical values are rejected.
8. The model-loading failure path returns an appropriate readiness response.
9. The API handles unexpected inference failures safely.

The automated test suite should be implemented under `tests/` and run with pytest.

Example test command, after tests have been created:

```bash
python -m pytest -v
```

Manual Swagger verification has been completed. A passing automated test suite should only be claimed after these tests are implemented and executed successfully.

## 16. Docker and Docker Compose

Docker is intended to package the application and its dependencies into a reproducible environment.

### 16.1 Dockerfile

The Dockerfile is configured to use Python 3.10 and:

- Set the working directory.
- Install dependencies from `requirements.txt`.
- Copy the application source code.
- Copy the saved model artifacts.
- Expose port 8000.
- Start the API using Uvicorn.

The container starts the application using:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Binding to `0.0.0.0` inside the container allows the service to accept connections through a published Docker port.

### 16.2 Docker Compose

The Docker Compose file defines the configuration for running the application.

Validate the configuration:

```bash
docker compose config
```

Build and start the services:

```bash
docker compose up --build
```

Run in the background:

```bash
docker compose up --build -d
```

Inspect the running services:

```bash
docker compose ps
```

View logs:

```bash
docker compose logs -f
```

Stop the services:

```bash
docker compose down
```

The exact service names, volume mappings, environment variables, and port mappings depend on the contents of the actual Compose file.

### 16.3 Verify the containerized application

After the container starts, test:

- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/ready

Then execute a prediction through Swagger.

A successful local API test does not by itself prove that the Docker image works. Containerized execution must be verified independently.

### 16.4 Current Docker status

The Dockerfile and Docker Compose configuration have been created. Docker execution remains pending verification because the Docker CLI was not recognized in Git Bash at the last check.

Install or start Docker Desktop for Windows, then verify:

```bash
docker --version
docker compose version
docker info
```

Once those commands succeed, continue with the Compose commands above.

### 16.5 Important artifact consideration

The Dockerfile copies the `artifacts/` directory into the image. Therefore, `artifacts/best_model.joblib` must exist locally when the image is built.

If model files are excluded from Git, a fresh repository clone will need to obtain or regenerate the model artifact before the container can run.

## 17. Git and Version Control

Git is used to manage the project source code and track changes.

### Initialize Git

From the project root:

```bash
git init
```

Check the repository state:

```bash
git status
```

### `.gitignore`

The `.gitignore` file excludes files that should not normally be tracked, including:

- Python virtual environments.
- Python cache files.
- Editor and operating-system files.
- Environment variables and local secrets.
- Local MLflow database files.
- Generated model files when configured to exclude them.
- Generated plots and temporary files.

This prevents thousands of environment and cache files from being added accidentally.

### `.dockerignore`

The `.dockerignore` file excludes unnecessary files from the Docker build context, such as:

- `.venv/`
- `.git/`
- `__pycache__/`
- Local MLflow tracking files.

The two ignore files serve different purposes: `.gitignore` controls Git tracking, while `.dockerignore` controls what Docker sends to the build process.

### Recommended Git workflow

```bash
git status
git add .gitignore .dockerignore Dockerfile docker-compose.yml requirements.txt README.md src scripts
git status
```

Review the staged changes before committing. Add tests and other required project files once they exist.

Then commit:

```bash
git commit -m "Add heart disease MLOps pipeline and FastAPI service"
```

Do not commit secrets, private environment files, or an entire virtual environment.

If the saved model is excluded from Git, document how to regenerate or obtain it so another developer can reproduce the deployment.

## 18. Reproducibility

The project includes several mechanisms that support repeatable execution:

- A dedicated Python virtual environment.
- A dependency file.
- Separate data ingestion and preprocessing scripts.
- Reusable preprocessing components.
- Fixed random-state settings in the model configurations.
- Stratified cross-validation.
- Recorded training parameters and evaluation metrics.
- A saved model artifact.
- A structured API input schema.
- Docker configuration for packaging the service.

For stronger reproducibility, future improvements should include:

- Pinning compatible package versions.
- Recording the dataset source and version.
- Tracking data and model artifact checksums.
- Recording the train/test split configuration.
- Persisting the selected model's complete hyperparameters.
- Testing inference consistency between local Python and Docker.
- Using a reproducible build and automated test pipeline.

## 19. Current Project Status

The status below reflects the work completed and reported so far.

| Component | Status |
|---|---|
| Data download script | Completed |
| Data preprocessing script | Completed |
| Exploratory data analysis | Completed |
| Logistic Regression training | Completed |
| Random Forest training | Completed |
| Hyperparameter tuning and cross-validation | Completed |
| Model comparison and evaluation | Completed |
| Best model selection | Completed |
| Model artifact and metadata | Generated |
| MLflow SQLite tracking | Configured and used |
| MLflow UI | Successfully opened |
| FastAPI application | Implemented and running |
| Swagger endpoint verification | Completed |
| Dockerfile | Created |
| Docker Compose file | Created; execution not yet verified |
| `.gitignore` | Created |
| `.dockerignore` | Created |
| Automated pytest suite | Pending |
| GitHub Actions CI/CD | Pending |
| Kubernetes deployment | Pending |
| Production monitoring | Pending |

## 20. Future Enhancements

### Automated testing

Implement pytest tests for preprocessing, model predictions, API responses, invalid inputs, and model loading.

### Continuous integration and delivery

Configure GitHub Actions to install dependencies and run automated tests on pushes and pull requests.

A later pipeline can build the Docker image and publish it to a container registry after all required checks pass.

### Kubernetes deployment

Deploy the container to Kubernetes and configure:

- Deployments and Services.
- Readiness and liveness probes.
- Resource requests and limits.
- Configuration and secret management.
- Replicas and scaling policies.

### Monitoring and observability

Extend the service with structured logging and monitoring for:

- Request counts and errors.
- Response latency.
- Model availability.
- Input validation failures.
- Prediction distribution changes.
- Data drift and model performance when suitable ground-truth labels become available.

A dedicated metrics endpoint can be added if required.

### Model governance

Improve model governance by recording model versions, training datasets, evaluation results, and approval criteria. Any model replacement should pass the agreed evaluation and validation checks before deployment.

### Model improvement

Explore threshold tuning, calibration, feature analysis, class-specific performance, and validation on independent datasets.

Any improvement should be assessed with a consistent evaluation strategy rather than relying solely on one test split.

## 21. Limitations

- The dataset contains only 303 observations, limiting the conclusions that can be drawn about generalization.
- The evaluation metrics depend on the dataset split and training configuration.
- The model may not generalize to populations or clinical settings that differ from the original dataset.
- Encoded features can be difficult to interpret without the dataset's original coding definitions.
- A high ROC-AUC does not guarantee that predicted probabilities are calibrated.
- The model has not been established as a clinically validated diagnostic system.
- Production-grade security, authentication, load testing, and operational monitoring require additional work.
- Docker configuration has been prepared, but successful container execution must be confirmed independently.

## 22. Disclaimer

This project is intended for educational, research, and software engineering purposes.

The model's output is a machine-learning classification based on a historical dataset. It is not a medical diagnosis, a substitute for professional clinical evaluation, or a recommendation for treatment.

Do not use this application to make healthcare decisions or to determine whether a person has heart disease without appropriate clinical assessment and independent validation.

## 23. Author

**Donifa Babu B M**

Software Developer | Cloud & Data Platform Engineer

The project demonstrates the integration of machine learning development, experiment tracking, model serialization, REST API development, and containerization practices in an end-to-end MLOps workflow.

---

## Acknowledgements

- UCI Machine Learning Repository for the Heart Disease dataset.
- scikit-learn for preprocessing, model training, and evaluation.
- MLflow for experiment tracking.
- FastAPI for API development.
- Docker for application containerization.
