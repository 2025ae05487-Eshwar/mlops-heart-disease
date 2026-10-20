
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor():
    numeric_features = ["age", "chol"]
    categorical_features = ["sex", "cp"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def sample_data():
    return pd.DataFrame({
        "age": [50, 60, np.nan, 45],
        "chol": [200, np.nan, 240, 180],
        "sex": [1, 0, 1, 0],
        "cp": [0, 1, 2, 1],
    })


def test_preprocessing_handles_missing_values():
    data = sample_data()
    preprocessor = make_preprocessor()

    transformed = preprocessor.fit_transform(data)

    assert not np.isnan(transformed).any()


def test_preprocessing_preserves_row_count():
    data = sample_data()
    preprocessor = make_preprocessor()

    transformed = preprocessor.fit_transform(data)

    assert transformed.shape[0] == len(data)


def test_preprocessing_produces_numeric_output():
    data = sample_data()
    preprocessor = make_preprocessor()

    transformed = preprocessor.fit_transform(data)

    assert np.issubdtype(transformed.dtype, np.number)


def test_preprocessing_accepts_unseen_category():
    train_data = sample_data()
    preprocessor = make_preprocessor()
    preprocessor.fit(train_data)

    new_data = pd.DataFrame({
        "age": [55],
        "chol": [220],
        "sex": [1],
        "cp": [99],
    })

    transformed = preprocessor.transform(new_data)

    assert transformed.shape[0] == 1
