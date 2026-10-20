
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.models.logistic_regression.model import build_model as build_lr
from src.models.random_forest.model import build_model as build_rf


def test_logistic_regression_model_creation():
    model = build_lr()

    assert isinstance(model, LogisticRegression)


def test_random_forest_model_creation():
    model = build_rf()

    assert isinstance(model, RandomForestClassifier)


def test_logistic_regression_has_fit_and_predict():
    model = build_lr()

    assert callable(model.fit)
    assert callable(model.predict)


def test_random_forest_has_fit_and_predict():
    model = build_rf()

    assert callable(model.fit)
    assert callable(model.predict)
