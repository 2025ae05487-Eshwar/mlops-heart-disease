
"""Logistic Regression model for heart disease classification."""

from sklearn.linear_model import LogisticRegression


def build_model(random_state: int = 42) -> LogisticRegression:
    """Return an untrained Logistic Regression classifier."""
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )
