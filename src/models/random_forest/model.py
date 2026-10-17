
"""Random Forest model for heart disease classification."""

from sklearn.ensemble import RandomForestClassifier


def build_model(random_state: int = 42) -> RandomForestClassifier:
    """Return an untrained Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
