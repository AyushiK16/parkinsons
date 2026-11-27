"""Shallow gradient boosting, with four restrained candidate settings."""
from sklearn.ensemble import GradientBoostingClassifier


def build_model():
    return GradientBoostingClassifier(n_estimators=80, random_state=42), {
        "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [1, 2],
    }
