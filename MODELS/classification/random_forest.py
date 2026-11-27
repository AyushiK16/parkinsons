"""Deterministic random forest with a modest four-candidate grid."""
from sklearn.ensemble import RandomForestClassifier


def build_model():
    return RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=1
    ), {
        "model__max_depth": [None, 6],
        "model__min_samples_leaf": [2, 5],
    }
