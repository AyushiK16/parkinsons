"""Regularized logistic regression; preprocessing is added by the trainer."""
from sklearn.linear_model import LogisticRegression


def build_model():
    return LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42), {
        "model__C": [0.1, 1.0, 10.0],
    }
