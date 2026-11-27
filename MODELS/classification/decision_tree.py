"""Small, regularized decision-tree search."""
from sklearn.tree import DecisionTreeClassifier


def build_model():
    return DecisionTreeClassifier(class_weight="balanced", random_state=42), {
        "model__max_depth": [3, 6],
        "model__min_samples_leaf": [2, 5],
    }
