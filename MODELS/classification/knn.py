"""Distance-weighted nearest neighbors in standardized feature space."""
from sklearn.neighbors import KNeighborsClassifier


def build_model():
    return KNeighborsClassifier(weights="distance"), {
        "model__n_neighbors": [3, 5, 9],
    }
