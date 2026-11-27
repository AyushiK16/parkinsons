"""SVM without hidden, recording-level probability calibration."""
from sklearn.svm import SVC


def build_model():
    return SVC(kernel="rbf", probability=False, class_weight="balanced", random_state=42), {
        "model__C": [0.1, 1.0, 10.0],
        "model__gamma": ["scale"],
    }
