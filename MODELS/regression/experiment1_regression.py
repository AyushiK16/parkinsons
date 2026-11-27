"""Run the standalone patient-disjoint motor-UPDRS regression experiment."""
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# ============================================================
# SETTINGS
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "DATA" / "raw" / "UCI_Oxford_Voice" / "telemonitoring" / "parkinsons_updrs.data"
RESULTS_DIR = BASE_DIR / "RESULTS" / "regression"

TARGET = "motor_UPDRS"
GROUP = "subject#"
N_SPLITS = 7

FEATURES = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "Shimmer:APQ11", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
]

# ============================================================
# STEP 1: LOAD AND CHECK DATA
# ============================================================
print("=" * 60)
print("STEP 1: LOAD AND CHECK DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("Unique patients:", df[GROUP].nunique())
print("Missing values:", df.isnull().sum().sum())

FORBIDDEN = ["total_UPDRS", "subject#", "age", "sex", "test_time", "motor_UPDRS"]
bad = [c for c in FEATURES if c in FORBIDDEN]
assert not bad, f"Forbidden columns used as features: {bad}"

missing_cols = [c for c in FEATURES + [TARGET, GROUP] if c not in df.columns]
assert not missing_cols, f"Missing columns: {missing_cols}"

assert df[FEATURES + [TARGET, GROUP]].notnull().all().all(), \
    "Missing values found in required columns."

X = df[FEATURES]
y = df[TARGET]
groups = df[GROUP]

print("\nNumber of input features:", X.shape[1])
print("Target:", TARGET)
print("X shape:", X.shape, "| y shape:", y.shape)
print("Safety check passed: only the 16 voice features are inputs.")

# ============================================================
# STEP 2: PATIENT-WISE FOLDS
# ============================================================
print("\n" + "=" * 60)
print(f"STEP 2: {N_SPLITS}-FOLD PATIENT-WISE SPLITS")
print("=" * 60)

gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(X, y, groups=groups))

tested_patients = []

for fold_no, (train_idx, test_idx) in enumerate(folds, start=1):

    train_patients = set(groups.iloc[train_idx])
    test_patients = set(groups.iloc[test_idx])

    assert train_patients.isdisjoint(test_patients), \
        "Patient leakage detected!"

    tested_patients.extend(test_patients)

    print(
        f"Fold {fold_no}: train patients={len(train_patients)}, "
        f"test patients={len(test_patients)}, "
        f"test recordings={len(test_idx)}"
    )

    print("   Test patient IDs:", sorted(int(p) for p in test_patients))

assert len(tested_patients) == len(set(tested_patients)) == groups.nunique()

print("\nCheck passed: no patient leakage.")
print("Every patient is tested exactly once.")

# ============================================================
# EVALUATION HELPERS
# ============================================================
def score(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def run_cv(name, model):

    oof_pred = np.zeros(len(df))
    fold_scores = []
    fold_baseline_maes = []

    for fold_no, (train_idx, test_idx) in enumerate(folds, start=1):

        m = clone(model)

        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = m.predict(X.iloc[test_idx])

        oof_pred[test_idx] = pred

        fold_scores.append(score(y.iloc[test_idx], pred))

        # Compare against the training-fold mean baseline
        baseline_pred = np.full(
            len(test_idx),
            y.iloc[train_idx].mean()
        )

        fold_baseline_maes.append(
            mean_absolute_error(y.iloc[test_idx], baseline_pred)
        )

    fold_scores = np.array(fold_scores)

    pooled = score(y, oof_pred)

    return {
        "Model": name,
        "Pooled MAE": pooled[0],
        "Pooled RMSE": pooled[1],
        "Pooled R2": pooled[2],
        "Mean fold MAE": fold_scores[:, 0].mean(),
        "Std fold MAE": fold_scores[:, 0].std(ddof=1),
        "Mean fold R2": fold_scores[:, 2].mean(),
        "Folds beating baseline": sum(
            model_mae < baseline_mae
            for model_mae, baseline_mae
            in zip(fold_scores[:, 0], fold_baseline_maes)
        ),
    }, oof_pred, fold_scores


# ============================================================
# STEP 3: MODELS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: TRAINING-MEAN BASELINE AND DEFAULT MODELS")
print("=" * 60)

models = {
    "Training-mean baseline": DummyRegressor(strategy="mean"),

    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression()),
    ]),

    "Decision Tree (default)": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest (default)": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
}

results = []
predictions = {}
fold_results = {}

for name, model in models.items():

    result, pred, fold_scores = run_cv(name, model)

    results.append(result)
    predictions[name] = pred
    fold_results[name] = fold_scores

    print(f"Finished: {name}")

# ============================================================
# STEP 4: MODEL COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: MODEL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame(results)

baseline_mae = comparison.loc[
    comparison["Model"] == "Training-mean baseline",
    "Pooled MAE"
].iloc[0]

comparison["Beats baseline (pooled)?"] = (
    comparison["Pooled MAE"] < baseline_mae
)

print("\nPooled and fold-level results:")
print(comparison.round(4).to_string(index=False))

# ============================================================
# STEP 5: PATIENT-LEVEL ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: PATIENT-LEVEL PREDICTION ANALYSIS")
print("=" * 60)

patient_results = []

for model_name, pred in predictions.items():

    for patient_id in sorted(groups.unique()):

        mask = groups.to_numpy() == patient_id

        actual = y.to_numpy()[mask]
        predicted = pred[mask]

        patient_results.append({
            "Model": model_name,
            "Patient ID": int(patient_id),
            "Actual Mean UPDRS": actual.mean(),
            "Predicted Mean UPDRS": predicted.mean(),
            "Patient MAE": mean_absolute_error(actual, predicted),
        })

patient_df = pd.DataFrame(patient_results)

for model_name in models:

    print(f"\n{model_name}")
    print("-" * 60)

    print(
        patient_df[patient_df["Model"] == model_name]
        .drop(columns=["Model"])
        .round(3)
        .to_string(index=False)
    )

# ============================================================
# STEP 6: SAVE RESULTS
# ============================================================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
comparison_path = RESULTS_DIR / "experiment1_model_comparison.csv"
patient_path = RESULTS_DIR / "experiment1_patient_results.csv"

comparison.to_csv(comparison_path, index=False)
patient_df.to_csv(patient_path, index=False)

print("\n" + "=" * 60)
print("EXPERIMENT 1 COMPLETED")
print("=" * 60)

print("Saved:")
print(f"- {comparison_path}")
print(f"- {patient_path}")
