"""
Parkinson's Disease Severity Prediction
Person 2 (extra model): Support Vector Regression (SVR, RBF kernel)

Same 16 voice features, target and patient split as Person 1.
SVR needs scaling, so StandardScaler is inside a Pipeline
(fitted on training data only, also inside cross-validation).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "parkinsons_updrs.data"
OUTPUT_DIR = BASE_DIR / "outputs" / "svr"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

TARGET = "total_UPDRS"
PATIENT_ID = "subject#"

FEATURES = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "Shimmer:APQ11", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
]

# Results from teammates / earlier runs (same strict patient split)
OTHER_RESULTS = [
    {"Model": "Linear Regression (Person 1)", "MAE": 7.8890, "RMSE": 9.6853, "R2": -0.1321},
    {"Model": "Random Forest tuned (Person 2)", "MAE": 7.6906, "RMSE": 9.1042, "R2": -0.0003},
    {"Model": "XGBoost (Person 3)", "MAE": 7.5306, "RMSE": 9.1361, "R2": -0.0073},
]


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 65)
print("STEP 1: LOAD AND CHECK DATA")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
required = [PATIENT_ID, TARGET] + FEATURES
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

print("Shape:", df.shape)
print("Unique patients:", df[PATIENT_ID].nunique())
print("Missing values:", df.isnull().sum().sum())


# ============================================================
# 3. PATIENT-WISE SPLIT (identical to Person 1)
# ============================================================

print("\n" + "=" * 65)
print("STEP 2: PATIENT-WISE TRAIN/TEST SPLIT")
print("=" * 65)

X = df[FEATURES]
y = df[TARGET]
groups = df[PATIENT_ID]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

train_patients = set(groups_train.unique())
test_patients = set(groups_test.unique())

print("Training recordings:", len(X_train), "| patients:", len(train_patients))
print("Testing recordings: ", len(X_test), "| patients:", len(test_patients))
print("Test patient IDs:", sorted(int(p) for p in test_patients))

assert train_patients.isdisjoint(test_patients), "Patient leakage detected!"
print("Patient-wise split verified: no overlap.")


# ============================================================
# 4. BASELINE AND DEFAULT SVR
# ============================================================

print("\n" + "=" * 65)
print("STEP 3: BASELINE AND DEFAULT SVR")
print("=" * 65)

baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
b_mae, b_rmse, b_r2 = evaluate(y_test, baseline.predict(X_test))

default_svr = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf")),
])
default_svr.fit(X_train, y_train)
d_mae, d_rmse, d_r2 = evaluate(y_test, default_svr.predict(X_test))

print(f"Training-mean baseline: MAE={b_mae:.4f}  RMSE={b_rmse:.4f}  R2={b_r2:.4f}")
print(f"Default SVR (RBF)     : MAE={d_mae:.4f}  RMSE={d_rmse:.4f}  R2={d_r2:.4f}")


# ============================================================
# 5. TUNE SVR (patient-grouped cross-validation)
# ============================================================

print("\n" + "=" * 65)
print("STEP 4: HYPERPARAMETER TUNING (may take 1-3 minutes)")
print("=" * 65)

# C: how strictly the model fits the training data (small = smoother)
# epsilon: errors smaller than this are ignored
# gamma: how far one recording's influence reaches (small = smoother)
param_grid = {
    "svr__C": [0.1, 1, 10, 100],
    "svr__epsilon": [0.1, 1.0, 3.0],
    "svr__gamma": ["scale", 0.01, 0.001],
}

grid = GridSearchCV(
    estimator=Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))]),
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=GroupKFold(n_splits=5),
    n_jobs=-1,
    refit=True,
)
grid.fit(X_train, y_train, groups=groups_train)

best_svr = grid.best_estimator_
print("Best parameters:", grid.best_params_)
print("Best grouped CV MAE:", round(-grid.best_score_, 4))


# ============================================================
# 6. EVALUATE TUNED SVR ON UNSEEN PATIENTS
# ============================================================

print("\n" + "=" * 65)
print("STEP 5: TEST-SET RESULTS")
print("=" * 65)

y_pred = best_svr.predict(X_test)
mae, rmse, r2 = evaluate(y_test, y_pred)
print(f"Tuned SVR: MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

comparison = pd.DataFrame([
    {"Model": "Training-mean baseline", "MAE": b_mae, "RMSE": b_rmse, "R2": b_r2},
    *OTHER_RESULTS,
    {"Model": "SVR default (RBF)", "MAE": d_mae, "RMSE": d_rmse, "R2": d_r2},
    {"Model": "SVR tuned (RBF)", "MAE": mae, "RMSE": rmse, "R2": r2},
])
comparison.to_csv(OUTPUT_DIR / "svr_comparison.csv", index=False)

print("\nComparison table:")
print(comparison.round(4).to_string(index=False))

pd.DataFrame({
    "subject#": groups_test.to_numpy(),
    "Actual_total_UPDRS": y_test.to_numpy(),
    "Predicted_total_UPDRS": y_pred,
    "Absolute_Error": np.abs(y_test.to_numpy() - y_pred),
}).to_csv(OUTPUT_DIR / "svr_predictions.csv", index=False)

pd.DataFrame({
    "Model": ["SVR tuned (RBF)"] * 3,
    "Metric": ["MAE", "RMSE", "R2"],
    "Value": [mae, rmse, r2],
}).to_csv(OUTPUT_DIR / "svr_metrics.csv", index=False)


# ============================================================
# 7. FEATURE IMPORTANCE (permutation, since RBF SVR has none built in)
# ============================================================

print("\n" + "=" * 65)
print("STEP 6: PERMUTATION IMPORTANCE (on test patients)")
print("=" * 65)

perm = permutation_importance(
    best_svr, X_test, y_test,
    scoring="neg_mean_absolute_error",
    n_repeats=5, random_state=42, n_jobs=-1,
)
importance = pd.DataFrame({
    "Feature": FEATURES,
    "MAE_increase_when_shuffled": perm.importances_mean,
}).sort_values("MAE_increase_when_shuffled", ascending=False)

print(importance.round(4).to_string(index=False))
importance.to_csv(OUTPUT_DIR / "svr_permutation_importance.csv", index=False)

plot_imp = importance.sort_values("MAE_increase_when_shuffled", ascending=True)
plt.figure(figsize=(10, 7))
plt.barh(plot_imp["Feature"], plot_imp["MAE_increase_when_shuffled"])
plt.axvline(0, color="black", linewidth=1)
plt.xlabel("Increase in MAE when the feature is shuffled")
plt.title("SVR: Permutation Importance (test patients)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "svr_permutation_importance.png", dpi=300)
plt.close()


# ============================================================
# 8. ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.5)
low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())
plt.plot([low, high], [low, high], "--", color="red", label="Perfect prediction")
plt.xlabel("Actual Total UPDRS")
plt.ylabel("Predicted Total UPDRS")
plt.title("SVR (RBF): Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "svr_actual_vs_predicted.png", dpi=300)
plt.close()


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print("Model: Support Vector Regression (RBF kernel), tuned")
print("Best parameters:", grid.best_params_)
print(f"Test MAE:  {mae:.4f} UPDRS points")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R2:   {r2:.4f}")
print("\nAll results saved in:", OUTPUT_DIR)
print("SVR experiment completed successfully.")