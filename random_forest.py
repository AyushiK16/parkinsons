"""
Parkinson's Disease Severity Prediction
Person 2: Random Forest Regressor

Same dataset, features, target and patient split as Person 1.
Random Forest does not need feature scaling, so no scaler is used.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "parkinsons_updrs.data"
OUTPUT_DIR = BASE_DIR / "outputs" / "random_forest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

TARGET = "total_UPDRS"
PATIENT_ID = "subject#"

FEATURES = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "Shimmer:APQ11", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
]

# Person 1's Linear Regression results (same split) for comparison
LINEAR_RESULTS = {"Model": "Linear Regression", "MAE": 7.8890,
                  "RMSE": 9.6853, "R2": -0.1321}


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


# ============================================================
# 2. LOAD AND CHECK DATA
# ============================================================

print("=" * 65)
print("STEP 1: LOAD AND CHECK DATA")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. Put parkinsons_updrs.data "
        "in the same folder as random_forest.py."
    )

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Unique patients:", df[PATIENT_ID].nunique())
print("Missing values:", df.isnull().sum().sum())

required = [PATIENT_ID, TARGET] + FEATURES
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

print(f"\n{TARGET} summary:")
print(df[TARGET].describe().round(3).to_string())


# ============================================================
# 3. EDA (distributions and outliers)
# ============================================================

print("\n" + "=" * 65)
print("STEP 2: EDA")
print("=" * 65)

# Target distribution
plt.figure(figsize=(8, 5))
sns.histplot(df[TARGET], bins=30, kde=True)
plt.title("Distribution of Total UPDRS")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=300)
plt.close()

# Feature distributions (histograms)
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
for ax, feature in zip(axes.flat, FEATURES):
    sns.histplot(df[feature], bins=30, ax=ax)
    ax.set_title(feature, fontsize=9)
    ax.set_xlabel("")
fig.suptitle("Voice Feature Distributions")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=300)
plt.close(fig)

# Boxplots to spot outliers (each feature scaled 0-1 just for plotting)
scaled_for_plot = (df[FEATURES] - df[FEATURES].min()) / (
    df[FEATURES].max() - df[FEATURES].min()
)
plt.figure(figsize=(14, 6))
sns.boxplot(data=scaled_for_plot)
plt.xticks(rotation=90)
plt.title("Boxplots of Voice Features (min-max scaled for plotting only)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_boxplots.png", dpi=300)
plt.close()

print("EDA plots saved in:", OUTPUT_DIR)


# ============================================================
# 4. PATIENT-WISE SPLIT (identical to Person 1)
# ============================================================

print("\n" + "=" * 65)
print("STEP 3: PATIENT-WISE TRAIN/TEST SPLIT")
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
print("Test patient IDs:", sorted(test_patients))

assert train_patients.isdisjoint(test_patients), "Patient leakage detected!"
print("Patient-wise split verified: no overlap.")


# ============================================================
# 5. BASELINE AND DEFAULT RANDOM FOREST
# ============================================================

print("\n" + "=" * 65)
print("STEP 4: BASELINE AND DEFAULT RANDOM FOREST")
print("=" * 65)

baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
b_mae, b_rmse, b_r2 = evaluate(y_test, baseline.predict(X_test))

default_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
default_rf.fit(X_train, y_train)
d_mae, d_rmse, d_r2 = evaluate(y_test, default_rf.predict(X_test))

print(f"Training-mean baseline : MAE={b_mae:.4f}  RMSE={b_rmse:.4f}  R2={b_r2:.4f}")
print(f"Default Random Forest  : MAE={d_mae:.4f}  RMSE={d_rmse:.4f}  R2={d_r2:.4f}")


# ============================================================
# 6. TUNE RANDOM FOREST (patient-grouped cross-validation)
# ============================================================

print("\n" + "=" * 65)
print("STEP 5: HYPERPARAMETER TUNING (this may take 1-3 minutes)")
print("=" * 65)

param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [5, 10, None],
    "min_samples_leaf": [5, 20, 40],
    "max_features": ["sqrt", 0.5],
}

grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=GroupKFold(n_splits=5),
    n_jobs=-1,
    refit=True,
)
grid.fit(X_train, y_train, groups=groups_train)

best_rf = grid.best_estimator_
print("Best parameters:", grid.best_params_)
print("Best grouped CV MAE:", round(-grid.best_score_, 4))


# ============================================================
# 7. EVALUATE TUNED MODEL ON UNSEEN PATIENTS
# ============================================================

print("\n" + "=" * 65)
print("STEP 6: TEST-SET RESULTS")
print("=" * 65)

y_pred = best_rf.predict(X_test)
mae, rmse, r2 = evaluate(y_test, y_pred)

print(f"Tuned Random Forest: MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

comparison = pd.DataFrame([
    {"Model": "Training-mean baseline", "MAE": b_mae, "RMSE": b_rmse, "R2": b_r2},
    LINEAR_RESULTS,
    {"Model": "Random Forest (default)", "MAE": d_mae, "RMSE": d_rmse, "R2": d_r2},
    {"Model": "Random Forest (tuned)", "MAE": mae, "RMSE": rmse, "R2": r2},
])
comparison.to_csv(OUTPUT_DIR / "random_forest_comparison.csv", index=False)

print("\nComparison table:")
print(comparison.round(4).to_string(index=False))

pd.DataFrame({
    "subject#": groups_test.to_numpy(),
    "Actual_total_UPDRS": y_test.to_numpy(),
    "Predicted_total_UPDRS": y_pred,
    "Absolute_Error": np.abs(y_test.to_numpy() - y_pred),
}).to_csv(OUTPUT_DIR / "random_forest_predictions.csv", index=False)

pd.DataFrame({
    "Model": ["Random Forest (tuned)"] * 3,
    "Metric": ["MAE", "RMSE", "R2"],
    "Value": [mae, rmse, r2],
}).to_csv(OUTPUT_DIR / "random_forest_metrics.csv", index=False)


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 65)
print("STEP 7: FEATURE IMPORTANCE")
print("=" * 65)

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": best_rf.feature_importances_,
}).sort_values("Importance", ascending=False)

print(importance.round(4).to_string(index=False))
importance.to_csv(OUTPUT_DIR / "random_forest_feature_importance.csv", index=False)

plot_imp = importance.sort_values("Importance", ascending=True)
plt.figure(figsize=(10, 7))
plt.barh(plot_imp["Feature"], plot_imp["Importance"])
plt.xlabel("Feature Importance")
plt.title("Random Forest: Voice Feature Importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "random_forest_feature_importance.png", dpi=300)
plt.close()


# ============================================================
# 9. ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.5)
low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())
plt.plot([low, high], [low, high], "--", color="red", label="Perfect prediction")
plt.xlabel("Actual Total UPDRS")
plt.ylabel("Predicted Total UPDRS")
plt.title("Random Forest: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "random_forest_actual_vs_predicted.png", dpi=300)
plt.close()


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print("Model: Random Forest Regressor (tuned)")
print("Best parameters:", grid.best_params_)
print(f"Test MAE:  {mae:.4f} UPDRS points")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R2:   {r2:.4f}")
print("Top 3 features:", importance["Feature"].head(3).tolist())
print("\nAll results saved in:", OUTPUT_DIR)
print("Random Forest experiment completed successfully.")