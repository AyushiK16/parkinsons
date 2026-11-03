"""
Parkinson's Disease Severity Prediction
Person 2: Random Forest, TIME-BASED split within each patient

Question answered: "If we already know a patient's earlier recordings,
how well can we predict their LATER severity from voice?"

Split rule (so teammates can copy it exactly):
  For every patient, sort recordings by test_time.
  First floor(70%) of recordings -> train, remaining -> test.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "parkinsons_updrs.data"
OUTPUT_DIR = BASE_DIR / "outputs" / "random_forest_timesplit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

TARGET = "total_UPDRS"
PATIENT_ID = "subject#"
TIME_COL = "test_time"
TRAIN_FRACTION = 0.70

VOICE_FEATURES = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "Shimmer:APQ11", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
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
print("Shape:", df.shape)
print("Unique patients:", df[PATIENT_ID].nunique())
print("Missing values:", df.isnull().sum().sum())

required = [PATIENT_ID, TARGET, TIME_COL] + VOICE_FEATURES
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")


# ============================================================
# 3. TIME-BASED SPLIT WITHIN EACH PATIENT
# ============================================================

print("\n" + "=" * 65)
print("STEP 2: TIME-BASED SPLIT (first 70% train, last 30% test, per patient)")
print("=" * 65)

df = df.sort_values([PATIENT_ID, TIME_COL], kind="mergesort").reset_index(drop=True)

df["rank_in_patient"] = df.groupby(PATIENT_ID).cumcount()
df["n_in_patient"] = df.groupby(PATIENT_ID)[PATIENT_ID].transform("size")
df["split"] = np.where(
    df["rank_in_patient"] < np.floor(df["n_in_patient"] * TRAIN_FRACTION),
    "train",
    "test",
)

train = df[df["split"] == "train"].copy()
test = df[df["split"] == "test"].copy()

print("Training recordings:", len(train), "| patients:", train[PATIENT_ID].nunique())
print("Testing recordings: ", len(test), "| patients:", test[PATIENT_ID].nunique())

# Safety checks
assert train[PATIENT_ID].nunique() == test[PATIENT_ID].nunique() == 42, \
    "Every patient should appear in both train and test."
latest_train_time = train.groupby(PATIENT_ID)[TIME_COL].max()
earliest_test_time = test.groupby(PATIENT_ID)[TIME_COL].min()
assert (earliest_test_time >= latest_train_time).all(), \
    "Some test recordings come BEFORE training recordings!"
print("Check passed: for every patient, all test recordings come AFTER training ones.")

df[[PATIENT_ID, TIME_COL, "split"]].to_csv(
    OUTPUT_DIR / "timesplit_rows.csv", index=False
)

y_train, y_test = train[TARGET], test[TARGET]


# ============================================================
# 4. BASELINES (what a "dumb" approach scores)
# ============================================================

print("\n" + "=" * 65)
print("STEP 3: BASELINES")
print("=" * 65)

results = []

# Baseline A: one global average for everybody
pred_global = np.full(len(test), y_train.mean())
results.append(("Baseline: global train mean", *evaluate(y_test, pred_global)))

# Baseline B: each patient's own average training score
patient_mean = train.groupby(PATIENT_ID)[TARGET].mean()
pred_patient_mean = test[PATIENT_ID].map(patient_mean).to_numpy()
results.append(("Baseline: patient's own train mean", *evaluate(y_test, pred_patient_mean)))

# Baseline C: each patient's LAST known training score
patient_last = train.groupby(PATIENT_ID)[TARGET].last()
pred_patient_last = test[PATIENT_ID].map(patient_last).to_numpy()
results.append(("Baseline: patient's last known score", *evaluate(y_test, pred_patient_last)))

for name, mae, rmse, r2 in results:
    print(f"{name:40s} MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")


# ============================================================
# 5. RANDOM FOREST MODELS
# ============================================================

print("\n" + "=" * 65)
print("STEP 4: RANDOM FOREST (no heavy tuning: fixed sensible settings)")
print("=" * 65)

rf_params = dict(
    n_estimators=300,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)

# Model 1: voice features only
rf_voice = RandomForestRegressor(**rf_params)
rf_voice.fit(train[VOICE_FEATURES], y_train)
pred_voice = rf_voice.predict(test[VOICE_FEATURES])
results.append(("RF: 16 voice features", *evaluate(y_test, pred_voice)))

# Model 2: voice features + test_time (how far into the study)
feats_time = VOICE_FEATURES + [TIME_COL]
rf_time = RandomForestRegressor(**rf_params)
rf_time.fit(train[feats_time], y_train)
pred_time = rf_time.predict(test[feats_time])
results.append(("RF: 16 voice + test_time", *evaluate(y_test, pred_time)))

comparison = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])
comparison.to_csv(OUTPUT_DIR / "timesplit_comparison.csv", index=False)

print("\nComparison table:")
print(comparison.round(4).to_string(index=False))


# ============================================================
# 6. PER-PATIENT ERRORS AND PREDICTIONS
# ============================================================

pred_df = pd.DataFrame({
    "subject#": test[PATIENT_ID].to_numpy(),
    "test_time": test[TIME_COL].to_numpy(),
    "Actual_total_UPDRS": y_test.to_numpy(),
    "RF_voice_pred": pred_voice,
    "RF_voice_time_pred": pred_time,
    "Patient_mean_pred": pred_patient_mean,
})
pred_df["RF_voice_abs_error"] = np.abs(pred_df["Actual_total_UPDRS"] - pred_df["RF_voice_pred"])
pred_df.to_csv(OUTPUT_DIR / "timesplit_predictions.csv", index=False)

per_patient = (
    pred_df.groupby("subject#")["RF_voice_abs_error"].mean().round(3)
    .sort_values(ascending=False)
)
per_patient.to_csv(OUTPUT_DIR / "timesplit_per_patient_mae.csv")
print("\nHardest 5 patients (highest MAE, RF voice-only):")
print(per_patient.head(5).to_string())


# ============================================================
# 7. FEATURE IMPORTANCE (voice-only model)
# ============================================================

importance = pd.DataFrame({
    "Feature": VOICE_FEATURES,
    "Importance": rf_voice.feature_importances_,
}).sort_values("Importance", ascending=False)
importance.to_csv(OUTPUT_DIR / "timesplit_feature_importance.csv", index=False)

print("\nTop 5 voice features:")
print(importance.head(5).round(4).to_string(index=False))

plot_imp = importance.sort_values("Importance", ascending=True)
plt.figure(figsize=(10, 7))
plt.barh(plot_imp["Feature"], plot_imp["Importance"])
plt.xlabel("Feature Importance")
plt.title("Random Forest (time split): Voice Feature Importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "timesplit_feature_importance.png", dpi=300)
plt.close()


# ============================================================
# 8. PLOTS
# ============================================================

# Actual vs predicted
plt.figure(figsize=(7, 7))
plt.scatter(y_test, pred_voice, alpha=0.4)
low = min(y_test.min(), pred_voice.min())
high = max(y_test.max(), pred_voice.max())
plt.plot([low, high], [low, high], "--", color="red", label="Perfect prediction")
plt.xlabel("Actual Total UPDRS")
plt.ylabel("Predicted Total UPDRS")
plt.title("RF (voice only), time split: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "timesplit_actual_vs_predicted.png", dpi=300)
plt.close()

# Four example patients over time
example_ids = sorted(df[PATIENT_ID].unique())[::10][:4]
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, pid in zip(axes.flat, example_ids):
    tr = train[train[PATIENT_ID] == pid]
    te = pred_df[pred_df["subject#"] == pid]
    ax.scatter(tr[TIME_COL], tr[TARGET], s=10, alpha=0.5, label="Train (actual)")
    ax.scatter(te["test_time"], te["Actual_total_UPDRS"], s=10, label="Test (actual)")
    ax.scatter(te["test_time"], te["RF_voice_pred"], s=10, label="Test (RF predicted)")
    ax.set_title(f"Patient {pid}")
    ax.set_xlabel("test_time (days)")
    ax.set_ylabel("total_UPDRS")
axes.flat[0].legend(fontsize=8)
fig.suptitle("Examples: training period, then predicted vs actual later scores")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "timesplit_example_patients.png", dpi=300)
plt.close(fig)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

best_rf = comparison[comparison["Model"].str.startswith("RF")].sort_values("MAE").iloc[0]
pm = comparison[comparison["Model"] == "Baseline: patient's own train mean"].iloc[0]

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print(f"Best RF this run: {best_rf['Model']}")
print(f"  MAE={best_rf['MAE']:.4f}  RMSE={best_rf['RMSE']:.4f}  R2={best_rf['R2']:.4f}")
print(f"Patient-mean baseline: MAE={pm['MAE']:.4f}  R2={pm['R2']:.4f}")
if best_rf["MAE"] < pm["MAE"]:
    print("RF beats the patient-mean baseline: voice adds information within a patient.")
else:
    print("RF does NOT beat the patient-mean baseline: it mostly recognises WHICH patient it is.")
print("\nAll results saved in:", OUTPUT_DIR)
print("Time-split experiment completed successfully.")