"""
Parkinson's Disease Severity Prediction
Person 2: GroupKFold cross-validation across ALL 42 patients

Every patient is a test patient exactly once.
Models: training-mean baseline, Linear Regression, Random Forest, SVR.
Same 16 voice features and target as everyone else.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "parkinsons_updrs.data"
OUTPUT_DIR = BASE_DIR / "outputs" / "groupkfold"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

TARGET = "total_UPDRS"
PATIENT_ID = "subject#"
N_SPLITS = 7  # 42 patients / 7 = 6 test patients per fold

FEATURES = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "Shimmer:APQ11", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
]

# NOTE: the RF and SVR settings below were chosen earlier using the 33
# training patients of the single split. Those patients now also appear in
# test folds, so results may be slightly optimistic. Mention this in the report.
MODELS = {
    "Training-mean baseline": DummyRegressor(strategy="mean"),
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression()),
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=5, min_samples_leaf=5,
        max_features="sqrt", random_state=42, n_jobs=-1,
    ),
    "SVR (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=0.1, epsilon=3.0, gamma=0.001)),
    ]),
}


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

X = df[FEATURES]
y = df[TARGET]
groups = df[PATIENT_ID]


# ============================================================
# 3. BUILD FOLDS
# ============================================================

print("\n" + "=" * 65)
print(f"STEP 2: {N_SPLITS}-FOLD GROUPKFOLD (patients never shared)")
print("=" * 65)

gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(X, y, groups=groups))

fold_rows = []
for fold_no, (tr, te) in enumerate(folds, start=1):
    train_p = set(groups.iloc[tr].unique())
    test_p = set(groups.iloc[te].unique())
    assert train_p.isdisjoint(test_p), "Patient leakage detected!"
    print(f"Fold {fold_no}: train patients={len(train_p)}, "
          f"test patients={len(test_p)} -> {sorted(int(p) for p in test_p)}")
    for p in sorted(test_p):
        fold_rows.append({"fold": fold_no, "subject#": int(p)})

# Save so teammates can use exactly the same folds
pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "groupkfold_patient_folds.csv", index=False)
print("Every patient appears in exactly one test fold: verified.")


# ============================================================
# 4. RUN ALL MODELS ON ALL FOLDS
# ============================================================

print("\n" + "=" * 65)
print("STEP 3: TRAINING AND TESTING (about 1 minute)")
print("=" * 65)

fold_metrics = []
oof_preds = {name: np.zeros(len(df)) for name in MODELS}

for fold_no, (tr, te) in enumerate(folds, start=1):
    X_tr, X_te = X.iloc[tr], X.iloc[te]
    y_tr, y_te = y.iloc[tr], y.iloc[te]
    for name, model in MODELS.items():
        m = clone(model).fit(X_tr, y_tr)
        pred = m.predict(X_te)
        oof_preds[name][te] = pred
        mae, rmse, r2 = evaluate(y_te, pred)
        fold_metrics.append({"fold": fold_no, "Model": name,
                             "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"Fold {fold_no} done.")

fold_df = pd.DataFrame(fold_metrics)
fold_df.to_csv(OUTPUT_DIR / "groupkfold_fold_metrics.csv", index=False)


# ============================================================
# 5. SUMMARY TABLES
# ============================================================

print("\n" + "=" * 65)
print("STEP 4: RESULTS")
print("=" * 65)

summary_rows = []
for name in MODELS:
    f = fold_df[fold_df["Model"] == name]
    p_mae, p_rmse, p_r2 = evaluate(y, oof_preds[name])
    summary_rows.append({
        "Model": name,
        "Mean fold MAE": f["MAE"].mean(),
        "Std fold MAE": f["MAE"].std(),
        "Mean fold RMSE": f["RMSE"].mean(),
        "Mean fold R2": f["R2"].mean(),
        "Pooled MAE": p_mae,
        "Pooled RMSE": p_rmse,
        "Pooled R2": p_r2,
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_DIR / "groupkfold_summary.csv", index=False)

print("\nSummary (all 42 patients, each tested once):")
print(summary.round(4).to_string(index=False))

print("\nMAE in each fold:")
print(fold_df.pivot(index="fold", columns="Model", values="MAE")
      .round(3).to_string())

# Per-patient MAE
pred_df = pd.DataFrame({"subject#": groups.to_numpy(), "Actual": y.to_numpy()})
for name in MODELS:
    pred_df[f"{name}_pred"] = oof_preds[name]
pred_df.to_csv(OUTPUT_DIR / "groupkfold_predictions.csv", index=False)

per_patient = pd.DataFrame({
    name: (pred_df["Actual"] - pred_df[f"{name}_pred"]).abs()
    .groupby(pred_df["subject#"]).mean()
    for name in MODELS
}).round(3)
per_patient.to_csv(OUTPUT_DIR / "groupkfold_per_patient_mae.csv")

rf_beats_base = (per_patient["Random Forest"] < per_patient["Training-mean baseline"]).sum()
print(f"\nRandom Forest beats the baseline for {rf_beats_base} of "
      f"{per_patient.shape[0]} patients (lower per-patient MAE).")


# ============================================================
# 6. PLOTS
# ============================================================

# MAE per fold, one box per model
plt.figure(figsize=(9, 6))
sns.boxplot(data=fold_df, x="Model", y="MAE")
sns.stripplot(data=fold_df, x="Model", y="MAE", color="black", alpha=0.6)
plt.xticks(rotation=20)
plt.title(f"MAE across {N_SPLITS} patient-grouped folds")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "groupkfold_mae_boxplot.png", dpi=300)
plt.close()

# Per-patient MAE: Random Forest vs baseline
order = per_patient["Training-mean baseline"].sort_values().index
x_pos = np.arange(len(order))
plt.figure(figsize=(14, 6))
plt.bar(x_pos - 0.2, per_patient.loc[order, "Training-mean baseline"], 0.4,
        label="Training-mean baseline")
plt.bar(x_pos + 0.2, per_patient.loc[order, "Random Forest"], 0.4,
        label="Random Forest")
plt.xticks(x_pos, order, rotation=90)
plt.xlabel("Patient ID")
plt.ylabel("MAE (UPDRS points)")
plt.title("Per-patient MAE: Random Forest vs baseline")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "groupkfold_per_patient_mae.png", dpi=300)
plt.close()

# Pooled actual vs predicted for Random Forest
plt.figure(figsize=(7, 7))
plt.scatter(y, oof_preds["Random Forest"], alpha=0.3)
low, high = y.min(), y.max()
plt.plot([low, high], [low, high], "--", color="red", label="Perfect prediction")
plt.xlabel("Actual Total UPDRS")
plt.ylabel("Predicted Total UPDRS")
plt.title("Random Forest, all 42 patients (out-of-fold)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "groupkfold_rf_actual_vs_predicted.png", dpi=300)
plt.close()


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

best = summary.sort_values("Pooled MAE").iloc[0]
print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print(f"Lowest pooled MAE: {best['Model']} ({best['Pooled MAE']:.4f})")
print("Models with pooled R2 > 0:",
      summary.loc[summary["Pooled R2"] > 0, "Model"].tolist() or "none")
print("\nAll results saved in:", OUTPUT_DIR)
print("GroupKFold experiment completed successfully.")