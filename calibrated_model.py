"""
Person 2: Calibrated monitoring model.
Each patient has a calibration period (first 20% of recordings by test_time).
Predict the CHANGE in total_UPDRS since calibration from the CHANGE in voice.
Strict patient-wise GroupKFold: test patients never appear in training.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "parkinsons_updrs.data"
OUTPUT_DIR = BASE_DIR / "outputs" / "calibrated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

TARGET, PID, TIME = "total_UPDRS", "subject#", "test_time"
CAL_FRACTION = 0.20
N_SPLITS = 7

FEATURES = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "Shimmer:APQ11", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE",
]

# ---------------- load ----------------
print("=" * 65)
print("STEP 1: LOAD DATA AND BUILD CALIBRATION / EVALUATION ROWS")
print("=" * 65)
df = pd.read_csv(DATA_PATH)
df = df.sort_values([PID, TIME], kind="mergesort").reset_index(drop=True)

df["rank"] = df.groupby(PID).cumcount()
df["n"] = df.groupby(PID)[PID].transform("size")
df["is_cal"] = df["rank"] < np.ceil(df["n"] * CAL_FRACTION)

cal = df[df["is_cal"]]
post = df[~df["is_cal"]].copy()

anchor_score = cal.groupby(PID)[TARGET].last()
anchor_time = cal.groupby(PID)[TIME].max()
voice_base = cal.groupby(PID)[FEATURES].mean()

pids = post[PID]
post["anchor"] = pids.map(anchor_score)
post["dt"] = post[TIME] - pids.map(anchor_time)
voice_delta = pd.DataFrame(
    post[FEATURES].to_numpy() - voice_base.loc[pids].to_numpy(),
    columns=FEATURES, index=post.index,
)
post["delta"] = post[TARGET] - post["anchor"]

print("Calibration rows:", len(cal), "| evaluation rows:", len(post))
print("Patients:", post[PID].nunique())
print("Anchor uses ONLY each patient's own early rows (no future info).")

y_delta = post["delta"].to_numpy()
y_total = post[TARGET].to_numpy()
anchor = post["anchor"].to_numpy()
groups = post[PID].to_numpy()

# ---------------- feature sets ----------------
sets = {
    "dt": post[["dt"]],
    "voice_delta": voice_delta,
    "voice_delta+dt": pd.concat([voice_delta, post[["dt"]]], axis=1),
}

rf = dict(n_estimators=300, max_depth=5, min_samples_leaf=20,
          max_features="sqrt", random_state=42, n_jobs=-1)

experiments = {
    "Anchor only (predict no change)": (None, "dt"),
    "Mean change (train average)": (DummyRegressor(strategy="mean"), "dt"),
    "Time trend only (Linear on dt)": (LinearRegression(), "dt"),
    "Ridge on voice change": (Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=100))]), "voice_delta"),
    "RF on voice change": (RandomForestRegressor(**rf), "voice_delta"),
    "Ridge on voice change + time": (Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=100))]), "voice_delta+dt"),
    "RF on voice change + time": (RandomForestRegressor(**rf), "voice_delta+dt"),
}

# ---------------- cross-validation ----------------
print("\n" + "=" * 65)
print(f"STEP 2: {N_SPLITS}-FOLD PATIENT-GROUPED CROSS-VALIDATION")
print("=" * 65)
folds = list(GroupKFold(n_splits=N_SPLITS).split(post, y_delta, groups))
for i, (tr, te) in enumerate(folds, 1):
    assert set(groups[tr]).isdisjoint(set(groups[te])), "Patient leakage!"
print("All folds verified: no patient in both train and test.")

oof = {name: np.zeros(len(post)) for name in experiments}
for tr, te in folds:
    for name, (model, fs) in experiments.items():
        if model is None:
            oof[name][te] = 0.0
        else:
            X = sets[fs].to_numpy()
            m = clone(model).fit(X[tr], y_delta[tr])
            oof[name][te] = m.predict(X[te])

# ---------------- results ----------------
print("\n" + "=" * 65)
print("STEP 3: RESULTS (all patients, each tested once)")
print("=" * 65)
rows = []
for name in experiments:
    d_pred = oof[name]
    t_pred = anchor + d_pred
    rows.append({
        "Model": name,
        "MAE (total score)": mean_absolute_error(y_total, t_pred),
        "RMSE (total score)": np.sqrt(mean_squared_error(y_total, t_pred)),
        "R2 (total score)": r2_score(y_total, t_pred),
        "MAE (change)": mean_absolute_error(y_delta, d_pred),
        "R2 (change)": r2_score(y_delta, d_pred),
    })
res = pd.DataFrame(rows)
res.to_csv(OUTPUT_DIR / "calibrated_results.csv", index=False)
print(res.round(4).to_string(index=False))

pred_df = pd.DataFrame({"subject#": groups, "Actual": y_total, "Anchor": anchor})
for name in experiments:
    pred_df[name] = anchor + oof[name]
pred_df.to_csv(OUTPUT_DIR / "calibrated_predictions.csv", index=False)

# ---------------- verdict ----------------
trend = res[res["Model"] == "Time trend only (Linear on dt)"].iloc[0]
voice_rows = res[res["Model"].str.contains("voice")]
best_voice = voice_rows.sort_values("MAE (change)").iloc[0]

print("\n" + "=" * 65)
print("VERDICT")
print("=" * 65)
print(f"Best voice-based model: {best_voice['Model']}")
print(f"  R2 (change) = {best_voice['R2 (change)']:.4f} | MAE (change) = {best_voice['MAE (change)']:.4f}")
print(f"Time-trend-only baseline: R2 (change) = {trend['R2 (change)']:.4f} | MAE (change) = {trend['MAE (change)']:.4f}")
if best_voice["MAE (change)"] < trend["MAE (change)"] - 0.05:
    print("Voice adds information beyond a simple time trend.")
else:
    print("Voice does NOT clearly add information beyond anchor + time trend.")

# ---------------- plot ----------------
plt.figure(figsize=(10, 6))
plot_df = res.sort_values("MAE (total score)")
plt.barh(plot_df["Model"], plot_df["MAE (total score)"])
plt.xlabel("MAE on total_UPDRS (lower is better)")
plt.title("Calibrated monitoring: model comparison")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "calibrated_mae_comparison.png", dpi=300)
plt.close()

print("\nAll results saved in:", OUTPUT_DIR)
print("Calibrated experiment completed successfully.")