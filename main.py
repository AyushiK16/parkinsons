
"""
Parkinson's Disease Severity Prediction
Person A: Linear Regression

Dataset: UCI Parkinson's Telemonitoring
Target: total_UPDRS
Inputs: 16 biomedical voice measurements

Pipeline:
1. Load and inspect data
2. Perform EDA
3. Split by patient
4. Scale features without test-data leakage
5. Train Linear Regression
6. Evaluate and save results
7. Interpret coefficients
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "parkinsons_updrs.data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

TARGET = "total_UPDRS"
PATIENT_ID = "subject#"

VOICE_FEATURES = [
    "Jitter(%)",
    "Jitter(Abs)",
    "Jitter:RAP",
    "Jitter:PPQ5",
    "Jitter:DDP",
    "Shimmer",
    "Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "Shimmer:APQ11",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "PPE",
]


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n" + "=" * 65)
print("STEP 1: LOADING DATASET")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found at:\n{DATA_PATH}\n"
        "Download parkinsons_updrs.data from UCI and place it "
        "in the same folder as main.py."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Shape:", df.shape)

print("\nFirst five rows:")
print(df.head().to_string(index=False))


# ============================================================
# 3. INSPECT DATA
# ============================================================

print("\n" + "=" * 65)
print("STEP 2: DATA INSPECTION")
print("=" * 65)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes.to_string())

print("\nMissing values per column:")
print(df.isnull().sum().to_string())

print("\nTotal missing values:", df.isnull().sum().sum())
print("Exact duplicate rows:", df.duplicated().sum())
print("Unique patients:", df[PATIENT_ID].nunique())

print("\nSummary statistics:")
print(df.describe().round(3).to_string())


# ============================================================
# 4. VALIDATE THE DATA
# ============================================================

print("\n" + "=" * 65)
print("STEP 3: DATA VALIDATION")
print("=" * 65)

required_columns = [PATIENT_ID, TARGET] + VOICE_FEATURES

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Expected columns missing from dataset: {missing_columns}"
    )

# This project requires numerical voice features and target scores.
if df[required_columns].isnull().any().any():
    raise ValueError(
        "Missing values were found in the required columns. "
        "Inspect the data before training."
    )

if not np.isfinite(
    df[VOICE_FEATURES + [TARGET]].to_numpy(dtype=float)
).all():
    raise ValueError(
        "Non-finite numerical values were found in the features or target."
    )

if df[PATIENT_ID].nunique() < 2:
    raise ValueError("At least two different patients are required.")

print("Required columns are present.")
print("No missing or non-finite values in model columns.")
print("Ready for EDA and model preparation.")


# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("STEP 4: EXPLORATORY DATA ANALYSIS")
print("=" * 65)

print(f"\nTarget: {TARGET}")
print(f"Mean:   {df[TARGET].mean():.3f}")
print(f"Median: {df[TARGET].median():.3f}")
print(f"Std:    {df[TARGET].std():.3f}")
print(f"Min:    {df[TARGET].min():.3f}")
print(f"Max:    {df[TARGET].max():.3f}")

# 5A. Target distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x=TARGET,
    bins=30,
    kde=True,
)

plt.title("Distribution of Total UPDRS Scores")
plt.xlabel("Total UPDRS")
plt.ylabel("Number of Recordings")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "target_distribution.png",
    dpi=300,
)
plt.close()


# 5B. Correlation heatmap
correlation_data = df[VOICE_FEATURES + [TARGET]]
correlation_matrix = correlation_data.corr()

plt.figure(figsize=(15, 12))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
)

plt.title("Voice Features and Total UPDRS: Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=300,
)
plt.close()

# Rank features by absolute Pearson correlation with the target.
target_correlations = (
    correlation_matrix[TARGET]
    .drop(TARGET)
    .sort_values(key=lambda values: values.abs(), ascending=False)
)

print("\nVoice features ranked by absolute correlation with target:")
print(target_correlations.round(4).to_string())

target_correlations.rename("Pearson_Correlation").to_csv(
    OUTPUT_DIR / "feature_correlations.csv"
)

# 5C. Plot the five features with the strongest absolute correlations.
top_correlated_features = target_correlations.head(5).index.tolist()

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(15, 9),
)

for ax, feature in zip(axes.flat, top_correlated_features):
    sns.regplot(
        data=df,
        x=feature,
        y=TARGET,
        ax=ax,
        scatter_kws={"alpha": 0.25, "s": 15},
        line_kws={"color": "red"},
        ci=None,
    )
    ax.set_title(f"{feature} vs {TARGET}")

# Hide the unused sixth subplot.
for ax in axes.flat[len(top_correlated_features):]:
    ax.set_visible(False)

fig.suptitle(
    "Top Five Voice Features vs Total UPDRS",
    fontsize=14,
)
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "top_correlated_features.png",
    dpi=300,
)
plt.close(fig)

print("\nEDA plots saved in:", OUTPUT_DIR)


# ============================================================
# 6. DEFINE X, y AND PATIENT GROUPS
# ============================================================

print("\n" + "=" * 65)
print("STEP 5: FEATURE AND TARGET SELECTION")
print("=" * 65)

X = df[VOICE_FEATURES].copy()
y = df[TARGET].copy()
groups = df[PATIENT_ID].copy()

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of patient groups:", groups.nunique())

print("\nInput features:")
print(VOICE_FEATURES)

print("\nTarget:", TARGET)


# ============================================================
# 7. PATIENT-WISE TRAIN/TEST SPLIT
# ============================================================

print("\n" + "=" * 65)
print("STEP 6: PATIENT-WISE TRAIN/TEST SPLIT")
print("=" * 65)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

train_patients = groups.iloc[train_indices]
test_patients = groups.iloc[test_indices]

train_patient_ids = sorted(train_patients.unique().tolist())
test_patient_ids = sorted(test_patients.unique().tolist())

overlap = set(train_patient_ids) & set(test_patient_ids)

print("Training recordings:", len(X_train))
print("Testing recordings:", len(X_test))

print("Training patients:", len(train_patient_ids))
print("Testing patients:", len(test_patient_ids))

print("\nTraining patient IDs:")
print(train_patient_ids)

print("\nTesting patient IDs:")
print(test_patient_ids)

print("\nPatient overlap:", overlap)
print("Patient-wise split successful:", len(overlap) == 0)

# Save the split details so teammates can use the same patients.

split_details = pd.DataFrame({
    "subject#": groups.to_numpy(),
    "split": np.where(
        np.isin(np.arange(len(df)), train_indices),
        "train",
        "test",
    ),
})


split_details.to_csv(
    OUTPUT_DIR / "patient_split_by_recording.csv",
    index=False,
)

pd.DataFrame({
    "split": ["train"] * len(train_patient_ids)
    + ["test"] * len(test_patient_ids),
    "subject#": train_patient_ids + test_patient_ids,
}).to_csv(
    OUTPUT_DIR / "patient_split_ids.csv",
    index=False,
)

if overlap:
    raise RuntimeError("Patient leakage detected in the split!")


# ============================================================
# 8. SCALE FEATURES
# ============================================================

print("\n" + "=" * 65)
print("STEP 7: STANDARD SCALING")
print("=" * 65)

scaler = StandardScaler()

# Fit ONLY on the training set.
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test set using the training-set statistics.
X_test_scaled = scaler.transform(X_test)

print("Training scaled shape:", X_train_scaled.shape)
print("Testing scaled shape:", X_test_scaled.shape)

print(
    "Mean of first five scaled training features:",
    np.round(X_train_scaled.mean(axis=0)[:5], 4),
)

print(
    "Std of first five scaled training features:",
    np.round(X_train_scaled.std(axis=0)[:5], 4),
)


# ============================================================
# 9. TRAIN LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 65)
print("STEP 8: TRAINING LINEAR REGRESSION")
print("=" * 65)

model = LinearRegression()

model.fit(X_train_scaled, y_train)

print("Model trained successfully.")
print("Intercept:", round(model.intercept_, 4))
print("Number of coefficients:", len(model.coef_))


# ============================================================
# 10. EVALUATE ON UNSEEN PATIENTS
# ============================================================

print("\n" + "=" * 65)
print("STEP 9: TEST-SET EVALUATION")
print("=" * 65)

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

metrics_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2"],
    "Value": [mae, rmse, r2],
})

metrics_df.to_csv(
    OUTPUT_DIR / "linear_regression_metrics.csv",
    index=False,
)

predictions_df = pd.DataFrame({
    "Patient_ID": groups.iloc[test_indices].to_numpy(),
    "Actual_total_UPDRS": y_test.to_numpy(),
    "Predicted_total_UPDRS": y_pred,
    "Prediction_Error": y_test.to_numpy() - y_pred,
})

predictions_df.to_csv(
    OUTPUT_DIR / "linear_regression_predictions.csv",
    index=False,
)


# ============================================================
# 11. COEFFICIENT ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("STEP 10: COEFFICIENT ANALYSIS")
print("=" * 65)

coefficients_df = pd.DataFrame({
    "Feature": VOICE_FEATURES,
    "Coefficient": model.coef_,
})

coefficients_df["Absolute_Coefficient"] = (
    coefficients_df["Coefficient"].abs()
)

coefficients_df = coefficients_df.sort_values(
    "Absolute_Coefficient",
    ascending=False,
)

print("\nCoefficients ranked by absolute magnitude:")
print(coefficients_df.to_string(
    index=False,
    float_format=lambda value: f"{value:.4f}",
))

coefficients_df.to_csv(
    OUTPUT_DIR / "linear_regression_coefficients.csv",
    index=False,
)

# Plot signed coefficients, sorted from negative to positive.
plot_coefficients = coefficients_df.sort_values("Coefficient")

plt.figure(figsize=(11, 8))

plt.barh(
    plot_coefficients["Feature"],
    plot_coefficients["Coefficient"],
)

plt.axvline(0, color="black", linewidth=1)
plt.title("Linear Regression Coefficients")
plt.xlabel("Coefficient (standardized features)")
plt.ylabel("Voice Feature")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "linear_regression_coefficients.png",
    dpi=300,
)
plt.close()


# ============================================================
# 12. ACTUAL VS PREDICTED PLOT
# ============================================================

print("\n" + "=" * 65)
print("STEP 11: PREDICTION VISUALIZATION")
print("=" * 65)

plt.figure(figsize=(7, 7))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5,
)

lower = min(float(y_test.min()), float(y_pred.min()))
upper = max(float(y_test.max()), float(y_pred.max()))

plt.plot(
    [lower, upper],
    [lower, upper],
    linestyle="--",
    color="red",
    label="Perfect prediction",
)

plt.xlabel("Actual Total UPDRS")
plt.ylabel("Predicted Total UPDRS")
plt.title("Actual vs Predicted: Linear Regression")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=300,
)
plt.close()


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FINAL PROJECT SUMMARY")
print("=" * 65)

print("Model: Linear Regression")
print("Target:", TARGET)
print("Input features:", len(VOICE_FEATURES))
print("Train/test strategy: patient-wise grouped split")
print("Training recordings:", len(X_train))
print("Testing recordings:", len(X_test))
print("Training patients:", len(train_patient_ids))
print("Testing patients:", len(test_patient_ids))

print(f"\nMAE:  {mae:.4f} UPDRS points")
print(f"RMSE: {rmse:.4f} UPDRS points")
print(f"R2:   {r2:.4f}")

print(
    f"\nAverage absolute prediction error: {mae:.3f} "
    "UPDRS points on the held-out recordings."
)

if r2 > 0:
    print(
        f"R2 indicates {100 * r2:.1f}% of test-target variation "
        "explained relative to the test-mean baseline."
    )
else:
    print(
        "The model did not outperform the test-mean baseline "
        "according to R2."
    )

print("\nTop five coefficients by absolute magnitude:")
for _, row in coefficients_df.head(5).iterrows():
    print(f"  {row['Feature']}: {row['Coefficient']:.4f}")

print("\nAll plots and CSV results saved to:")
print(OUTPUT_DIR)

print("\nProject completed successfully.")
