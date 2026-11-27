
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# STEP 1: CONFIGURATION
# ============================================================

DATA_FILE = "parkinsons_updrs.data"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use the same 16 voice features as the Linear Regression model.
FEATURES = [
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

TARGET = "total_UPDRS"
GROUP = "subject#"


# ============================================================
# STEP 2: LOAD AND VALIDATE DATA
# ============================================================

print("=" * 65)
print("DECISION TREE REGRESSION: PARKINSON'S SEVERITY PREDICTION")
print("=" * 65)

df = pd.read_csv(DATA_FILE)

required_columns = FEATURES + [TARGET, GROUP]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if df[required_columns].isnull().any().any():
    raise ValueError("Missing values found in required columns.")

if not np.isfinite(df[required_columns].to_numpy()).all():
    raise ValueError("Non-finite values found in required columns.")

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)
print("Number of patients:", df[GROUP].nunique())
print("Number of voice features:", len(FEATURES))


# ============================================================
# STEP 3: SELECT FEATURES, TARGET AND PATIENT GROUPS
# ============================================================

X = df[FEATURES]
y = df[TARGET]
groups = df[GROUP]

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# STEP 4: PATIENT-WISE TRAIN/TEST SPLIT
# ============================================================

# Same split settings as the Linear Regression experiment.
# Recordings from a patient cannot appear in both sets.

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

groups_train = groups.iloc[train_indices]
groups_test = groups.iloc[test_indices]

train_patients = set(groups_train.unique())
test_patients = set(groups_test.unique())

print("\n" + "=" * 65)
print("PATIENT-WISE TRAIN/TEST SPLIT")
print("=" * 65)

print("Training recordings:", len(X_train))
print("Testing recordings:", len(X_test))
print("Training patients:", len(train_patients))
print("Testing patients:", len(test_patients))
print("Patient overlap:", train_patients.intersection(test_patients))

assert train_patients.isdisjoint(test_patients), (
    "ERROR: A patient appears in both training and testing."
)

print("Patient-wise split verified.")


# ============================================================
# STEP 5: HYPERPARAMETER TUNING
# ============================================================

# Decision trees can overfit if they grow too deep.
# We use grouped cross-validation to select tree complexity.
# The validation folds also keep each patient within one fold.

tree = DecisionTreeRegressor(random_state=42)

param_grid = {
    "max_depth": [3, 5, 8, 12, None],
    "min_samples_leaf": [5, 10, 20, 40],
    "min_samples_split": [2, 10, 20],
}

cv = GroupKFold(n_splits=5)

grid_search = GridSearchCV(
    estimator=tree,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    refit=True,
)

print("\n" + "=" * 65)
print("HYPERPARAMETER TUNING")
print("=" * 65)

print("Testing different tree depths and minimum leaf sizes...")
print("Using 5-fold patient-grouped cross-validation.")

grid_search.fit(
    X_train,
    y_train,
    groups=groups_train,
)

best_model = grid_search.best_estimator_

print("\nBest parameters:")
print(grid_search.best_params_)

print(
    "Best grouped CV MAE:",
    round(-grid_search.best_score_, 4),
)


# ============================================================
# STEP 6: EVALUATE ON UNSEEN TEST PATIENTS
# ============================================================

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Baseline: always predict the mean target value from training data.
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_pred)
)
baseline_r2 = r2_score(y_test, baseline_pred)

print("\n" + "=" * 65)
print("TEST-SET RESULTS")
print("=" * 65)

print("\nDecision Tree Regressor")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

print("\nTraining-mean baseline")
print(f"MAE:  {baseline_mae:.4f}")
print(f"RMSE: {baseline_rmse:.4f}")
print(f"R2:   {baseline_r2:.4f}")


# ============================================================
# STEP 7: COMPARE WITH YOUR EXISTING LINEAR REGRESSION
# ============================================================

# These are the metrics from your earlier Linear Regression run.
# This comparison assumes the same dataset and patient split.

linear_results = {
    "Model": "Linear Regression",
    "MAE": 7.8890,
    "RMSE": 9.6853,
    "R2": -0.1321,
}

decision_tree_results = {
    "Model": "Decision Tree",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
}

comparison = pd.DataFrame(
    [linear_results, decision_tree_results]
)

comparison.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False,
)

print("\n" + "=" * 65)
print("LINEAR REGRESSION VS DECISION TREE")
print("=" * 65)

print(comparison.to_string(index=False))

if mae < linear_results["MAE"]:
    print("\nDecision Tree has lower MAE than Linear Regression.")
else:
    print("\nDecision Tree does not have lower MAE than Linear Regression.")

if r2 > linear_results["R2"]:
    print("Decision Tree has higher R2 than Linear Regression.")
else:
    print("Decision Tree does not have higher R2 than Linear Regression.")

if r2 > baseline_r2:
    print("Decision Tree beats the training-mean baseline on R2.")
else:
    print("Decision Tree does not beat the training-mean baseline on R2.")


# ============================================================
# STEP 8: SAVE PREDICTIONS
# ============================================================

predictions = pd.DataFrame({
    "subject#": groups_test.to_numpy(),
    "Actual_total_UPDRS": y_test.to_numpy(),
    "Predicted_total_UPDRS": y_pred,
})

predictions["Absolute_Error"] = np.abs(
    predictions["Actual_total_UPDRS"]
    - predictions["Predicted_total_UPDRS"]
)

predictions.to_csv(
    os.path.join(OUTPUT_DIR, "decision_tree_predictions.csv"),
    index=False,
)

# Save evaluation metrics.
metrics = pd.DataFrame([
    {"Model": "Decision Tree", "Metric": "MAE", "Value": mae},
    {"Model": "Decision Tree", "Metric": "RMSE", "Value": rmse},
    {"Model": "Decision Tree", "Metric": "R2", "Value": r2},
    {
        "Model": "Training-mean baseline",
        "Metric": "MAE",
        "Value": baseline_mae,
    },
    {
        "Model": "Training-mean baseline",
        "Metric": "RMSE",
        "Value": baseline_rmse,
    },
    {
        "Model": "Training-mean baseline",
        "Metric": "R2",
        "Value": baseline_r2,
    },
])

metrics.to_csv(
    os.path.join(OUTPUT_DIR, "decision_tree_metrics.csv"),
    index=False,
)


# ============================================================
# STEP 9: FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": best_model.feature_importances_,
})

importance = importance.sort_values(
    "Importance",
    ascending=False,
)

importance.to_csv(
    os.path.join(OUTPUT_DIR, "decision_tree_feature_importance.csv"),
    index=False,
)

print("\n" + "=" * 65)
print("FEATURE IMPORTANCE")
print("=" * 65)

print(importance.to_string(index=False))

plt.figure(figsize=(10, 7))

importance_plot = importance.sort_values(
    "Importance",
    ascending=True,
)

plt.barh(
    importance_plot["Feature"],
    importance_plot["Importance"],
)

plt.xlabel("Feature Importance")
plt.ylabel("Voice Feature")
plt.title("Decision Tree: Voice Feature Importance")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "decision_tree_feature_importance.png"),
    dpi=300,
)

plt.close()


# ============================================================
# STEP 10: ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5,
)

plot_min = min(y_test.min(), y_pred.min())
plot_max = max(y_test.max(), y_pred.max())

plt.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    linestyle="--",
)

plt.xlabel("Actual total_UPDRS")
plt.ylabel("Predicted total_UPDRS")
plt.title("Decision Tree: Actual vs Predicted")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "decision_tree_actual_vs_predicted.png"),
    dpi=300,
)

plt.close()


# ============================================================
# STEP 11: VISUALIZE THE TREE
# ============================================================

plt.figure(figsize=(24, 12))

plot_tree(
    best_model,
    feature_names=FEATURES,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8,
)

plt.title(
    "Decision Tree Regressor (Top 3 Levels Shown)"
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "decision_tree_structure.png"),
    dpi=200,
)

plt.close()


# ============================================================
# STEP 12: FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)

print("Model: Decision Tree Regressor")
print("Target:", TARGET)
print("Input features:", len(FEATURES))
print("Best parameters:", grid_search.best_params_)

print(f"\nTest MAE:  {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R2:   {r2:.4f}")

print("\nOutput files saved in:", OUTPUT_DIR)
print("- decision_tree_metrics.csv")
print("- decision_tree_predictions.csv")
print("- decision_tree_feature_importance.csv")
print("- decision_tree_feature_importance.png")
print("- decision_tree_actual_vs_predicted.png")
print("- decision_tree_structure.png")
print("- model_comparison.csv")

print("\nDecision Tree experiment completed successfully.")
