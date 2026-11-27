import importlib
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

preprocess_module = importlib.import_module("04_preprocess")
prepare_data = preprocess_module.prepare_data


def train_and_visualize():
  eval_df, _, time_cols, voice_delta_cols, _, _ = prepare_data(
      voice_window_days=14.0, anchor_mode="day0"
  )

  gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
  train_idx, test_idx = next(
      gss.split(eval_df, eval_df["total_UPDRS"], groups=eval_df["subject#"])
  )

  train_df = eval_df.iloc[train_idx]
  test_df = eval_df.iloc[test_idx]

  y_train_delta = train_df["total_UPDRS"] - train_df["anchor_UPDRS"]
  y_test = test_df["total_UPDRS"]
  groups_train = train_df["subject#"]

  inner_splits = list(
      GroupKFold(n_splits=5).split(train_df, y_train_delta, groups=groups_train)
  )
  hybrid_cols = ["anchor_UPDRS"] + time_cols + voice_delta_cols

  models = {
      "Persistence (Day-0 Anchor)": ("persistence", None, None),
      "Anchor + Voice Deltas Only": (
          "delta",
          Pipeline([
              ("scaler", StandardScaler()),
              ("reg", RidgeCV(alphas=np.logspace(-1, 4, 30), cv=inner_splits)),
          ]),
          voice_delta_cols,
      ),
      "Anchor + Time Trajectory (OLS)": (
          "delta",
          Pipeline(
              [("scaler", StandardScaler()), ("reg", LinearRegression())]
          ),
          time_cols,
      ),
      "Hybrid RidgeCV (Time + Voice)": (
          "delta",
          Pipeline([
              ("scaler", StandardScaler()),
              ("reg", RidgeCV(alphas=np.logspace(-1, 4, 30), cv=inner_splits)),
          ]),
          hybrid_cols,
      ),
      "Hybrid ElasticNetCV (Time + Voice)": (
          "delta",
          Pipeline([
              ("scaler", StandardScaler()),
              (
                  "reg",
                  ElasticNetCV(
                      l1_ratio=[0.1, 0.5, 0.9, 1.0],
                      alphas=np.logspace(-3, 1, 30),
                      cv=inner_splits,
                      max_iter=5000,
                      random_state=42,
                  ),
              ),
          ]),
          hybrid_cols,
      ),
  }

  results = []
  best_preds = None

  print("\n" + "=" * 76)
  print(
      f"{'Model (Day-0 Anchor, Post-Day 14)':<36} {'MAE':>10} {'RMSE':>10}"
      f" {'R²':>10}"
  )
  print("=" * 76)

  for name, (mode, model, cols) in models.items():
    if mode == "persistence":
      preds = test_df["anchor_UPDRS"].values
    else:
      model.fit(train_df[cols], y_train_delta)
      preds = model.predict(test_df[cols]) + test_df["anchor_UPDRS"].values

      if name == "Hybrid RidgeCV (Time + Voice)":
        best_preds = preds
        with open("unified_hybrid_model.pkl", "wb") as f:
          pickle.dump({"model": model, "features": hybrid_cols}, f)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"{name:<36} {mae:10.4f} {rmse:10.4f} {r2:10.4f}")

  res_df = pd.DataFrame(results).sort_values(by="R2", ascending=True)

  fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
  colors = [
      "#1b9e77" if "Hybrid" in m or "Time" in m else "#8da0cb"
      for m in res_df["Model"]
  ]
  bars = axes[0].barh(res_df["Model"], res_df["R2"], color=colors)
  axes[0].set_xlim(0, 1.0)
  axes[0].set_xlabel("Test R² (Unseen Patients, Post-Day 14)")
  axes[0].set_title("Model Comparison (Day-0 Clinical Anchor)")
  axes[0].grid(axis="x", linestyle="--", alpha=0.5)
  for bar, val in zip(bars, res_df["R2"]):
    axes[0].text(
        val + 0.015,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}",
        va="center",
        fontsize=9,
    )

  axes[1].scatter(y_test, best_preds, alpha=0.45, color="#1b9e77")
  lims = [
      min(y_test.min(), best_preds.min()) - 2,
      max(y_test.max(), best_preds.max()) + 2,
  ]
  axes[1].plot(lims, lims, "r--", linewidth=1.5, label="Ideal (y = x)")
  axes[1].set_xlim(lims)
  axes[1].set_ylim(lims)
  axes[1].set_xlabel("Actual total_UPDRS")
  axes[1].set_ylabel("Predicted total_UPDRS")
  axes[1].set_title("Actual vs. Predicted Severity (Unseen Patients)")
  axes[1].legend()
  axes[1].grid(True, linestyle="--", alpha=0.5)

  plt.tight_layout()
  plt.savefig("model_performance_comparison.png", dpi=300)
  print(
      "\nSaved 'model_performance_comparison.png' and 'unified_hybrid_model.pkl'!"
  )


if __name__ == "__main__":
  train_and_visualize()