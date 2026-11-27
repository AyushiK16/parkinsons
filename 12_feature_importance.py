import importlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

preprocess_module = importlib.import_module("04_preprocess")
prepare_data = preprocess_module.prepare_data


def plot_feature_importance():
  eval_df, _, time_cols, voice_delta_cols, _, _ = prepare_data(
      voice_window_days=14.0, anchor_mode="day0"
  )
  y_delta = eval_df["total_UPDRS"] - eval_df["anchor_UPDRS"]
  hybrid_cols = ["anchor_UPDRS"] + time_cols + voice_delta_cols

  cv_splits = list(
      GroupKFold(n_splits=7).split(
          eval_df, y_delta, groups=eval_df["subject#"]
      )
  )

  pipe = Pipeline([
      ("scaler", StandardScaler()),
      ("reg", RidgeCV(alphas=np.logspace(-1, 4, 30), cv=cv_splits)),
  ])
  pipe.fit(eval_df[hybrid_cols], y_delta)

  coefs = pipe.named_steps["reg"].coef_
  imp_df = pd.DataFrame({
      "Feature": hybrid_cols,
      "Coefficient": coefs,
      "AbsCoef": np.abs(coefs),
  }).sort_values(by="AbsCoef", ascending=True)

  plt.figure(figsize=(10, 6))
  colors = ["#d95f02" if c > 0 else "#1b9e77" for c in imp_df["Coefficient"]]
  plt.barh(imp_df["Feature"], imp_df["Coefficient"], color=colors)
  plt.axvline(0, color="black", linewidth=0.8)
  plt.xlabel("Standardized Ridge Coefficient (Effect on Δ total_UPDRS)")
  plt.title(
      "Drivers of Parkinson's Progression (Day-0 Anchor + 14-Day Voice Baseline)"
  )
  plt.grid(axis="x", linestyle="--", alpha=0.5)
  plt.tight_layout()
  plt.savefig("feature_importance.png", dpi=300)
  print("Saved standardized feature weights to 'feature_importance.png'!")


if __name__ == "__main__":
  plot_feature_importance()