import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import importlib

preprocess_module = importlib.import_module("04_preprocess")
prepare_data = preprocess_module.prepare_data


def analyze_correlations():
  # 1. Get preprocessed X and y
  X, y, groups = prepare_data()

  # Combine X and y temporarily for correlation computation
  df_corr = X.copy()
  df_corr['total_UPDRS'] = y

  # 2. Compute Pearson correlation matrix
  corr_matrix = df_corr.corr()

  # 3. Display correlations of features with total_UPDRS
  print("\n" + "=" * 60)
  print("CORRELATION OF FEATURES WITH TARGET (total_UPDRS)")
  print("=" * 60)
  target_corr = (
      corr_matrix['total_UPDRS']
      .drop('total_UPDRS')
      .sort_values(ascending=False)
  )
  print(target_corr.to_string())

  # 4. Plot Correlation Heatmap
  plt.figure(figsize=(14, 10))
  mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
  sns.heatmap(
      corr_matrix,
      mask=mask,
      annot=True,
      fmt=".2f",
      cmap="coolwarm",
      linewidths=0.5,
      cbar_kws={"shrink": 0.8},
  )
  plt.title(
      "Feature Correlation Heatmap (Parkinson's Telemonitoring)",
      fontsize=14,
      fontweight="bold",
  )
  plt.tight_layout()

  # Save figure
  output_image = "eda_correlation_heatmap.png"
  plt.savefig(output_image, dpi=300)
  print(f"\nCorrelation heatmap saved successfully as '{output_image}'!")
  plt.show()


if __name__ == "__main__":
  analyze_correlations()