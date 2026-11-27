from pathlib import Path
import json
import pandas as pd

CSV_PATH = Path("RESULTS/regression/paper_reproduction_comparison.csv")

# Change to "mean_test_rmse" if RMSE should determine the winner.
PRIMARY = "mean_test_mae"
SECONDARY = (
    "mean_test_rmse"
    if PRIMARY == "mean_test_mae"
    else "mean_test_mae"
)

df = pd.read_csv(CSV_PATH)

best = (
    df.dropna(subset=[PRIMARY, SECONDARY])
    .sort_values([PRIMARY, SECONDARY])
    .groupby(["protocol", "target"], sort=False)
    .head(1)
)

columns = [
    "protocol", "target", "model", "feature_set",
    "mean_test_mae", "mean_test_rmse", "pooled_test_r2",
]

print(f"\nBEST RESULTS — ranked by {PRIMARY}\n")
print(
    best[columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

for _, row in best.iterrows():
    print(f"\nFeatures: {row['protocol']} / {row['target']}")
    for features, count in json.loads(row["feature_subsets"]).items():
        print(f"  {features} — used in {count} split(s)")