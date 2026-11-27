import importlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Models & Preprocessing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Import grouped split function
split_module = importlib.import_module("06_split_data")
get_grouped_split = split_module.get_grouped_split


def evaluate_models():
  # 1. Obtain patient-wise train/test split
  X_train, X_test, y_train, y_test, _, _ = get_grouped_split()

  # 2. Define models (using Pipelines where scaling is needed)
  models = {
      "Linear Regression": Pipeline(
          [("scaler", StandardScaler()), ("regressor", LinearRegression())]
      ),
      "Ridge Regression": Pipeline(
          [(
              "scaler",
              StandardScaler(),
          ), ("regressor", Ridge(alpha=10.0, random_state=42))],
      ),
      "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
      "Random Forest": RandomForestRegressor(
          n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
      ),
      "XGBoost": XGBRegressor(
          n_estimators=100,
          learning_rate=0.05,
          max_depth=5,
          random_state=42,
          n_jobs=-1,
      ),
  }

  # 3. Train and Evaluate each model
  results = []

  print("\n" + "=" * 70)
  print("TRAINING AND EVALUATING 5 REGRESSION MODELS")
  print("=" * 70)

  for name, model in models.items():
    # Fit model on training set
    model.fit(X_train, y_train)

    # Make predictions on test set
    preds = model.predict(X_test)

    # Compute evaluation metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append(
        {"Model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R²": round(r2, 4)}
    )

  # 4. Display Summary Table
  results_df = pd.DataFrame(results).sort_values(by="RMSE")
  print("\nMODEL PERFORMANCE COMPARISON (TEST SET):")
  print(results_df.to_string(index=False))

  return results_df, models, X_train, X_test, y_train, y_test


if __name__ == "__main__":
  results_df, models, X_train, X_test, y_train, y_test = evaluate_models()