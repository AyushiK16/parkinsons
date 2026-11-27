import importlib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from xgboost import XGBRegressor

preprocess_module = importlib.import_module("04_preprocess")
prepare_data = preprocess_module.prepare_data


def tune_and_evaluate():
  X, y, groups = prepare_data()
  gkf = GroupKFold(n_splits=5)

  # Define parameter search spaces
  xgb_params = {
      "n_estimators": [100, 200, 300],
      "max_depth": [3, 5, 7, 9],
      "learning_rate": [0.01, 0.03, 0.05, 0.1],
      "subsample": [0.6, 0.8, 1.0],
      "colsample_bytree": [0.6, 0.8, 1.0],
      "reg_alpha": [0.1, 1.0, 10.0],
      "reg_lambda": [0.1, 1.0, 10.0],
  }

  rf_params = {
      "n_estimators": [100, 200, 300],
      "max_depth": [6, 10, 15, 20],
      "min_samples_split": [2, 5, 10],
      "min_samples_leaf": [1, 2, 4],
  }

  print("\n" + "=" * 70)
  print("TUNING XGBOOST WITH GROUP CROSS-VALIDATION")
  print("=" * 70)

  xgb_search = RandomizedSearchCV(
      estimator=XGBRegressor(random_state=42, n_jobs=-1),
      param_distributions=xgb_params,
      n_iter=15,
      scoring="neg_root_mean_squared_error",
      cv=gkf,
      random_state=42,
      n_jobs=-1,
  )

  xgb_search.fit(X, y, groups=groups)

  print(f"Best XGBoost Parameters: {xgb_search.best_params_}")
  print(
      f"Best Cross-Validation RMSE: {-xgb_search.best_score_:.4f} UPDRS points"
  )

  # Evaluate best tuned model across folds
  best_xgb = xgb_search.best_estimator_

  mae_list, rmse_list, r2_list = [], [], []
  for train_idx, val_idx in gkf.split(X, y, groups=groups):
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

    best_xgb.fit(X_tr, y_tr)
    preds = best_xgb.predict(X_va)

    mae_list.append(mean_absolute_error(y_va, preds))
    rmse_list.append(np.sqrt(mean_squared_error(y_va, preds)))
    r2_list.append(r2_score(y_va, preds))

  print("\n" + "=" * 70)
  print("TUNED XGBOOST PERFORMANCE METRICS (CROSS-VALIDATION)")
  print("=" * 70)
  print(f"Mean MAE  : {np.mean(mae_list):.4f}")
  print(f"Mean RMSE : {np.mean(rmse_list):.4f}")
  print(f"Mean R²   : {np.mean(r2_list):.4f}")

  return best_xgb


if __name__ == "__main__":
  tune_and_evaluate()