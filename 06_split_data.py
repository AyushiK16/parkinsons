import importlib
from sklearn.model_selection import GroupShuffleSplit

preprocess_module = importlib.import_module("04_preprocess")
prepare_data = preprocess_module.prepare_data


def get_grouped_split(test_size=0.2, random_state=42):
  X, y, groups, acoustic_cols, time_cols = prepare_data()

  gss = GroupShuffleSplit(
      n_splits=1, test_size=test_size, random_state=random_state
  )
  train_idx, test_idx = next(gss.split(X, y, groups=groups))

  X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
  y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
  groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

  print("\n" + "=" * 60)
  print("PATIENT-WISE SPLIT VERIFICATION")
  print("=" * 60)
  print(f"Total Patients       : {groups.nunique()}")
  print(
      f"Training Patients    : {groups_train.nunique()} ({len(X_train)} samples)"
  )
  print(
      f"Testing Patients     : {groups_test.nunique()} ({len(X_test)} samples)"
  )
  print(
      "Overlapping Patients :"
      f" {len(set(groups_train.unique()).intersection(set(groups_test.unique())))}"
  )

  return (
      X_train,
      X_test,
      y_train,
      y_test,
      groups_train,
      groups_test,
      acoustic_cols,
      time_cols,
  )


if __name__ == "__main__":
  get_grouped_split()