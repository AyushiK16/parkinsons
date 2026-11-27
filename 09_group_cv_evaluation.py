import importlib
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

preprocess_module = importlib.import_module("04_preprocess")
prepare_data = preprocess_module.prepare_data


def cluster_bootstrap_ci(y_true, y_pred, groups, n_boot=1000, seed=42):
  """Computes 95% confidence intervals by resampling patients with replacement."""
  rng = np.random.default_rng(seed)
  unique_patients = np.unique(groups)
  n_patients = len(unique_patients)

  df_eval = pd.DataFrame({"y": y_true, "pred": y_pred, "g": groups})
  patient_groups = {p: sub for p, sub in df_eval.groupby("g")}

  r2_boot, mae_boot = [], []
  for _ in range(n_boot):
    sampled_patients = rng.choice(unique_patients, size=n_patients, replace=True)
    boot_df = pd.concat(
        [patient_groups[p] for p in sampled_patients], ignore_index=True
    )
    r2_boot.append(r2_score(boot_df["y"], boot_df["pred"]))
    mae_boot.append(mean_absolute_error(boot_df["y"], boot_df["pred"]))

  r2_ci = (np.percentile(r2_boot, 2.5), np.percentile(r2_boot, 97.5))
  mae_ci = (np.percentile(mae_boot, 2.5), np.percentile(mae_boot, 97.5))
  return r2_ci, mae_ci


def run_out_of_fold_eval(
    eval_df,
    cols,
    mode="delta",
    model_type="ridge",
    clip_delta=False,
    permute_target_anchor=False,
    permute_voice=False,
):
  X = eval_df.copy()
  y = eval_df["total_UPDRS"].values
  groups = eval_df["subject#"].values
  rng = np.random.default_rng(42)

  if permute_target_anchor:
    subj_anchors = X.groupby("subject#")["anchor_UPDRS"].first()
    shuffled = rng.permutation(subj_anchors.values)
    mapping = dict(zip(subj_anchors.index, shuffled))
    X["anchor_UPDRS"] = X["subject#"].map(mapping)

  if permute_voice:
    unique_subjs = X["subject#"].unique()
    shuffled_subjs = rng.permutation(unique_subjs)
    subj_map = dict(zip(unique_subjs, shuffled_subjs))
    voice_cols = [c for c in cols if c.endswith("_delta")]
    for c in voice_cols:
      temp_map = X.groupby("subject#")[c].apply(list).to_dict()
      new_vals = []
      for s, idx_grp in X.groupby("subject#", sort=False):
        donor = subj_map[s]
        donor_vals = temp_map[donor]
        reps = int(np.ceil(len(idx_grp) / len(donor_vals)))
        new_vals.extend((donor_vals * reps)[: len(idx_grp)])
      X[c] = new_vals

  y_delta = y - X["anchor_UPDRS"].values
  oof_preds = np.zeros(len(X))

  outer_gkf = GroupKFold(n_splits=7)
  inner_gkf = GroupKFold(n_splits=5)

  for tr_idx, va_idx in outer_gkf.split(X, y, groups=groups):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_delta_tr = y_delta[tr_idx]
    groups_tr = groups[tr_idx]

    if mode == "persistence":
      oof_preds[va_idx] = X_va["anchor_UPDRS"].values
      continue

    inner_splits = list(inner_gkf.split(X_tr, y_delta_tr, groups=groups_tr))

    if model_type == "ols":
      model = Pipeline(
          [("scaler", StandardScaler()), ("reg", LinearRegression())]
      )
    elif model_type == "ridge":
      model = Pipeline([
          ("scaler", StandardScaler()),
          ("reg", RidgeCV(alphas=np.logspace(-1, 4, 30), cv=inner_splits)),
      ])
    elif model_type == "elasticnet":
      model = Pipeline([
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
      ])

    model.fit(X_tr[cols], y_delta_tr)
    pred_delta = model.predict(X_va[cols])

    if clip_delta:
      low, high = np.percentile(y_delta_tr, [1, 99])
      pred_delta = np.clip(pred_delta, low, high)

    pred_total = X_va["anchor_UPDRS"].values + pred_delta
    if clip_delta:
      pred_total = np.clip(pred_total, 0.0, 176.0)

    oof_preds[va_idx] = pred_total

  r2 = r2_score(y, oof_preds)
  mae = mean_absolute_error(y, oof_preds)
  rmse = np.sqrt(mean_squared_error(y, oof_preds))
  r2_ci, mae_ci = cluster_bootstrap_ci(y, oof_preds, groups)
  return r2, mae, rmse, r2_ci, mae_ci


def run_mixedlm_inference(full_df, voice_delta_cols):
  """Fits Linear Mixed-Effects Models on the full cohort for statistical inference."""
  df_m = full_df.copy()
  df_m["subject"] = df_m["subject#"].astype(str)

  null_mod = smf.mixedlm("total_UPDRS ~ 1", df_m, groups=df_m["subject"]).fit(
      reml=True
  )
  var_between = float(null_mod.cov_re.iloc[0, 0])
  var_within = float(null_mod.scale)
  icc = (var_between / (var_between + var_within)) * 100

  voice_terms = " + ".join([f'Q("{c}")' for c in voice_delta_cols])
  formula = f"total_UPDRS ~ post_anchor_time_months + {voice_terms}"
  full_mod = smf.mixedlm(
      formula,
      df_m,
      groups=df_m["subject"],
      re_formula="~post_anchor_time_months",
  ).fit(reml=True, method="lbfgs")

  slope_est = full_mod.fe_params["post_anchor_time_months"]
  slope_ci = full_mod.conf_int().loc["post_anchor_time_months"].values
  slope_var = float(full_mod.cov_re.iloc[1, 1])

  wald_res = full_mod.wald_test(
      [f'Q("{c}") = 0' for c in voice_delta_cols], scalar=True
  )

  print("\n" + "=" * 95)
  print(
      "PART 4: LINEAR MIXED-EFFECTS INFERENCE (REML ON FULL COHORT, N = 42"
      " PATIENTS)"
  )
  print("=" * 95)
  print(
      f"Null-Model Intraclass Correlation (ICC) : {icc:.2f}% between-patient"
      " variance"
  )
  print(
      f"Fixed Population Progression Slope      : {slope_est:+.4f} UPDRS"
      f" pts/month (95% CI: [{slope_ci[0]:+.4f}, {slope_ci[1]:+.4f}])"
  )
  print(f"Random Subject Slope Variance           : {slope_var:.4f}")
  print(
      "Joint Wald Test for Voice Deltas        : chi2 ="
      f" {float(wald_res.statistic):.4f}, p-value ="
      f" {float(wald_res.pvalue):.4f}"
  )


def main():
  eval_df, full_df, time_cols, voice_delta_cols, raw_acoustic_cols, min_rec = (
      prepare_data(anchor_mode="day0")
  )
  hybrid_cols = ["anchor_UPDRS"] + time_cols + voice_delta_cols

  print("\n" + "=" * 95)
  print("PART 1: DATA LEAKAGE AUDIT (NAIVE RANDOM SPLIT VS. PATIENT-ISOLATED SPLIT)")
  print("=" * 95)
  print(
      "Minimum recordings per patient in 14-day voice intake window :"
      f" {min_rec}"
  )

  # Raw features (excluding subject# ID and motor_UPDRS)
  all_raw_cols = raw_acoustic_cols + ["test_time"]
  X_all = full_df[all_raw_cols]
  y_all = full_df["total_UPDRS"]
  g_all = full_df["subject#"]

  rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

  # 1A. Naive Ungrouped Random KFold -> ~0.99 R² (Identity Memorization Leakage)
  rf_leaky_r2, rf_leaky_mae = [], []
  for tr, va in KFold(n_splits=5, shuffle=True, random_state=42).split(X_all):
    rf.fit(X_all.iloc[tr], y_all.iloc[tr])
    preds = rf.predict(X_all.iloc[va])
    rf_leaky_r2.append(r2_score(y_all.iloc[va], preds))
    rf_leaky_mae.append(mean_absolute_error(y_all.iloc[va], preds))

  # 1B. Patient-Isolated GroupKFold -> Collapses on unseen patients
  rf_clean_r2, rf_clean_mae = [], []
  for tr, va in GroupKFold(n_splits=5).split(X_all, y_all, groups=g_all):
    rf.fit(X_all.iloc[tr], y_all.iloc[tr])
    preds = rf.predict(X_all.iloc[va])
    rf_clean_r2.append(r2_score(y_all.iloc[va], preds))
    rf_clean_mae.append(mean_absolute_error(y_all.iloc[va], preds))

  audit_df = pd.DataFrame([
      {
          "Baseline Audit Method": (
              "1A. Naive Random Forest (Ungrouped Random KFold - Leaky)"
          ),
          "R² Score": round(np.mean(rf_leaky_r2), 4),
          "MAE (UPDRS)": round(np.mean(rf_leaky_mae), 4),
          "Diagnosis": "Invalid: Memorizes patient voice & age/sex across splits",
      },
      {
          "Baseline Audit Method": (
              "1B. Cold-Start Random Forest (Patient GroupKFold - Clean)"
          ),
          "R² Score": round(np.mean(rf_clean_r2), 4),
          "MAE (UPDRS)": round(np.mean(rf_clean_mae), 4),
          "Diagnosis": "Fails: Cannot predict baseline UPDRS without Day-0 anchor",
      },
  ])
  print(audit_df.to_string(index=False))

  print("\n" + "=" * 95)
  print(
      "PART 2: MAIN LEAK-FREE ABLATION & CONTROLS (DAY-0 ANCHOR, POST-DAY 14"
      " FOLLOW-UPS)"
  )
  print("=" * 95)

  configs = [
      (
          "1. Persistence (Day-0 Anchor Only)",
          eval_df,
          None,
          "persistence",
          None,
          False,
          False,
          False,
      ),
      (
          "2. Anchor + Voice Deltas Only (No Time)",
          eval_df,
          voice_delta_cols,
          "delta",
          "ridge",
          False,
          False,
          False,
      ),
      (
          "3. Anchor + Time Trajectory (OLS)",
          eval_df,
          time_cols,
          "delta",
          "ols",
          False,
          False,
          False,
      ),
      (
          "4. Hybrid RidgeCV (Time + Voice Deltas)",
          eval_df,
          hybrid_cols,
          "delta",
          "ridge",
          False,
          False,
          False,
      ),
      (
          "5. Hybrid ElasticNetCV (Time + Voice Deltas)",
          eval_df,
          hybrid_cols,
          "delta",
          "elasticnet",
          False,
          False,
          False,
      ),
      (
          "6. Hybrid RidgeCV + Train-Fold Clipping",
          eval_df,
          hybrid_cols,
          "delta",
          "ridge",
          True,
          False,
          False,
      ),
      (
          "7. Neg. Control A: Permuted Anchor",
          eval_df,
          hybrid_cols,
          "delta",
          "ridge",
          False,
          True,
          False,
      ),
      (
          "8. Neg. Control B: Permuted Voice Deltas",
          eval_df,
          hybrid_cols,
          "delta",
          "ridge",
          False,
          False,
          True,
      ),
  ]

  rows = []
  for label, df_sub, cols, mode, mtype, clip, p_anc, p_voi in configs:
    r2, mae, rmse, r2_ci, mae_ci = run_out_of_fold_eval(
        df_sub, cols, mode, mtype, clip, p_anc, p_voi
    )
    rows.append({
        "Experiment": label,
        "OOF R² [95% Cluster CI]": f"{r2:.4f} [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]",
        "OOF MAE [95% Cluster CI]": (
            f"{mae:.4f} [{mae_ci[0]:.2f}, {mae_ci[1]:.2f}]"
        ),
        "OOF RMSE": round(rmse, 4),
    })

  print(pd.DataFrame(rows).to_string(index=False))

  print("\n" + "=" * 95)
  print(
      "PART 3: ANCHOR-DEFINITION ABLATION (QUANTIFYING INTERPOLATION"
      " CONTAMINATION)"
  )
  print("=" * 95)
  anchor_rows = []
  for a_mode, desc in [
      ("day0", "Day-0 Clinic Exam (Zero Interpolation Leak)"),
      ("mean14", "14-Day Label Mean (~8% Month-3 Slope Blend)"),
      ("mean35", "35-Day Label Mean (~20% Month-3 Slope Blend)"),
  ]:
    df_a, _, t_c, v_c, _, _ = prepare_data(anchor_mode=a_mode)
    h_c = ["anchor_UPDRS"] + t_c + v_c
    r2_p, mae_p, _, _, _ = run_out_of_fold_eval(df_a, None, "persistence")
    r2_h, mae_h, _, _, _ = run_out_of_fold_eval(df_a, h_c, "delta", "ridge")
    anchor_rows.append({
        "Anchor Definition": desc,
        "Persistence R²": round(r2_p, 4),
        "Persistence MAE": round(mae_p, 4),
        "Hybrid Ridge R²": round(r2_h, 4),
        "Hybrid Ridge MAE": round(mae_h, 4),
    })
  print(pd.DataFrame(anchor_rows).to_string(index=False))

  run_mixedlm_inference(full_df, voice_delta_cols)


if __name__ == "__main__":
  main()