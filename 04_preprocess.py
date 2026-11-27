import numpy as np
import pandas as pd


def prepare_data(
    filepath="parkinsons_updrs.data",
    voice_window_days=14.0,
    anchor_mode="day0",
):
  """
  anchor_mode:
    - 'day0'   : Pure Day-0 initial clinic exam score (strictly leak-free)
    - 'mean14' : 14-day window mean (ablation to measure interpolation blend)
    - 'mean35' : 35-day window mean (ablation to measure interpolation blend)
  """
  df = pd.read_csv(filepath)
  df = df.sort_values(by=["subject#", "test_time"]).reset_index(drop=True)

  # 1. Patient timeline alignment
  min_times = (
      df.groupby("subject#")["test_time"]
      .min()
      .rename("min_test_time")
      .reset_index()
  )
  df = pd.merge(df, min_times, on="subject#", how="left")
  df["delta_time"] = df["test_time"] - df["min_test_time"]

  # 2. Tsanas et al. (2010) log10 transforms for skewed perturbation features
  skewed_cols = [
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
  ]
  for col in skewed_cols:
    df[f"log_{col}"] = np.log10(df[col] + 1e-8)

  core_voice_cols = [
      "HNR",
      "RPDE",
      "DFA",
      "PPE",
      "log_Jitter(Abs)",
      "log_Jitter:PPQ5",
      "log_Shimmer(dB)",
      "log_Shimmer:APQ11",
      "log_NHR",
  ]

  # 3. Causal 14-day voice baseline (averaging noisy voice features is valid)
  voice_intake = df[df["delta_time"] <= voice_window_days]
  min_intake_recordings = voice_intake.groupby("subject#").size().min()

  voice_means = (
      voice_intake.groupby("subject#")[core_voice_cols]
      .mean()
      .add_prefix("early_")
      .reset_index()
  )
  df = pd.merge(df, voice_means, on="subject#", how="left")

  # 4. Clinical UPDRS Anchor definition
  if anchor_mode == "day0":
    anchor_df = (
        df.groupby("subject#")
        .agg(anchor_UPDRS=("total_UPDRS", "first"))
        .reset_index()
    )
    anchor_df["anchor_time"] = 0.0
  elif anchor_mode == "mean14":
    w = df[df["delta_time"] <= 14.0]
    anchor_df = (
        w.groupby("subject#")
        .agg(
            anchor_UPDRS=("total_UPDRS", "mean"),
            anchor_time=("delta_time", "mean"),
        )
        .reset_index()
    )
  elif anchor_mode == "mean35":
    w = df[df["delta_time"] <= 35.0]
    anchor_df = (
        w.groupby("subject#")
        .agg(
            anchor_UPDRS=("total_UPDRS", "mean"),
            anchor_time=("delta_time", "mean"),
        )
        .reset_index()
    )
  else:
    raise ValueError("anchor_mode must be 'day0', 'mean14', or 'mean35'")

  df = pd.merge(df, anchor_df, on="subject#", how="left")

  # Temporal features relative to anchor
  df["post_anchor_time"] = df["delta_time"] - df["anchor_time"]
  df["post_anchor_time_months"] = df["post_anchor_time"] / 30.4375
  df["post_anchor_time_sq"] = df["post_anchor_time"] ** 2
  df["anchor_x_time"] = df["anchor_UPDRS"] * df["post_anchor_time"]

  # Causal within-subject voice deltas
  voice_delta_cols = []
  for col in core_voice_cols:
    d_col = f"{col}_delta"
    df[d_col] = df[col] - df[f"early_{col}"]
    voice_delta_cols.append(d_col)

  # Strict post-calibration evaluation set (delta_time > 14 days)
  eval_df = df[df["delta_time"] > voice_window_days].reset_index(drop=True)

  time_cols = ["post_anchor_time", "post_anchor_time_sq", "anchor_x_time"]
  raw_acoustic_cols = core_voice_cols + ["age", "sex"]

  return (
      eval_df,
      df,
      time_cols,
      voice_delta_cols,
      raw_acoustic_cols,
      min_intake_recordings,
  )


if __name__ == "__main__":
  eval_df, full_df, _, _, _, min_rec = prepare_data(anchor_mode="day0")
  print(f"Minimum recordings per patient in 14-day voice window: {min_rec}")
  print(
      f"Strict Follow-Up Evaluation Set (Day > 14): {len(eval_df)} rows across"
      f" {eval_df['subject#'].nunique()} patients"
  )