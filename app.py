import importlib
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(
    page_title="Parkinson's UPDRS Telemonitoring", layout="wide"
)


@st.cache_resource
def load_or_train_model():
  if os.path.exists("unified_hybrid_model.pkl"):
    with open("unified_hybrid_model.pkl", "rb") as f:
      return pickle.load(f)

  preprocess_module = importlib.import_module("04_preprocess")
  eval_df, _, time_cols, voice_delta_cols, _, _ = (
      preprocess_module.prepare_data(
          voice_window_days=14.0, anchor_mode="day0"
      )
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
  return {"model": pipe, "features": hybrid_cols}


artifact = load_or_train_model()
model = artifact["model"]
feature_cols = artifact["features"]

st.title("Parkinson's Disease Longitudinal UPDRS Predictor")
st.markdown(
    "Predict follow-up **`total_UPDRS`** disease severity from a patient's"
    " **Day-0 clinical exam anchor**, elapsed days, and **14-day within-subject"
    " vocal shifts**."
)

col1, col2 = st.columns(2)

with col1:
  st.subheader("1. Clinical Anchor & Monitoring Timeline")
  anchor_updrs = st.number_input(
      "Day-0 Baseline Clinic Exam (anchor_UPDRS)",
      min_value=5.0,
      max_value=70.0,
      value=25.0,
      step=0.5,
  )
  post_anchor_time = st.slider(
      "Days Since Day-0 Exam (post_anchor_time)",
      min_value=15,
      max_value=200,
      value=90,
  )

with col2:
  st.subheader("2. Within-Subject Acoustic Shifts (vs. 14-Day Intake)")
  st.caption("Change relative to patient's own first 14-day vocal baseline")
  hnr_delta = st.slider("Δ HNR (dB)", -10.0, 10.0, -0.5, 0.1)
  rpde_delta = st.slider("Δ RPDE", -0.30, 0.30, 0.02, 0.01)
  dfa_delta = st.slider("Δ DFA", -0.20, 0.20, 0.01, 0.01)
  ppe_delta = st.slider("Δ PPE", -0.30, 0.30, 0.02, 0.01)
  log_jitter_abs_delta = st.slider(
      "Δ log10(Jitter Abs)", -0.5, 0.5, 0.02, 0.01
  )
  log_jitter_ppq5_delta = st.slider(
      "Δ log10(Jitter:PPQ5)", -0.5, 0.5, 0.02, 0.01
  )
  log_shimmer_db_delta = st.slider(
      "Δ log10(Shimmer dB)", -0.5, 0.5, 0.03, 0.01
  )
  log_shimmer_apq11_delta = st.slider(
      "Δ log10(Shimmer:APQ11)", -0.5, 0.5, 0.03, 0.01
  )
  log_nhr_delta = st.slider("Δ log10(NHR)", -0.5, 0.5, 0.02, 0.01)

input_row = {
    "anchor_UPDRS": anchor_updrs,
    "post_anchor_time": float(post_anchor_time),
    "post_anchor_time_sq": float(post_anchor_time) ** 2,
    "anchor_x_time": anchor_updrs * float(post_anchor_time),
    "HNR_delta": hnr_delta,
    "RPDE_delta": rpde_delta,
    "DFA_delta": dfa_delta,
    "PPE_delta": ppe_delta,
    "log_Jitter(Abs)_delta": log_jitter_abs_delta,
    "log_Jitter:PPQ5_delta": log_jitter_ppq5_delta,
    "log_Shimmer(dB)_delta": log_shimmer_db_delta,
    "log_Shimmer:APQ11_delta": log_shimmer_apq11_delta,
    "log_NHR_delta": log_nhr_delta,
}

input_df = pd.DataFrame([input_row])[feature_cols]
pred_delta = float(model.predict(input_df)[0])
pred_total = float(np.clip(anchor_updrs + pred_delta, 0.0, 176.0))

st.divider()
res_col1, res_col2, res_col3 = st.columns(3)
res_col1.metric("Predicted total_UPDRS", f"{pred_total:.2f}")
res_col2.metric(
    "Predicted Change (Δ UPDRS)",
    f"{pred_delta:+.2f}",
    delta_color="inverse",
)

if pred_total < 20:
  stage = "Mild Stage (UPDRS < 20)"
elif pred_total < 35:
  stage = "Moderate Stage (20 – 35)"
else:
  stage = "Advanced Stage (> 35)"

res_col3.metric("Clinical Severity Stage", stage)