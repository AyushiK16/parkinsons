"""Run six classifiers without sharing subjects across fitting and evaluation.

Entry point: python MODELS/classification/train_all_models.py
Inputs are the explicit experiment_specs.json written by preprocessing. Primary
metrics average recording scores per subject, then apply the fixed 0.5 threshold.
An SVM score is expit(decision_function), NOT a calibrated PD probability.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from logistic_regression import build_model as logistic_model
from decision_tree import build_model as tree_model
from random_forest import build_model as forest_model
from svm import build_model as svm_model
from knn import build_model as knn_model
from gradient_boosting import build_model as boosting_model


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "RESULTS" / "classification"
INTERNAL = ROOT / "RESULTS" / "_pipeline"
SEED = 42
BUILDERS = {
    "logistic_regression": logistic_model,
    "decision_tree": tree_model,
    "random_forest": forest_model,
    "svm": svm_model,
    "knn": knn_model,
    "gradient_boosting": boosting_model,
}


def json_safe(value):
    """Write strict JSON: unidentified metrics are null, not invented numbers."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name):
    """Keep experiment/model identifiers suitable for artifact bundle keys."""
    value = str(name)
    if not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in value):
        raise ValueError(f"Experiment/model names must be simple filename stems: {value!r}")
    return value


def load_experiment(spec):
    name = safe_name(spec["name"])
    path = (ROOT / spec["path"]).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"{name}: data path must remain inside project")
    frame = pd.read_csv(path, dtype={"subject_id": str, "recording_id": str})
    if spec.get("target", "target") != "target" or spec.get("group", "subject_id") != "subject_id":
        raise ValueError(f"{name}: expected target and subject_id contract")
    features = spec["features"]
    forbidden = {"target", "subject_id", "dataset_source", "diagnosis", "disease_stage", "recording_id", "split"}
    if not features or len(features) != len(set(features)) or forbidden.intersection(features):
        raise ValueError(f"{name}: feature list is empty, duplicated, or includes metadata")
    required = set(features) | forbidden
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing required columns: {sorted(missing)}")
    if frame[["subject_id", "dataset_source", "recording_id"]].isna().any().any():
        raise ValueError(f"{name}: missing identity/provenance values")
    frame["target"] = pd.to_numeric(frame["target"], errors="raise")
    frame["diagnosis"] = pd.to_numeric(frame["diagnosis"], errors="raise")
    for feature in features:
        frame[feature] = pd.to_numeric(frame[feature], errors="raise")
    frame[features] = frame[features].replace([np.inf, -np.inf], np.nan)
    if not frame["diagnosis"].isin([0, 1, 3]).all():
        raise ValueError(f"{name}: diagnoses must be Healthy=0, confirmed PD=1, or RBD=3")
    if frame.loc[frame["diagnosis"].eq(3), "target"].notna().any():
        raise ValueError(f"{name}: RBD must not carry a classification target")
    labeled = frame[frame["diagnosis"].isin([0, 1])].copy()
    if labeled["target"].isna().any() or not labeled["target"].eq(labeled["diagnosis"]).all():
        raise ValueError(f"{name}: labeled targets must reflect confirmed diagnoses")
    labeled["target"] = labeled["target"].astype(int)
    if (labeled.groupby("subject_id")["target"].nunique() > 1).any():
        raise ValueError(f"{name}: conflicting diagnoses within a subject")
    if (frame.groupby("subject_id")["dataset_source"].nunique() > 1).any():
        raise ValueError(f"{name}: subject IDs must be source-namespaced")
    if frame.groupby("subject_id")["split"].nunique().gt(1).any():
        raise ValueError(f"{name}: a subject appears in more than one split")
    return frame, labeled, path


def grouped_folds(frame, requested):
    """Stratify a subject table so prolific speakers cannot dominate folds."""
    subjects = frame[["subject_id", "target"]].drop_duplicates().reset_index(drop=True)
    counts = subjects["target"].value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("Grouped tuning needs at least two distinct subjects in each class")
    count = min(requested, int(counts.min()))
    splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=SEED)
    folds = []
    for train_subjects, test_subjects in splitter.split(subjects, subjects["target"], subjects["subject_id"]):
        train_ids = subjects.iloc[train_subjects]["subject_id"]
        test_ids = subjects.iloc[test_subjects]["subject_id"]
        train = np.flatnonzero(frame["subject_id"].isin(train_ids).to_numpy())
        test = np.flatnonzero(frame["subject_id"].isin(test_ids).to_numpy())
        if frame.iloc[train]["target"].nunique() != 2:
            raise ValueError("A grouped training fold has only one class; cannot fit classifiers honestly")
        folds.append((train, test))
    return folds


def make_pipeline(estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def score_records(pipeline, features):
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "predict_proba"):
        positive = list(estimator.classes_).index(1)
        return pipeline.predict_proba(features)[:, positive]
    # Monotone for recording AUC, but averaging expit margins defines the
    # predeclared subject-level score. No hidden ungrouped SVM calibration.
    return expit(pipeline.decision_function(features))


def prediction_tables(frame, scores, model, experiment, split, fold=None):
    columns = ["subject_id", "dataset_source", "diagnosis", "disease_stage", "recording_id", "target"]
    records = frame[columns].copy().reset_index(drop=True)
    records["score"] = np.asarray(scores)
    records["prediction"] = (records["score"] >= 0.5).astype(int)
    records["n_records"] = 1
    if fold is not None:
        records["fold"] = fold
    subjects = records.groupby("subject_id", sort=True).agg(
        dataset_source=("dataset_source", "first"),
        diagnosis=("diagnosis", "first"),
        disease_stage=("disease_stage", lambda values: "|".join(sorted(set(values.astype(str))))),
        target=("target", "first"), score=("score", "mean"),
        n_records=("recording_id", "size"),
    ).reset_index()
    subjects["prediction"] = (subjects["score"] >= 0.5).astype(int)
    if fold is not None:
        subjects["fold"] = fold
    for table, unit in ((records, "recording"), (subjects, "subject")):
        table["experiment"] = experiment
        table["model"] = model
        table["split"] = split
        table["evaluation_unit"] = unit
        table["threshold"] = 0.5
        table["score_kind"] = "expit_margin_not_calibrated_probability" if model == "svm" else "model_probability_not_clinically_calibrated"
    return records, subjects


def compute_metrics(table, external=False):
    truth = table["target"].to_numpy(dtype=int)
    predicted = table["prediction"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    both = len(np.unique(truth)) == 2
    metrics = {
        "accuracy": accuracy_score(truth, predicted) if both else np.nan,
        "precision": precision_score(truth, predicted, zero_division=0) if both else np.nan,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(truth, predicted, zero_division=0) if both else np.nan,
        "roc_auc": roc_auc_score(truth, table["score"]) if both else np.nan,
        "pr_auc": average_precision_score(truth, table["score"]) if both else np.nan,
        "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
        "n_subjects": table["subject_id"].nunique(),
        "n_records": int(table["n_records"].sum()),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    metrics["recall"] = metrics["sensitivity"]
    if external and np.all(truth == 1):
        # A PD-only external collection can estimate sensitivity, not screening
        # accuracy, precision, specificity, F1, AUC, or false-positive rates.
        for key in ("accuracy", "precision", "specificity", "f1", "roc_auc", "pr_auc", "false_positive_rate", "TN", "FP"):
            metrics[key] = np.nan
    return metrics


def rank_metrics(metrics):
    return tuple(float(metrics[key]) if np.isfinite(metrics[key]) else -np.inf
                 for key in ("roc_auc", "sensitivity", "specificity"))


def tune(frame, features, builder, folds):
    """Choose a grid candidate on grouped, SUBJECT-level out-of-fold scores."""
    estimator, grid = builder()
    base = make_pipeline(estimator)
    best_rank = None
    best_params = None
    candidates = []
    for parameters in ParameterGrid(grid):
        scores = np.full(len(frame), np.nan)
        for train_index, valid_index in folds:
            fitted = clone(base).set_params(**parameters)
            fitted.fit(frame.iloc[train_index][features], frame.iloc[train_index]["target"])
            scores[valid_index] = score_records(fitted, frame.iloc[valid_index][features])
        if not np.isfinite(scores).all():
            raise ValueError("Grouped out-of-fold scoring did not cover every training record")
        _, subjects = prediction_tables(frame, scores, "tuning", "tuning", "inner_oof")
        metrics = compute_metrics(subjects)
        candidates.append({"params": parameters, "subject_oof_metrics": metrics})
        rank = rank_metrics(metrics)
        if best_rank is None or rank > best_rank:
            best_rank, best_params = rank, parameters
    fitted = clone(base).set_params(**best_params)
    fitted.fit(frame[features], frame["target"])
    return fitted, best_params, candidates


def save_pipeline(pipeline, stem, spec, params, tuning, role, training_frame, models, pipeline_metadata):
    # Each tuning call returns a fresh fitted pipeline; none is fitted again.
    models[stem] = pipeline
    pipeline_metadata[stem] = {
        "experiment": spec["name"], "features_in_order": spec["features"],
        "target": "confirmed PD=1 versus Healthy=0; RBD excluded",
        "group": "subject_id", "role": role, "best_params": params,
        "tuning": tuning, "seed": SEED, "fixed_threshold": 0.5,
        "score_note": "SVM uses expit(raw margin), not calibrated probabilities. All subject scores are mean recording scores.",
        "preprocessing": "Training-fold median imputation, StandardScaler, classifier; serialized as one Pipeline",
        "training_n_subjects": training_frame["subject_id"].nunique(),
        "training_n_records": len(training_frame),
        "training_subject_ids": sorted(training_frame["subject_id"].unique()),
        "externals_used_for_fit_or_selection": False,
    }


def save_evaluation(records, subjects, experiment, model, split, rows, predictions):
    external = split == "external"
    for table, unit in ((subjects, "subject"), (records, "recording")):
        predictions.append(table)
        metrics = compute_metrics(table, external=external)
        row = {"experiment": experiment, "model": model, "split": split,
               "evaluation_unit": unit, "external_never_used_in_training": external,
               "threshold": 0.5, **metrics}
        rows.append(row)


def fold_manifest(frame, folds):
    return [{"fold": index + 1,
             "train_subjects": sorted(frame.iloc[train]["subject_id"].unique()),
             "evaluation_subjects": sorted(frame.iloc[test]["subject_id"].unique())}
            for index, (train, test) in enumerate(folds)]


def check_disjoint(*frames):
    seen = set()
    for frame in frames:
        ids = set(frame["subject_id"])
        if seen.intersection(ids):
            raise ValueError("Subject leakage between fitting/validation/test/external sets")
        seen.update(ids)


def exploratory_scores(frame, pipeline, spec, model, predictions):
    highrisk = frame[frame["diagnosis"].eq(3)].copy()
    if highrisk.empty:
        return
    columns = ["subject_id", "recording_id", "dataset_source", "diagnosis", "disease_stage"]
    output = highrisk[columns].copy()
    output["exploratory_model_score"] = score_records(pipeline, highrisk[spec["features"]])
    output["interpretation"] = "Exploratory uncalibrated score only; not a PD diagnosis, prognosis, or validated RBD prediction"
    output["used_in_training_or_evaluation"] = False
    output["experiment"] = spec["name"]
    output["model"] = model
    output["split"] = "rbd_exploratory"
    output["evaluation_unit"] = "recording"
    predictions.append(output)


def run_holdout(spec, frame, labeled, rows, predictions, models, pipeline_metadata, manifest):
    train = labeled[labeled["split"].eq("train")].copy().reset_index(drop=True)
    validation = labeled[labeled["split"].eq("validation")].copy().reset_index(drop=True)
    test = labeled[labeled["split"].eq("test")].copy().reset_index(drop=True)
    external = labeled[labeled["split"].eq("external")].copy().reset_index(drop=True)
    if spec.get("train_source"):
        train = train[train["dataset_source"].eq(spec["train_source"])].reset_index(drop=True)
        validation = validation[validation["dataset_source"].eq(spec["train_source"])].reset_index(drop=True)
    if spec.get("test_source"):
        test = labeled[labeled["dataset_source"].eq(spec["test_source"]) & ~labeled["split"].eq("external")].copy().reset_index(drop=True)
    if train.empty or validation.empty or test.empty:
        raise ValueError(f"{spec['name']}: holdout requires nonempty train, validation, and test sets")
    check_disjoint(train, validation, test, external)
    folds = grouped_folds(train, 3)
    manifest["inner_folds"] = fold_manifest(train, folds)
    manifest["evaluation_subjects"] = {
        name: sorted(part["subject_id"].unique())
        for name, part in (("validation", validation), ("test", test), ("external", external))
    }
    # Fit/tune every classifier and score validation BEFORE accessing any test
    # features. The selected winner is frozen in this explicit two-phase flow.
    fitted_models = {}
    validation_rows = []
    for model, builder in BUILDERS.items():
        print(f"{spec['name']}: tuning {model} on training subjects", flush=True)
        fitted, params, tuning = tune(train, spec["features"], builder, folds)
        fitted_models[model] = fitted
        save_pipeline(fitted, f"{spec['name']}__{model}", spec, params, tuning,
                      "Training-only fit; selected parameters by grouped inner CV; validation/test/external not fitted",
                      train, models, pipeline_metadata)
        records, subjects = prediction_tables(validation, score_records(fitted, validation[spec["features"]]), model, spec["name"], "validation")
        validation_rows.append({"experiment": spec["name"], "model": model, "split": "validation",
                                "evaluation_unit": "subject", **compute_metrics(subjects)})
        save_evaluation(records, subjects, spec["name"], model, "validation", rows, predictions)
    eligible = [row for row in validation_rows if np.isfinite(row["roc_auc"])]
    winner = max(eligible, key=rank_metrics) if eligible else None
    manifest["validation_selected_model"] = winner["model"] if winner else None
    manifest["selection_rule"] = "Subject validation ROC AUC, then sensitivity, then specificity; grid/order ties keep first; threshold fixed at 0.5"
    for model, fitted in fitted_models.items():
        for split, part in (("test", test), ("external", external)):
            if not part.empty:
                records, subjects = prediction_tables(part, score_records(fitted, part[spec["features"]]), model, spec["name"], split)
                save_evaluation(records, subjects, spec["name"], model, split, rows, predictions)
        exploratory_scores(frame, fitted, spec, model, predictions)
    return validation_rows, winner


def run_nested(spec, frame, labeled, rows, predictions, models, pipeline_metadata, manifest):
    cv = labeled[labeled["split"].eq("cv")].copy().reset_index(drop=True)
    external = labeled[labeled["split"].eq("external")].copy().reset_index(drop=True)
    if cv.empty:
        raise ValueError(f"{spec['name']}: nested evaluation requires split='cv' rows")
    check_disjoint(cv, external)
    outer_folds = grouped_folds(cv, 4)
    # These exact outer/inner subject partitions are reused for all classifiers.
    inner_folds = [grouped_folds(cv.iloc[train].reset_index(drop=True), 3) for train, _ in outer_folds]
    final_folds = grouped_folds(cv, 3)
    manifest["outer_folds"] = fold_manifest(cv, outer_folds)
    manifest["inner_folds_by_outer"] = [fold_manifest(cv.iloc[train].reset_index(drop=True), inner)
                                        for (train, _), inner in zip(outer_folds, inner_folds)]
    manifest["deployment_tuning_folds"] = fold_manifest(cv, final_folds)
    manifest["selection_note"] = "No classifier chosen by outer OOF or external performance; all six reported. Full-data deployments are not independently tested."
    for model, builder in BUILDERS.items():
        print(f"{spec['name']}: nested grouped CV {model}", flush=True)
        record_parts, subject_parts = [], []
        for fold, ((train_index, test_index), inner) in enumerate(zip(outer_folds, inner_folds), start=1):
            train = cv.iloc[train_index].reset_index(drop=True)
            test = cv.iloc[test_index].reset_index(drop=True)
            fitted, params, tuning = tune(train, spec["features"], builder, inner)
            save_pipeline(fitted, f"{spec['name']}__{model}__outer_fold_{fold}", spec, params, tuning,
                          f"Outer fold {fold}; not fitted to its outer evaluation subjects",
                          train, models, pipeline_metadata)
            records, subjects = prediction_tables(test, score_records(fitted, test[spec["features"]]), model, spec["name"], "nested_oof", fold)
            record_parts.append(records)
            subject_parts.append(subjects)
        save_evaluation(pd.concat(record_parts, ignore_index=True), pd.concat(subject_parts, ignore_index=True),
                        spec["name"], model, "nested_oof", rows, predictions)
        fitted, params, tuning = tune(cv, spec["features"], builder, final_folds)
        save_pipeline(fitted, f"{spec['name']}__{model}__deployment_not_independently_tested", spec, params, tuning,
                      "Full-data grouped-tuned deployment pipeline; no independent test of this fitted artifact; see outer-fold OOF estimates instead",
                      cv, models, pipeline_metadata)
        if not external.empty:
            records, subjects = prediction_tables(external, score_records(fitted, external[spec["features"]]), model, spec["name"], "external")
            save_evaluation(records, subjects, spec["name"], model, "external", rows, predictions)
        exploratory_scores(frame, fitted, spec, model, predictions)




def write_best_summary(eligible_winners):
    text = [
        "MODEL SELECTION: EARLY-STAGE VALIDATION ONLY",
        "Ranking: subject-level validation ROC AUC, then sensitivity, then specificity.",
        "Threshold fixed at 0.5; subject score is mean recording score.",
        "Test, nested outer OOF, external data, and RBD never select the winner.",
        "Models remain fitted on training only; validation is not refitted.",
        "SVM expit margins and other model scores are not clinically calibrated probabilities.",
    ]
    if eligible_winners:
        winner = max(eligible_winners, key=rank_metrics)
        text += [f"Selected experiment: {winner['experiment']}", f"Selected model: {winner['model']}",
                 f"Validation subject ROC AUC: {winner['roc_auc']:.6f}",
                 f"Validation sensitivity: {winner['sensitivity']:.6f}",
                 f"Validation specificity: {winner['specificity']:.6f}",
                 f"Pipeline bundle: {(RESULTS / 'models.joblib').relative_to(ROOT)}",
                 f"Pipeline stem: {winner['experiment']}__{winner['model']}",
                 "This is research evidence for confirmed-diagnosis discrimination, not prospective diagnostic validation."]
        if winner["sensitivity"] == 0:
            text += ["WARNING: the validation-AUC winner detects NONE of the validation PD subjects at the fixed threshold.",
                     "It is a ranking winner only, not a useful early-detection classifier or a deployment recommendation."]
    else:
        text += ["No eligible early-stage validation experiment with identifiable ROC AUC; no best-model claim is made."]
    return "\n".join(text) + "\n"


def main():
    INTERNAL.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    specs_path = INTERNAL / "experiment_specs.json"
    specs = json.loads(specs_path.read_text())
    if not isinstance(specs, list) or not specs:
        raise ValueError("experiment_specs.json must contain a nonempty list of real experiments")
    names = [safe_name(spec["name"]) for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("Experiment names must be unique")
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "python": platform.python_version(),
        "versions": {package: importlib.metadata.version(package) for package in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib")},
        "specs_sha256": file_hash(specs_path), "specifications": specs,
        "grids": {name: builder()[1] for name, builder in BUILDERS.items()},
        "estimator_defaults": {name: builder()[0].get_params() for name, builder in BUILDERS.items()},
        "threshold": 0.5,
        "primary_unit": "subject: average recording scores then threshold",
        "tuning_objective": "Grouped inner-OOF subject ROC AUC, then sensitivity, then specificity",
        "score_caveat": "SVM uses expit margin, not probability calibration; no model has clinical calibration evidence",
        "uncertainty_note": "Point estimates only. No confidence intervals or clinical generalizability claim.",
        "experiments": {},
        "pipeline_metadata": {},
    }
    rows, predictions, eligible_winners = [], [], []
    models = {}
    for spec in specs:
        frame, labeled, path = load_experiment(spec)
        detail = {"input_sha256": file_hash(path), "input_n_records": len(frame),
                  "input_n_subjects": frame["subject_id"].nunique(),
                  "rbd_n_records_excluded": int(frame["diagnosis"].eq(3).sum())}
        manifest["experiments"][spec["name"]] = detail
        if spec["evaluation"] == "holdout":
            _, winner = run_holdout(
                spec, frame, labeled, rows, predictions, models, manifest["pipeline_metadata"], detail,
            )
            selection_eligible = spec.get("selection_eligible", "early" in spec["name"].lower())
            if winner is not None and selection_eligible:
                eligible_winners.append(winner)
        elif spec["evaluation"] == "nested_group_cv":
            run_nested(spec, frame, labeled, rows, predictions, models, manifest["pipeline_metadata"], detail)
        else:
            raise ValueError(f"Unknown evaluation mode: {spec['evaluation']}")
    pd.DataFrame(rows).to_csv(RESULTS / "model_comparison.csv", index=False)
    identifying_columns = ["experiment", "model", "split", "evaluation_unit"]
    combined_predictions = pd.concat(predictions, ignore_index=True)
    combined_predictions = combined_predictions[
        identifying_columns + [column for column in combined_predictions if column not in identifying_columns]
    ]
    combined_predictions.to_csv(RESULTS / "predictions.csv", index=False)
    joblib.dump(models, RESULTS / "models.joblib")
    manifest["model_selection_summary"] = write_best_summary(eligible_winners)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["early_validation_selected_model"] = max(eligible_winners, key=rank_metrics) if eligible_winners else None
    save_json(RESULTS / "run_manifest.json", manifest)
    print(f"Finished: real evaluation artifacts written to {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
