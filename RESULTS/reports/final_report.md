# Final project report

## Data actually acquired

4 of four specified datasets downloaded (see RESULTS/_pipeline/download_manifest.json for exact statuses, URLs and SHA-256 hashes). Original archives and extracted source files remain in DATA/raw. Exactly one Kaggle source is configured. UCI174 also bundles telemonitoring files: preserved as original archive contents and excluded from this classification pipeline. A separate motor-UPDRS regression experiment is documented in README.md.

| dataset | raw_records | processed_records | participants | class_counts | acoustic_features |
| --- | --- | --- | --- | --- | --- |
| early_biomarkers | 130 | 130 | 130 | {"RBD": 50, "HC": 50, "PD": 30} | 24 |
| multiple_speech_training | 1040 | 1040 | 40 | {"1": 20, "0": 20} | 26 |
| multiple_speech_external | 168 | 167 | 28 | {"1": 28} | 26 |
| oxford_voice | 195 | 195 | 32 | {"1": 24, "0": 8} | 22 |
| kaggle_voice | 1134 | 567 | Unknown | {"Healthy": 574, "Parkinsons": 560} | 44 |

UCI392 rows summarize participants across reading and monologue, not individual audio files. UCI301 external duplicate removal is audited in cleaning_audit.json. Oxford contains 32 distinct participant-code prefixes (24 PD/8 healthy) while its description claims 31 people (23 PD/8 healthy): unresolved, so original codes are preserved. Kaggle class counts describe audio-folder labels, not independently verified clinical participants; participant count/independence and stage are unknown.

## Preprocessing and harmonization

Two-row UCI392 headers parsed; clinical missing tokens (-/?) recognized. Subject/source/recording provenance retained. Exact raw-row duplicates removed; invalid physical values become missing. Outliers inspected using IQR and retained without disease-signal clipping. Clinical scales, medication, disease duration, demographics and identifiers excluded from predictors. Categorical diagnosis/stage mappings explicit. Raw audio decoded, converted to mono, resampled to 22,050 Hz; 13 MFCC mean/std plus spectral/energy/ZCR and confidence-filtered pYIN pitch descriptors extracted. Corrupt/short/silent recordings logged; unreliable pitch left missing. No fake cycle jitter/shimmer derived from frame F0/RMS.

Kaggle cleaning removed 567 exact decoded-audio duplicates; retained class counts: {'0': 287, '1': 280}. Missing reliable F0 stays missing; the other valid descriptors are retained. Extraction and cleaning audits are in RESULTS/_pipeline/audio_extraction_report.json and RESULTS/_pipeline/cleaning_audit.json.

Median imputation and StandardScaler live inside the saved model pipelines; fitted only on each training fold. No feature selection or whole-dataset normalization. No artificial zeros for missing schemas. Acoustic source schemas remain separate because validated cross-source measurement equivalence was not established. Only UCI301 training/external share the documented 26-feature schema, with external always held out. Required source-specific datasets are retained in RESULTS/_pipeline/; redundant metadata registries and subset CSVs are not saved.

Kaggle extracted features are saved and audited but not modeled: the source contains no participant mapping. Inventing subjects or splitting recordings would make the requested leakage prevention unverifiable. To enable modeling, obtain the genuine recording-to-participant mapping from the data provider and place recording_id,subject_id in RESULTS/kaggle_subject_mapping.csv; do not create IDs from filenames. This is a missing scientific prerequisite, not a failed audio download.

## Participant splits

Early PD versus healthy: 70/15/15 split of 80 participants; all 50 RBD subjects separate. UCI301 original training cohort: 70/15/15 participant split; independent PD-only collection external. Oxford: nested four-fold subject CV with three-fold grouped tuning because only eight healthy participants make a 15% test split unreliable. All classifiers reuse the same partitions. Subject scores average recording scores; fixed threshold 0.5. SVM scores are expit margins, not calibrated probabilities.

```json
{
  "kaggle_voice": {
    "status": "excluded_from_modeling",
    "reason": "No documented recording-to-participant mapping; a recording split could leak speakers.",
    "available_processed_audio_rows": 567,
    "missing_subject_ids": 567
  },
  "early_stage": {
    "evaluation": "holdout",
    "feature_count": 24,
    "partitions": {
      "highrisk": {
        "records": 50,
        "participants": 50,
        "participant_classes": {
          "3": 50
        }
      },
      "test": {
        "records": 12,
        "participants": 12,
        "participant_classes": {
          "0": 8,
          "1": 4
        }
      },
      "train": {
        "records": 56,
        "participants": 56,
        "participant_classes": {
          "0": 35,
          "1": 21
        }
      },
      "validation": {
        "records": 12,
        "participants": 12,
        "participant_classes": {
          "0": 7,
          "1": 5
        }
      }
    },
    "limitations": "70/15/15 by participants, stratified diagnosis. Small validation/test counts yield unstable point estimates."
  },
  "multiple_speech": {
    "evaluation": "holdout",
    "feature_count": 26,
    "partitions": {
      "external": {
        "records": 167,
        "participants": 28,
        "participant_classes": {
          "1": 28
        }
      },
      "test": {
        "records": 156,
        "participants": 6,
        "participant_classes": {
          "1": 3,
          "0": 3
        }
      },
      "train": {
        "records": 728,
        "participants": 28,
        "participant_classes": {
          "1": 14,
          "0": 14
        }
      },
      "validation": {
        "records": 156,
        "participants": 6,
        "participant_classes": {
          "1": 3,
          "0": 3
        }
      }
    },
    "limitations": "70/15/15 by participants, stratified diagnosis. Small validation/test counts yield unstable point estimates."
  },
  "oxford_voice": {
    "evaluation": "nested_group_cv",
    "feature_count": 22,
    "partitions": {
      "cv": {
        "records": 195,
        "participants": 32,
        "participant_classes": {
          "1": 24,
          "0": 8
        }
      }
    },
    "limitations": "Only eight healthy Oxford participants: use four-fold outer / three-fold inner grouped CV instead of ~1 healthy person per holdout."
  }
}
```

## Model comparison — participant-level, untouched test or nested outer OOF

| experiment | model | split | accuracy | precision | sensitivity | specificity | false_positive_rate | f1 | roc_auc | pr_auc | n_subjects |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early_stage | logistic_regression | test | 0.5000 | 0.3333 | 0.5000 | 0.5000 | 0.5000 | 0.4000 | 0.4062 | 0.3305 | 12 |
| early_stage | decision_tree | test | 0.5000 | 0.2500 | 0.2500 | 0.6250 | 0.3750 | 0.2500 | 0.4531 | 0.3333 | 12 |
| early_stage | random_forest | test | 0.6667 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5180 | 12 |
| early_stage | svm | test | 0.6667 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.4375 | 0.3439 | 12 |
| early_stage | knn | test | 0.4167 | 0.0000 | 0.0000 | 0.6250 | 0.3750 | 0.0000 | 0.3750 | 0.3238 | 12 |
| early_stage | gradient_boosting | test | 0.6667 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5781 | 0.4437 | 12 |
| multiple_speech | logistic_regression | test | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.3333 | 0.6667 | 0.7778 | 0.8667 | 6 |
| multiple_speech | decision_tree | test | 0.1667 | 0.0000 | 0.0000 | 0.3333 | 0.6667 | 0.0000 | 0.2222 | 0.4444 | 6 |
| multiple_speech | random_forest | test | 0.6667 | 1.0000 | 0.3333 | 1.0000 | 0.0000 | 0.5000 | 0.5556 | 0.7000 | 6 |
| multiple_speech | svm | test | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.3333 | 0.6667 | 0.5556 | 0.5889 | 6 |
| multiple_speech | knn | test | 0.6667 | 1.0000 | 0.3333 | 1.0000 | 0.0000 | 0.5000 | 0.8889 | 0.9167 | 6 |
| multiple_speech | gradient_boosting | test | 0.5000 | 0.5000 | 0.3333 | 0.6667 | 0.3333 | 0.4000 | 0.6667 | 0.7556 | 6 |
| oxford_voice | logistic_regression | nested_oof | 0.6562 | 0.8095 | 0.7083 | 0.5000 | 0.5000 | 0.7556 | 0.7552 | 0.9116 | 32 |
| oxford_voice | decision_tree | nested_oof | 0.6875 | 0.7333 | 0.9167 | 0.0000 | 1.0000 | 0.8148 | 0.4766 | 0.7574 | 32 |
| oxford_voice | random_forest | nested_oof | 0.7500 | 0.7857 | 0.9167 | 0.2500 | 0.7500 | 0.8462 | 0.7552 | 0.9169 | 32 |
| oxford_voice | svm | nested_oof | 0.6875 | 0.8182 | 0.7500 | 0.5000 | 0.5000 | 0.7826 | 0.6042 | 0.7661 | 32 |
| oxford_voice | knn | nested_oof | 0.8125 | 0.8214 | 0.9583 | 0.3750 | 0.6250 | 0.8846 | 0.6484 | 0.8242 | 32 |
| oxford_voice | gradient_boosting | nested_oof | 0.8438 | 0.8276 | 1.0000 | 0.3750 | 0.6250 | 0.9057 | 0.6510 | 0.8090 | 32 |

PR-AUC is average precision (not trapezoidal PR integration). Recall equals sensitivity. Recording-level and validation metrics and confusion counts are in RESULTS/classification/model_comparison.csv; predictions are consolidated in RESULTS/classification/predictions.csv. Source-specific scores are not directly comparable clinical benchmarks because cohorts, tasks, feature sets and evaluation designs differ.

## Model selection

MODEL SELECTION: EARLY-STAGE VALIDATION ONLY
Ranking: subject-level validation ROC AUC, then sensitivity, then specificity.
Threshold fixed at 0.5; subject score is mean recording score.
Test, nested outer OOF, external data, and RBD never select the winner.
Models remain fitted on training only; validation is not refitted.
SVM expit margins and other model scores are not clinically calibrated probabilities.
Selected experiment: early_stage
Selected model: random_forest
Validation subject ROC AUC: 0.714286
Validation sensitivity: 0.000000
Validation specificity: 1.000000
Pipeline bundle: RESULTS/classification/models.joblib
Pipeline stem: early_stage__random_forest
This is research evidence for confirmed-diagnosis discrimination, not prospective diagnostic validation.
WARNING: the validation-AUC winner detects NONE of the validation PD subjects at the fixed threshold.
It is a ranking winner only, not a useful early-detection classifier or a deployment recommendation.


This choice is based on validation ROC-AUC, with sensitivity and specificity as tie-breakers, not test accuracy. Its held-out results appear in the comparison above. Validation/test sample sizes are small; the selected winner is tentative, not an established clinical screening model.

**Operating-point failure:** selected model detected 0 of 4 held-out early PD subjects. It is not recommended for early detection. Its apparent accuracy does not establish useful screening sensitivity; no threshold or selection criterion was changed after seeing the test set.

## Independent UCI301 PD-only evaluation

| model | sensitivity | FN | TP | n_subjects | n_records |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.1071 | 25 | 3 | 28 | 167 |
| decision_tree | 0.2143 | 22 | 6 | 28 | 167 |
| random_forest | 0.3929 | 17 | 11 | 28 | 167 |
| svm | 0.5000 | 14 | 14 | 28 | 167 |
| knn | 0.5357 | 13 | 15 | 28 | 167 |
| gradient_boosting | 0.3571 | 18 | 10 | 28 | 167 |

No healthy controls exist in this collection: specificity, false-positive rate, ROC/PR-AUC, screening precision/F1 and overall screening accuracy are not claimed.

## Limitations for early detection

- Only UCI392 supplies verified early untreated PD labels: 30 already-diagnosed patients and 50 controls. This does not establish prediction before diagnosis or future disease conversion.
- RBD is a distinct high-risk group, not confirmed PD; exploratory model scores have no validated conversion endpoint.
- No justified established-stage labels exist in the other sources; PD stage remains Unknown. A healthy/early/established three-class model cannot honestly be trained here.
- Very small independent participant counts, male predominance in UCI392, acquisition/task/source confounding, and no clinical calibration or prospective external early-stage validation.
- One fixed split and point estimates only; no confidence intervals or population screening prevalence adjustment. High test sensitivity/AUC on a few people is not clinical reliability.
- No pooled cross-source model or cross-source transfer claim. No Kaggle evaluation without real participant identity/provenance. No longitudinal RBD conversion evaluation.

Oxford grouping uses observed source participant codes. The unresolved published participant-count discrepancy means biological identity linkage cannot be independently guaranteed beyond those codes.

## Verification

early_stage/logistic_regression: reloaded pipeline reproduces 12 held-out recording scores
early_stage/decision_tree: reloaded pipeline reproduces 12 held-out recording scores
early_stage/random_forest: reloaded pipeline reproduces 12 held-out recording scores
early_stage/svm: reloaded pipeline reproduces 12 held-out recording scores
early_stage/knn: reloaded pipeline reproduces 12 held-out recording scores
early_stage/gradient_boosting: reloaded pipeline reproduces 12 held-out recording scores
multiple_speech/logistic_regression: reloaded pipeline reproduces 156 held-out recording scores
multiple_speech/decision_tree: reloaded pipeline reproduces 156 held-out recording scores
multiple_speech/random_forest: reloaded pipeline reproduces 156 held-out recording scores
multiple_speech/svm: reloaded pipeline reproduces 156 held-out recording scores
multiple_speech/knn: reloaded pipeline reproduces 156 held-out recording scores
multiple_speech/gradient_boosting: reloaded pipeline reproduces 156 held-out recording scores
oxford_voice/logistic_regression: reloaded pipeline reproduces 49 held-out recording scores
oxford_voice/decision_tree: reloaded pipeline reproduces 49 held-out recording scores
oxford_voice/random_forest: reloaded pipeline reproduces 49 held-out recording scores
oxford_voice/svm: reloaded pipeline reproduces 49 held-out recording scores
oxford_voice/knn: reloaded pipeline reproduces 49 held-out recording scores
oxford_voice/gradient_boosting: reloaded pipeline reproduces 49 held-out recording scores
UCI_Early_Biomarkers: original downloaded archive checksum unchanged
UCI_Multiple_Speech: original downloaded archive checksum unchanged
UCI_Oxford_Voice: original downloaded archive checksum unchanged
KAGGLE_Voice: original downloaded archive checksum unchanged
Participant partitions and all recorded inner/outer folds are disjoint; RBD has no binary target.


## Reproduce

Use Python 3.12, `python -m pip install -r requirements.txt`, then `python run_project.py --skip-download --skip-audio` to reuse existing inputs. Already downloaded data are never overwritten. After deleting RESULTS, run without either skip flag. The report is under RESULTS/reports; comparisons, bundled models and predictions are under RESULTS/classification; required intermediate data are under RESULTS/_pipeline. Run regression separately with `python MODELS/regression/experiment1_regression.py`. Full commands, prediction example and provenance links are in README.md.
