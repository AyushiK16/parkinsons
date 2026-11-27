# Parkinson's Disease Detection and Severity Monitoring Using Voice Biomarkers

An academic machine-learning project with two independent objectives:

- **Classification:** distinguish confirmed Parkinson's disease (PD) from healthy controls using acoustic measurements.
- **Regression:** estimate continuous motor-UPDRS severity in people who already have PD.

This is not a medical device, prospective screening system, or clinically validated predictor of progression. Future longitudinal progression experiments are planned, **not implemented**. The existing regression experiment estimates severity; it does not forecast future visits.

## Installation

Use Python **3.12**. Run commands below from the `PARKINSONS_ML/` project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`libarchive-c` also needs the native libarchive library for UCI301's nested RAR. macOS normally supplies it; Debian/Ubuntu users can install `libarchive13`. Authenticated Kaggle CLI access is optional; never commit credentials.

## Folder structure

```text
PARKINSONS_ML/
├── README.md
├── requirements.txt
├── run_project.py
├── download_data.py
├── DATA/
│   └── raw/
│       ├── UCI_Early_Biomarkers/
│       ├── UCI_Multiple_Speech/
│       ├── UCI_Oxford_Voice/
│       │   └── telemonitoring/       # UCI189 regression data bundled in archive
│       └── KAGGLE_Voice/
├── PREPROCESSING/
│   ├── inspect_data.py
│   ├── extract_audio_features.py
│   ├── clean_data.py
│   ├── merge_datasets.py
│   └── prepare_train_test.py
├── MODELS/
│   ├── classification/
│   │   ├── train_all_models.py
│   │   ├── logistic_regression.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   ├── svm.py
│   │   ├── knn.py
│   │   └── gradient_boosting.py
│   └── regression/
│       └── experiment1_regression.py
└── RESULTS/
    ├── classification/
    │   ├── model_comparison.csv
    │   ├── models.joblib
    │   ├── predictions.csv
    │   └── run_manifest.json
    ├── regression/
    │   ├── experiment1_model_comparison.csv
    │   └── experiment1_patient_results.csv
    ├── reports/
    │   └── final_report.md
    └── _pipeline/                   # prepared data, schemas, caches and audits
```

Prepared data intentionally remain in `RESULTS/_pipeline/`, rather than a new `DATA/processed/` directory: existing schemas/specifications reference this relative path, and the training manifest records their hashes. Keeping it preserves saved artifact contents and reproducibility without rewriting metadata or adding path aliases. No empty `processed/` or `external/` directories are created.

Original source folders, archives and extracted files remain intact under `DATA/raw/`. The older sibling `PARKINSONS_EARLY_DETECTION copy/` is an unchanged historical snapshot, not another entry point for this organised project.

## Datasets and roles

| Source | Available data | Purpose |
|---|---|---|
| [UCI392 — Early Biomarkers](https://archive.ics.uci.edu/dataset/392/early+biomarkers+of+parkinson+s+disease+based+on+natural+connected+speech) | 130 participant summaries: 30 early untreated PD, 50 healthy, 50 RBD; 24 reading/monologue measurements | Primary early-PD versus healthy classification; RBD kept separate |
| [UCI301 — Multiple Speech](https://archive.ics.uci.edu/dataset/301/parkinson+speech+dataset+with+multiple+types+of+sound+recordings) | 1,040 training recordings from 40 people; 168 independent PD-only recordings from 28 people; 26 features | Classification and independent sensitivity check; one duplicate external row removed in derived data |
| [UCI174 — Oxford Parkinson's](https://archive.ics.uci.edu/dataset/174/parkinsons) | 195 recordings, 22 features; 32 observed subject-code prefixes | Classification using nested participant-grouped CV; PD stage unknown |
| [Kaggle voice source](https://www.kaggle.com/datasets/ucimachinelearning/parkinsons-voice-dataset) | 1,134 WAV files; 567 unique recordings after exact-audio deduplication | Extract and audit 44 descriptors; excluded from modeling without genuine participant IDs |
| [UCI189 — Parkinson's Telemonitoring](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring) | 5,875 recordings from 42 PD patients; 16 acoustic predictors | Independent motor-UPDRS regression experiment |

The telemonitoring `.data` and `.names` files were already bundled inside the downloaded Oxford archive. They remain at `DATA/raw/UCI_Oxford_Voice/telemonitoring/`; no separate download or duplicate copy is needed. The source notes describe motor/total UPDRS labels as linearly interpolated clinician scores.

The UCI classification sources already contain numerical measurements. Only Kaggle audio goes through feature extraction. No heterogeneous acoustic feature matrix is pooled across sources, and classification/regression datasets and targets are never mixed.

## What each script does

| Script | Responsibility |
|---|---|
| `run_project.py` | Orchestrates classification stages, verifies reloaded predictions and archive hashes, and generates the classification report. Never launches regression. |
| `download_data.py` | Acquires missing configured archives, extracts originals without replacing existing files, and saves acquisition metadata/checksums. |
| `PREPROCESSING/inspect_data.py` | Defines source schemas/parsers and inspects counts, IDs, duplicates, missing values and audio metadata; its loaders are reused by cleaning. |
| `PREPROCESSING/extract_audio_features.py` | Extracts Kaggle acoustic descriptors and audits audio/identity/duplicate issues. |
| `PREPROCESSING/clean_data.py` | Cleans tabular/audio feature copies, preserves provenance, and saves source-specific datasets, schemas and audits. |
| `PREPROCESSING/merge_datasets.py` | Checks metadata harmonisation, provenance and feature compatibility; prints the report without pooling incompatible features. |
| `PREPROCESSING/prepare_train_test.py` | Builds participant-disjoint classification splits and writes experiment datasets/specifications. |
| `MODELS/classification/train_all_models.py` | Tunes, trains and evaluates six classifiers; saves one comparison table, model bundle, prediction table and training manifest. |
| The six adjacent classifier files | Each defines one estimator and its existing parameter grid; the trainer imports them. They do not run training independently. |
| `MODELS/regression/experiment1_regression.py` | Standalone patient-wise motor-UPDRS regression comparison; saves only its two CSVs. |
| `requirements.txt` | Pinned project dependencies. |

All path resolution is based on each script's project root, not the caller's working directory. Scripts can also be invoked by absolute path.

## Classification: execution

Reuse the existing local datasets and extracted audio cache:

```bash
python run_project.py --skip-download --skip-audio
```

**This command still reruns preprocessing and classification training and overwrites generated classification artifacts.** It is not a read-only inspection command. Neither skip flag changes the ML methodology.

For a fresh output directory, `python run_project.py` runs all stages, including downloading missing sources and audio extraction. Existing raw files are not overwritten. pYIN extraction of the complete audio collection is CPU-intensive; do not repeat it just to verify a folder move.

Individual stages, in order:

```bash
python download_data.py
python PREPROCESSING/inspect_data.py
python PREPROCESSING/extract_audio_features.py
python PREPROCESSING/clean_data.py
python PREPROCESSING/merge_datasets.py
python PREPROCESSING/prepare_train_test.py
python MODELS/classification/train_all_models.py
python run_project.py --report-only
```

The downloader fetches source metadata even when archives already exist; use `--skip-download` to avoid those network requests. `--skip-audio` requires `RESULTS/_pipeline/kaggle_audio_features.csv`. Deleting `RESULTS/` removes the cached inputs, models and saved results; a full reconstruction then requires running without the skip flags.

To verify existing saved classification predictions **without fitting or writing results**:

```bash
python -B -c "import run_project; run_project.verify_reproducibility()"
```

To verify and regenerate the report without training:

```bash
python run_project.py --report-only
```

The latter **overwrites `RESULTS/reports/final_report.md`**. Do not use it when existing report bytes must remain preserved.

### Existing classification methodology

- Exact duplicates are removed from derived copies; selected impossible physical values become missing. IQR outliers are audited and retained, not clipped. IDs, clinical scales, medications and demographics are excluded from predictors.
- Audio is converted to mono and resampled to 22,050 Hz. Features are 13 MFCC mean/std pairs, spectral/energy/ZCR summaries, confidence-filtered pYIN pitch summaries and voiced fraction. Unreliable F0 remains missing; no cycle-level jitter/shimmer is fabricated.
- Median imputation and StandardScaler are fitted inside each training-fold pipeline. No global normalisation or learned feature selection is performed.
- The six classifiers are Logistic Regression, Decision Tree, Random Forest, RBF SVM, KNN and Gradient Boosting. Existing model settings/grids and seed 42 are preserved.
- UCI392 uses 56/12/12 train/validation/test participants; its 50 RBD participants are separate. UCI301 uses 28/6/6 participants, with every person's recordings together and the independent PD-only collection held out. Oxford uses four outer/three inner participant-grouped folds.
- Tuning ranks participant-level inner-OOF ROC-AUC, then sensitivity and specificity. Overall model selection uses early-stage validation only, never test/external data. Participant scores average recording scores before applying the fixed 0.5 threshold.
- SVM scores are sigmoid-transformed decision margins, not clinically calibrated probabilities. PD-only external evaluation cannot establish specificity or screening accuracy.

## Regression: independent execution

```bash
python MODELS/regression/experiment1_regression.py
```

This script does **not** require classification preprocessing or classification models. It reads the original telemonitoring table directly, uses only the supplied 16 voice features, and predicts `motor_UPDRS`. Subject ID, age, sex, test time and total UPDRS are excluded from predictors.

Seven-fold GroupKFold holds out six patients per fold and trains on the other 36; each patient is tested once. Models are the training-mean baseline, Linear Regression with training-fold StandardScaler, default Decision Tree and 100-tree Random Forest. There is no imputation, feature selection or hyperparameter tuning.

Outputs are MAE/RMSE/R² comparisons and per-patient analyses under `RESULTS/regression/`. Pooled metrics weight recordings equally; patient MAE averages recording errors within that patient. Fold standard deviation is not a confidence interval. The script saves evaluation CSVs, not fitted deployment models.

**Executing or importing this script runs its experiment and overwrites its two CSVs.** Do not import it to inspect paths or verify restructuring.

## Reading and preserving results

| Location | Contents |
|---|---|
| `RESULTS/classification/model_comparison.csv` | All classification models, splits and evaluation units in one table |
| `RESULTS/classification/models.joblib` | Dictionary of fitted classifiers and Oxford fold/full-data pipelines |
| `RESULTS/classification/predictions.csv` | Consolidated recording/participant predictions and explicitly exploratory RBD scores |
| `RESULTS/classification/run_manifest.json` | Version/grid metadata, participant folds, model metadata, selection summary and recorded hashes |
| `RESULTS/regression/` | Regression comparison and per-patient result CSVs |
| `RESULTS/reports/final_report.md` | Readable classification report, model selection, verification and limitations |
| `RESULTS/_pipeline/` | Required source-specific/experiment CSVs, schemas/specifications, audio cache, inventories and acquisition/cleaning/split audits |

For primary classification comparisons, filter `evaluation_unit == 'subject'` and `split == 'test'`; Oxford uses `split == 'nested_oof'`. Validation/external rows are separate evaluation contexts, not extra independent studies.

**Restructuring preservation:** existing datasets, models, CSVs, JSON artifacts and the report were moved byte-for-byte, not regenerated. Historical absolute locations and old bundle/report references inside saved artifacts describe their original run. Use this README for current paths; new executions use the new output locations. Prepared-data references and their recorded hash relationships remain valid because `_pipeline/` stayed in place.

Reload a saved classifier:

```python
from pathlib import Path
import json
import joblib
import pandas as pd

root = Path('.')
classification = root / 'RESULTS/classification'
stem = 'early_stage__logistic_regression'
manifest = json.loads((classification / 'run_manifest.json').read_text())
metadata = manifest['pipeline_metadata'][stem]
pipeline = joblib.load(classification / 'models.joblib')[stem]
data = pd.read_csv(root / 'RESULTS/_pipeline/experiment_early_stage.csv')
x = data.loc[data['split'].eq('test'), metadata['features_in_order']]
scores = pipeline.predict_proba(x)[:, 1]
```

Use the same feature order, units and recording protocol. Tabular models cannot accept raw audio or another source's schema. SVM instead uses `scipy.special.expit(pipeline.decision_function(x))`. Load only trusted joblib files.

## Current limitations and observed outcomes

- This is discrimination of already diagnosed participants, not proof of detecting disease before diagnosis. RBD has no longitudinal conversion endpoint; other classification PD cohorts lack established-stage labels. No stage labels are invented.
- Classification's early-stage validation-AUC winner, Random Forest, missed all four held-out PD patients at the fixed threshold. It is not recommended as an early-detection model.
- In the existing regression run, none of the learned models beat the training-mean baseline on pooled MAE. Severity estimation is not validated future progression prediction.
- Independent participant counts are small; no clinical calibration, confidence intervals or prospective early-stage external validation is established. Recording/task/source differences limit generalisation and cross-source comparisons.
- Oxford has 32 observed subject-code prefixes, while its published description says 31 people; the discrepancy remains unresolved and source IDs are preserved.
- Kaggle's folder labels do not establish clinical provenance or participant independence. Exact duplicate removal leaves 567 recordings, not 567 known people. A genuine provider-supplied `recording_id,subject_id` mapping is required before subject-disjoint modeling. If obtained, place it at `RESULTS/kaggle_subject_mapping.csv` (optional documented `disease_stage`); use recording paths relative to `DATA/raw/KAGGLE_Voice/`, such as `Voice_Dataset/Healthy/healthy_000.wav`. Do not invent IDs from filenames. Rerun extraction and downstream stages only when intentionally updating the experiment.

Original study citations and feature definitions remain in the downloaded `.names` files and acquisition metadata. The reorganisation adds no new experiment, clinical claim, dataset download, or ML-method change.
