"""Execute the complete project in dependency order and write the final report."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'RESULTS'
INTERNAL = RESULTS / '_pipeline'
CLASSIFICATION = RESULTS / 'classification'
REPORTS = RESULTS / 'reports'


def markdown_table(frame):
    # Avoid an extra tabulate dependency.
    def cell(value):
        if pd.isna(value):
            return 'Unavailable'
        if isinstance(value, (float, np.floating)):
            return f'{value:.4f}'
        return str(value).replace('|', '/')
    columns = list(frame.columns)
    return '\n'.join(['| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join(['---'] * len(columns)) + ' |'] +
        ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)])


def verify_reproducibility():
    """Exercise saved predictions and independently assert participant isolation."""
    specs = json.loads((INTERNAL / 'experiment_specs.json').read_text())
    manifest = json.loads((CLASSIFICATION / 'run_manifest.json').read_text())
    predictions = pd.read_csv(CLASSIFICATION / 'predictions.csv')
    pipelines = joblib.load(CLASSIFICATION / 'models.joblib')
    checks = []
    for spec in specs:
        data = pd.read_csv(ROOT / spec['path'])
        assert not data.groupby('subject_id').split.nunique().gt(1).any()
        assert data.loc[data.diagnosis.eq(3), 'target'].isna().all()
        detail = manifest['experiments'][spec['name']]
        folds = detail.get('inner_folds', []) + detail.get('outer_folds', [])
        for inner in detail.get('inner_folds_by_outer', []):
            folds += inner
        for fold in folds:
            assert not (set(fold['train_subjects']) & set(fold['evaluation_subjects']))
        for model in ('logistic_regression', 'decision_tree', 'random_forest', 'svm', 'knn', 'gradient_boosting'):
            reference = predictions.loc[
                predictions.experiment.eq(spec['name']) & predictions.model.eq(model)
                & predictions.evaluation_unit.eq('recording')
            ]
            if spec['evaluation'] == 'holdout':
                stem = f"{spec['name']}__{model}"
                reference = reference.loc[reference.split.eq('test')]
                sample = data.loc[data.recording_id.isin(reference.recording_id)].copy()
            else:
                stem = f"{spec['name']}__{model}__outer_fold_1"
                reference = reference.loc[reference.split.eq('nested_oof') & reference.fold.eq(1)]
                sample = data.loc[data.recording_id.isin(reference.recording_id)].copy()
            pipeline = pipelines[stem]
            if hasattr(pipeline, 'predict_proba'):
                scores = pipeline.predict_proba(sample[spec['features']])[:, list(pipeline.classes_).index(1)]
            else:
                scores = expit(pipeline.decision_function(sample[spec['features']]))
            actual = pd.Series(scores, index=sample.recording_id).sort_index()
            expected = reference.set_index('recording_id').score.sort_index()
            np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-10, atol=1e-12)
            checks.append(f"{spec['name']}/{model}: reloaded pipeline reproduces {len(actual)} held-out recording scores")
    downloads = json.loads((INTERNAL / 'download_manifest.json').read_text())
    for source, info in downloads.items():
        if info['status'] != 'downloaded':
            continue
        candidates = list((ROOT / 'DATA' / 'raw' / source).glob('*.zip'))
        assert any(hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256'] for path in candidates)
        checks.append(source + ': original downloaded archive checksum unchanged')
    print('\n'.join(checks), flush=True)
    return '\n'.join(checks) + '\nParticipant partitions and all recorded inner/outer folds are disjoint; RBD has no binary target.\n'


def final_report(verification):
    inventory = json.loads((INTERNAL / 'dataset_inventory.json').read_text())
    audit = json.loads((INTERNAL / 'cleaning_audit.json').read_text())
    splits = json.loads((INTERNAL / 'split_summary.json').read_text())
    comparison = pd.read_csv(CLASSIFICATION / 'model_comparison.csv')
    manifest = json.loads((CLASSIFICATION / 'run_manifest.json').read_text())
    selected = manifest['early_validation_selected_model']
    rows = []
    for name, info in audit.items():
        original = inventory.get(name, {})
        rows.append(dict(dataset=name, raw_records=info['raw_rows'], processed_records=info['processed_rows'],
            participants=info['participants'] if info['participants'] is not None else 'Unknown',
            class_counts=json.dumps(original.get('class_subjects', original.get('folder_label_counts', {}))),
            acoustic_features=info['feature_count']))
    counts = pd.DataFrame(rows)
    primary = comparison.loc[comparison.evaluation_unit.eq('subject') & comparison.split.isin(['test', 'nested_oof'])]
    columns = ['experiment', 'model', 'split', 'accuracy', 'precision', 'sensitivity', 'specificity', 'false_positive_rate', 'f1', 'roc_auc', 'pr_auc', 'n_subjects']
    downloads = json.loads((INTERNAL / 'download_manifest.json').read_text())
    downloaded_count = sum(item['status'] == 'downloaded' for item in downloads.values())
    audio_audit = audit.get('kaggle_voice', {})
    sections = ['# Final project report',
        '## Data actually acquired',
        f'{downloaded_count} of four specified datasets downloaded (see RESULTS/_pipeline/download_manifest.json for exact statuses, URLs and SHA-256 hashes). Original archives and extracted source files remain in DATA/raw. Exactly one Kaggle source is configured. UCI174 also bundles telemonitoring files: preserved as original archive contents and excluded from this classification pipeline. A separate motor-UPDRS regression experiment is documented in README.md.',
        markdown_table(counts),
        'UCI392 rows summarize participants across reading and monologue, not individual audio files. UCI301 external duplicate removal is audited in cleaning_audit.json. Oxford contains 32 distinct participant-code prefixes (24 PD/8 healthy) while its description claims 31 people (23 PD/8 healthy): unresolved, so original codes are preserved. Kaggle class counts describe audio-folder labels, not independently verified clinical participants; participant count/independence and stage are unknown.',
        '## Preprocessing and harmonization',
        'Two-row UCI392 headers parsed; clinical missing tokens (-/?) recognized. Subject/source/recording provenance retained. Exact raw-row duplicates removed; invalid physical values become missing. Outliers inspected using IQR and retained without disease-signal clipping. Clinical scales, medication, disease duration, demographics and identifiers excluded from predictors. Categorical diagnosis/stage mappings explicit. Raw audio decoded, converted to mono, resampled to 22,050 Hz; 13 MFCC mean/std plus spectral/energy/ZCR and confidence-filtered pYIN pitch descriptors extracted. Corrupt/short/silent recordings logged; unreliable pitch left missing. No fake cycle jitter/shimmer derived from frame F0/RMS.',
        f"Kaggle cleaning removed {audio_audit.get('exact_audio_duplicates_removed', 0)} exact decoded-audio duplicates; retained class counts: {audio_audit.get('retained_class_counts', {})}. Missing reliable F0 stays missing; the other valid descriptors are retained. Extraction and cleaning audits are in RESULTS/_pipeline/audio_extraction_report.json and RESULTS/_pipeline/cleaning_audit.json.",
        'Median imputation and StandardScaler live inside the saved model pipelines; fitted only on each training fold. No feature selection or whole-dataset normalization. No artificial zeros for missing schemas. Acoustic source schemas remain separate because validated cross-source measurement equivalence was not established. Only UCI301 training/external share the documented 26-feature schema, with external always held out. Required source-specific datasets are retained in RESULTS/_pipeline/; redundant metadata registries and subset CSVs are not saved.',
        'Kaggle extracted features are saved and audited but not modeled: the source contains no participant mapping. Inventing subjects or splitting recordings would make the requested leakage prevention unverifiable. To enable modeling, obtain the genuine recording-to-participant mapping from the data provider and place recording_id,subject_id in RESULTS/kaggle_subject_mapping.csv; do not create IDs from filenames. This is a missing scientific prerequisite, not a failed audio download.',
        '## Participant splits',
        'Early PD versus healthy: 70/15/15 split of 80 participants; all 50 RBD subjects separate. UCI301 original training cohort: 70/15/15 participant split; independent PD-only collection external. Oxford: nested four-fold subject CV with three-fold grouped tuning because only eight healthy participants make a 15% test split unreliable. All classifiers reuse the same partitions. Subject scores average recording scores; fixed threshold 0.5. SVM scores are expit margins, not calibrated probabilities.',
        '```json\n' + json.dumps(splits, indent=2) + '\n```',
        '## Model comparison — participant-level, untouched test or nested outer OOF',
        markdown_table(primary[columns]),
        'PR-AUC is average precision (not trapezoidal PR integration). Recall equals sensitivity. Recording-level and validation metrics and confusion counts are in RESULTS/classification/model_comparison.csv; predictions are consolidated in RESULTS/classification/predictions.csv. Source-specific scores are not directly comparable clinical benchmarks because cohorts, tasks, feature sets and evaluation designs differ.',
        '## Model selection',
        manifest['model_selection_summary'],
    ]
    if selected:
        sections += ['This choice is based on validation ROC-AUC, with sensitivity and specificity as tie-breakers, not test accuracy. Its held-out results appear in the comparison above. Validation/test sample sizes are small; the selected winner is tentative, not an established clinical screening model.']
        chosen = primary.loc[primary.experiment.eq(selected['experiment']) & primary.model.eq(selected['model'])].iloc[0]
        if selected['sensitivity'] == 0 or chosen['sensitivity'] == 0:
            sections += [f"**Operating-point failure:** selected model detected {int(chosen['TP'])} of {int(chosen['TP'] + chosen['FN'])} held-out early PD subjects. It is not recommended for early detection. Its apparent accuracy does not establish useful screening sensitivity; no threshold or selection criterion was changed after seeing the test set."]
    external = comparison.loc[comparison.evaluation_unit.eq('subject') & comparison.split.eq('external')]
    if not external.empty:
        sections += ['## Independent UCI301 PD-only evaluation',
            markdown_table(external[['model', 'sensitivity', 'FN', 'TP', 'n_subjects', 'n_records']]),
            'No healthy controls exist in this collection: specificity, false-positive rate, ROC/PR-AUC, screening precision/F1 and overall screening accuracy are not claimed.']
    sections += ['## Limitations for early detection',
        '- Only UCI392 supplies verified early untreated PD labels: 30 already-diagnosed patients and 50 controls. This does not establish prediction before diagnosis or future disease conversion.\n- RBD is a distinct high-risk group, not confirmed PD; exploratory model scores have no validated conversion endpoint.\n- No justified established-stage labels exist in the other sources; PD stage remains Unknown. A healthy/early/established three-class model cannot honestly be trained here.\n- Very small independent participant counts, male predominance in UCI392, acquisition/task/source confounding, and no clinical calibration or prospective external early-stage validation.\n- One fixed split and point estimates only; no confidence intervals or population screening prevalence adjustment. High test sensitivity/AUC on a few people is not clinical reliability.\n- No pooled cross-source model or cross-source transfer claim. No Kaggle evaluation without real participant identity/provenance. No longitudinal RBD conversion evaluation.',
        'Oxford grouping uses observed source participant codes. The unresolved published participant-count discrepancy means biological identity linkage cannot be independently guaranteed beyond those codes.',
        '## Verification', verification,
        '## Reproduce',
        'Use Python 3.12, `python -m pip install -r requirements.txt`, then `python run_project.py --skip-download --skip-audio` to reuse existing inputs. Already downloaded data are never overwritten. After deleting RESULTS, run without either skip flag. The report is under RESULTS/reports; comparisons, bundled models and predictions are under RESULTS/classification; required intermediate data are under RESULTS/_pipeline. Run regression separately with `python MODELS/regression/experiment1_regression.py`. Full commands, prediction example and provenance links are in README.md.']
    text = '\n\n'.join(sections) + '\n'
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / 'final_report.md').write_text(text)
    print('Final report:', REPORTS / 'final_report.md', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-download', action='store_true', help='Reuse the original downloaded files')
    parser.add_argument('--skip-audio', action='store_true', help='Reuse RESULTS/_pipeline/kaggle_audio_features.csv')
    parser.add_argument('--report-only', action='store_true', help='Verify existing trained pipelines and regenerate final report')
    args = parser.parse_args()
    if not args.report_only:
        steps = []
        if not args.skip_download:
            steps.append('download_data.py')
        steps.append('PREPROCESSING/inspect_data.py')
        if not args.skip_audio:
            steps.append('PREPROCESSING/extract_audio_features.py')
        elif not (INTERNAL / 'kaggle_audio_features.csv').exists():
            raise FileNotFoundError('--skip-audio requires existing extracted features; run without that flag')
        steps += ['PREPROCESSING/clean_data.py', 'PREPROCESSING/merge_datasets.py',
                  'PREPROCESSING/prepare_train_test.py', 'MODELS/classification/train_all_models.py']
        for script in steps:
            print('\nRUNNING', script, flush=True)
            subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)
    verification = verify_reproducibility()
    final_report(verification)


if __name__ == '__main__':
    main()
