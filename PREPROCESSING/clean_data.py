"""Deterministic source-specific cleaning; learned transformations belong in model CV."""
import json
import numpy as np
import pandas as pd
from inspect_data import ROOT, RESULTS, load_raw, subject_codes, raw_labels

METADATA = ['subject_id', 'dataset_source', 'diagnosis', 'disease_stage',
            'disease_stage_label', 'recording_id', 'collection', 'target']


def clean_table(name, item):
    raw = item['frame']
    duplicate_count = int(raw.duplicated().sum())
    frame = raw.loc[~raw.duplicated()].copy()
    # Keep original row positions: row locators are provenance, not patient IDs.
    row_numbers = frame.index + 1
    labels = raw_labels(name, frame)
    codes = subject_codes(name, {**item, 'frame': frame})
    prefix = item['source'] + ':' + item['collection'] + ':'
    output = pd.DataFrame(index=frame.index)
    output['subject_id'] = prefix + codes
    output['dataset_source'] = item['source']
    if name == 'early_biomarkers':
        output['diagnosis'] = labels.map({'HC': 0, 'PD': 1, 'RBD': 3})
        if output.diagnosis.isna().any():
            raise ValueError('Unknown early-biomarker participant prefix')
        output['disease_stage'] = labels.map({'HC': 'Not applicable', 'PD': 'Early-stage', 'RBD': 'High-risk'})
        output['disease_stage_label'] = labels.map({'HC': '0', 'PD': '1', 'RBD': '3'})
        output['recording_id'] = 'dataset.csv:subject_summary:' + codes
    else:
        if not set(labels.unique()).issubset({0, 1}):
            raise ValueError('Unexpected binary diagnosis label')
        output['diagnosis'] = labels.astype(int)
        output['disease_stage'] = np.where(labels.eq(1), 'Unknown', 'Not applicable')
        output['disease_stage_label'] = np.where(labels.eq(1), 'Unknown', '0')
        if name == 'oxford_voice':
            output['recording_id'] = frame['name']
        else:
            filename = 'train_data.txt' if item['collection'] == 'training' else 'test_data.txt'
            output['recording_id'] = [f'{filename}:row:{i}' for i in row_numbers]
    output['collection'] = item['collection']
    output['target'] = output.diagnosis.where(output.diagnosis.isin([0, 1]))
    invalid = {}
    outliers = {}
    for feature in item['features']:
        values = pd.to_numeric(frame[feature], errors='coerce').replace([np.inf, -np.inf], np.nan)
        bad = pd.Series(False, index=values.index)
        # Only physical constraints, not percentile-based clipping.
        if name == 'early_biomarkers':
            if any(token in feature for token in ['duration_ms', 'latency_ms', 'entropy', 'gaping_per_minute', 'pauses_per_respiration', 'respiration_rate_per_minute']):
                bad |= values < 0
        elif name.startswith('multiple_speech'):
            if feature != 'harmonics_to_noise_db':
                bad |= values < 0
            if feature.endswith('_percent'):
                bad |= values > 100
            if feature == 'autocorrelation':
                bad |= values > 1
            if feature in ['pitch_mean_hz', 'pitch_median_hz', 'pitch_min_hz', 'pitch_max_hz', 'period_mean_seconds']:
                bad |= values <= 0
        else:
            if feature not in ['spread1', 'spread2', 'hnr']:
                bad |= values < 0
            if feature in ['mdvp_fo_hz', 'mdvp_fhi_hz', 'mdvp_flo_hz']:
                bad |= values <= 0
        invalid[feature] = int(bad.sum())
        values = values.mask(bad)
        q1, q3 = values.quantile([.25, .75])
        iqr = q3 - q1
        outliers[feature] = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        output[feature] = values
    if output.subject_id.isna().any() or output.groupby('subject_id').diagnosis.nunique().gt(1).any():
        raise ValueError('Missing participant identifier or conflicting diagnosis')
    audit = dict(raw_rows=len(raw), processed_rows=len(output), participants=output.subject_id.nunique(),
        feature_count=len(item['features']), exact_duplicates_removed=duplicate_count,
        exact_duplicate_source_rows_removed=[int(i + 1) for i in raw.index[raw.duplicated()]],
        source_missing_values=raw.isna().sum().to_dict(),
        invalid_values_to_missing={k:v for k,v in invalid.items() if v},
        missing_feature_values=output[item['features']].isna().sum().to_dict(),
        iqr_outlier_counts=outliers, outlier_action='Inspected and retained: extreme speech measurements can be disease signal; no justified clipping.',
        categorical_action='Diagnosis and stage mapped explicitly. No categorical predictors retained: demographics, medications, clinical scales and IDs excluded to avoid diagnostic leakage.',
        imputation_scaling='Not applied globally. Median imputation and StandardScaler fit inside training-only grouped CV pipelines, then applied to validation/test.',
        notes=item['notes'])
    return output.reset_index(drop=True), audit


def clean_audio():
    path = RESULTS / 'kaggle_audio_features.csv'
    if not path.exists():
        return None
    from extract_audio_features import AUDIO_FEATURES
    audio = pd.read_csv(path)
    if audio.empty:
        return None
    # Extraction rows retain all originals; model-ready copy excludes invalid clips.
    usable = audio.loc[audio['extraction_status'].isin(['ok', 'partial'])].copy()
    hash_column = 'audio_sha256' if 'audio_sha256' in usable else 'decoded_audio_sha256'
    if hash_column not in usable:
        hash_column = 'raw_sha256'
    conflicting = []
    if hash_column in usable:
        conflicts = usable.groupby(hash_column).diagnosis.nunique()
        conflicting = conflicts[conflicts > 1].index.tolist()
        usable = usable.loc[~usable[hash_column].isin(conflicting)]
        before = len(usable)
        usable = usable.drop_duplicates(hash_column)
        duplicates = before - len(usable)
    else:
        duplicates = 0
    usable['disease_stage_label'] = usable.diagnosis.map({0: '0', 1: 'Unknown', 3: '3'}).fillna('Unknown')
    usable['collection'] = 'original_audio'
    usable['target'] = usable.diagnosis.where(usable.diagnosis.isin([0, 1]))
    # Unknown identity stays missing; it is NOT replaced with a filename.
    usable['subject_id'] = usable.subject_id.replace('', np.nan).astype(object)
    known = usable.subject_id.notna()
    usable.loc[known, 'subject_id'] = usable.loc[known, 'subject_id'].astype(str).map(
        lambda value: value if value.startswith('KAGGLE_Voice:') else 'KAGGLE_Voice:' + value)
    features = [f for f in AUDIO_FEATURES if f in usable]
    usable[features] = usable[features].replace([np.inf, -np.inf], np.nan)
    q1, q3 = usable[features].quantile(.25), usable[features].quantile(.75)
    iqr = q3 - q1
    outliers = ((usable[features] < q1 - 1.5 * iqr) | (usable[features] > q3 + 1.5 * iqr)).sum()
    usable[features].describe().T
    usable.to_csv(RESULTS / 'source_specific_kaggle_voice.csv', index=False)
    return dict(raw_rows=len(audio), processed_rows=len(usable), participants=None if not known.any() else int(usable.subject_id.nunique()),
        missing_subject_ids=int(usable.subject_id.isna().sum()), feature_count=len(features),
        exact_audio_duplicates_removed=duplicates, conflicting_label_audio_hashes=conflicting,
        unusable_audio_excluded=int((~audio.extraction_status.isin(['ok', 'partial'])).sum()),
        missing_feature_values=usable[features].isna().sum().to_dict(),
        iqr_outlier_counts=outliers.to_dict(),
        retained_class_counts={str(k): int(v) for k,v in usable.diagnosis.value_counts().items()},
        outlier_action='No clipping; physiological and recording extremes retained.',
        modeling_eligible=bool(len(usable) and usable.subject_id.notna().all()),
        notes='No participant IDs supplied by Kaggle archive. No subject-level evaluation without a documented mapping; arbitrary filenames are not people.')


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    schemas, audit = {}, {}
    for name, item in load_raw().items():
        frame, audit[name] = clean_table(name, item)
        path = RESULTS / f'source_specific_{name}.csv'
        frame.to_csv(path, index=False)
        schemas[name] = dict(path=str(path.relative_to(ROOT)), features=item['features'], source=item['source'])
        print(name, audit[name]['processed_rows'], 'rows;', audit[name]['participants'], 'participants;', len(item['features']), 'features')
    audio_audit = clean_audio()
    if audio_audit:
        audit['kaggle_voice'] = audio_audit
        from extract_audio_features import AUDIO_FEATURES
        schemas['kaggle_voice'] = dict(path=str((RESULTS / 'source_specific_kaggle_voice.csv').relative_to(ROOT)), features=list(AUDIO_FEATURES), source='KAGGLE_Voice')
        print('kaggle_voice', audio_audit)
    (RESULTS / 'feature_schemas.json').write_text(json.dumps(schemas, indent=2))
    (RESULTS / 'cleaning_audit.json').write_text(json.dumps(audit, indent=2))
    lines = ['PREPROCESSING REPORT',
        'Raw DATA preserved. Only source byte extraction writes there; all derived CSVs are in RESULTS/_pipeline.',
        'Binary diagnosis: 0=Healthy, 1=confirmed PD, 3=RBD. Binary target excludes RBD.',
        'disease_stage_label: 0=Healthy, 1=Early-stage PD, 2=Established PD, 3=RBD, Unknown=PD stage unavailable.',
        'No established PD assigned without evidence; PD diagnosis alone is not evidence of established stage.',
        'No synthetic features/zero-filled schemas; missing values retained for training-fold median imputation.',
        'No clinical scales/medications/subject identifiers/demographics used as model features.',
        'Early speech measures retain task-specific source units; no conversion between incomparable tasks.',
        'No feature selection performed. Every retained acoustic predictor is listed in feature_schemas.json.',
        json.dumps(audit, indent=2)]
    text = '\n'.join(lines) + '\n'
    print(text)


if __name__ == '__main__':
    main()
