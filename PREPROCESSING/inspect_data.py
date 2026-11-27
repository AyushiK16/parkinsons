"""Inspect original datasets separately; also provide their documented parsers."""
from pathlib import Path
import contextlib
import io
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'RESULTS' / '_pipeline'
MULTIPLE_FEATURES = [
    'jitter_local_percent', 'jitter_absolute_seconds', 'jitter_rap_percent',
    'jitter_ppq5_percent', 'jitter_ddp_percent', 'shimmer_local_percent',
    'shimmer_db', 'shimmer_apq3_percent', 'shimmer_apq5_percent',
    'shimmer_apq11_percent', 'shimmer_dda_percent', 'autocorrelation',
    'noise_to_harmonics', 'harmonics_to_noise_db', 'pitch_median_hz',
    'pitch_mean_hz', 'pitch_std_hz', 'pitch_min_hz', 'pitch_max_hz',
    'number_of_pulses', 'number_of_periods', 'period_mean_seconds',
    'period_std_seconds', 'unvoiced_frames_percent', 'number_of_voice_breaks',
    'voice_breaks_percent',
]
EARLY_MEASURES = [
    'entropy', 'timing_rate_per_minute', 'timing_acceleration_per_minute_squared',
    'pause_duration_ms', 'voiced_duration_ms', 'gaping_per_minute',
    'unvoiced_stop_duration_ms', 'fricative_decay_source_units',
    'respiration_loudness_db', 'pauses_per_respiration',
    'respiration_rate_per_minute', 'respiratory_exchange_latency_ms',
]


def standard_name(name):
    return re.sub(r'[^a-z0-9]+', '_', str(name).lower()).strip('_')


def load_raw():
    """Return observed tables, not UCI's occasionally inconsistent headline counts."""
    tables = {}
    early_path = ROOT / 'DATA/raw/UCI_Early_Biomarkers/dataset.csv'
    if early_path.exists():
        frame = pd.read_csv(early_path, header=[0, 1], na_values=['-', '?'])
        original = [' | '.join(col) for col in frame.columns]
        columns = ['participant_code'] + [standard_name(c[1]) for c in frame.columns[1:41]]
        features = [f'{task}_{measure}' for task in ('reading', 'monologue') for measure in EARLY_MEASURES]
        frame.columns = columns + features
        tables['early_biomarkers'] = dict(frame=frame, original_columns=original,
            features=features, id_column='participant_code', target_column='participant_code prefix',
            source='UCI_Early_Biomarkers', collection='single_session',
            notes='130 subject-level summaries; reading and monologue are separate feature blocks, not independent people. Clinical/medication/demographic fields are excluded from model inputs. Literal - and ? missing-value tokens are parsed as missing, despite UCI headline metadata saying no missing values. Monologue fricative decay unit is printed ?/min rather than promile/min: retain source-specific unit ambiguity, do not merge the two tasks.')
    folder = ROOT / 'DATA/raw/UCI_Multiple_Speech'
    for collection, filename in [('training', 'train_data.txt'), ('external', 'test_data.txt')]:
        if not (folder / filename).exists():
            continue
        columns = ['raw_subject_id'] + MULTIPLE_FEATURES
        if collection == 'training':
            columns += ['updrs']
        columns += ['class']
        frame = pd.read_csv(folder / filename, header=None)
        if frame.shape[1] != len(columns):
            raise ValueError(f'{filename}: expected {len(columns)} columns, found {frame.shape[1]}')
        frame.columns = columns
        tables[f'multiple_speech_{collection}'] = dict(frame=frame, original_columns=columns,
            features=MULTIPLE_FEATURES, id_column='raw_subject_id', target_column='class',
            source='UCI_Multiple_Speech', collection=collection,
            notes='UCI provides no recording IDs: use original filename and 1-based data-row locator, not an invented participant ID. Training has 26 ordered speech tasks/subject; external has six sustained vowel recordings/subject and PD only. Training and external subject-number namespaces are distinct, as documented independent collections. UPDRS excluded from predictors.')
    oxford_path = ROOT / 'DATA/raw/UCI_Oxford_Voice/parkinsons.data'
    if oxford_path.exists():
        frame = pd.read_csv(oxford_path)
        original = list(frame.columns)
        frame.columns = [standard_name(c) for c in frame.columns]
        tables['oxford_voice'] = dict(frame=frame, original_columns=original,
            features=[c for c in frame if c not in ('name', 'status')], id_column='name',
            target_column='status', source='UCI_Oxford_Voice', collection='original',
            notes='Participant identification is recording name without final underscore recording number. Bundled telemonitoring files in the UCI archive are a different study, not added as a fifth dataset. MDVP jitter(%) values are retained as published; cross-source percent/fraction conversion is not inferred from similar names.')
    return tables


def subject_codes(name, item):
    frame = item['frame']
    if name == 'early_biomarkers':
        return frame.participant_code.astype(str)
    if name == 'oxford_voice':
        return frame['name'].str.replace(r'_\d+$', '', regex=True)
    return frame.raw_subject_id.astype(str)


def raw_labels(name, frame):
    if name == 'early_biomarkers':
        return frame.participant_code.str.extract(r'^([A-Z]+)')[0]
    return frame['status' if name == 'oxford_voice' else 'class']


def describe_table(name, item):
    frame = item['frame']
    ids = subject_codes(name, item)
    labels = raw_labels(name, frame)
    print('\n' + '=' * 80 + '\n' + name)
    print('Dimensions:', frame.shape)
    print('ALL ORIGINAL COLUMNS:', json.dumps(item['original_columns'], ensure_ascii=False))
    print('STANDARDIZED COLUMNS:', json.dumps(list(frame.columns)))
    print('Target:', item['target_column'], '; participant field:', item['id_column'])
    print('Recording identifiers:', 'name' if name == 'oxford_voice' else 'subject summary' if name == 'early_biomarkers' else 'source filename + row locator (not supplied by source)')
    print('Numerical columns:', list(frame.select_dtypes(include='number')))
    print('Categorical columns:', list(frame.select_dtypes(exclude='number')))
    print('Missing by column:', frame.isna().sum().to_dict())
    print('Exact duplicate records:', int(frame.duplicated().sum()))
    print('Class distribution (rows):', labels.value_counts(dropna=False).to_dict())
    subjects = pd.DataFrame({'subject': ids, 'label': labels}).drop_duplicates()
    if subjects.subject.duplicated().any():
        raise ValueError('Conflicting subject labels in ' + name)
    print('Participants:', ids.nunique(), '; class distribution (subjects):', subjects.label.value_counts().to_dict())
    print('Records per participant:', ids.value_counts().describe().to_dict())
    print('Repeated-participant record counts:', ids.value_counts()[lambda s: s > 1].to_dict())
    print('Basic statistical summaries:\n', frame.describe(include='all').to_string())
    print('Notes:', item['notes'])
    return dict(rows=len(frame), columns=len(frame.columns), participants=ids.nunique(),
        class_rows={str(k):int(v) for k,v in labels.value_counts().items()},
        class_subjects={str(k):int(v) for k,v in subjects.label.value_counts().items()},
        exact_duplicates=int(frame.duplicated().sum()), numerical_features=len(item['features']))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = io.StringIO()
    summary = {}
    with contextlib.redirect_stdout(report):
        print('RAW DATA INSPECTION — no raw files changed. Counts calculated from downloaded bytes.')
        for name, item in load_raw().items():
            summary[name] = describe_table(name, item)
        from extract_audio_features import inspect_audio_inventory
        audio = inspect_audio_inventory()
        summary['kaggle_voice'] = audio
        print('\nKAGGLE_Voice: original audio rather than a tabular feature file.')
        print(json.dumps(audio, indent=2))
        print('Audio features are extracted only after this inspection. No original tabular target/columns exist; labels come from original folders. Subject counts and repeated-subject counts cannot be determined without documented participant mapping.')
    text = report.getvalue()
    (RESULTS / 'dataset_inventory.json').write_text(json.dumps(summary, indent=2))
    print(text)


if __name__ == '__main__':
    main()
