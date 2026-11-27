"""Prepare deterministic participant-disjoint splits; do not fit any transforms."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from inspect_data import ROOT, RESULTS

SEED = 42


def assign_splits(frame):
    labeled = frame.loc[frame.target.isin([0, 1])]
    if labeled.subject_id.isna().any():
        raise ValueError('Cannot split recordings with missing participant identity')
    subjects = labeled[['subject_id', 'target']].drop_duplicates()
    if subjects.subject_id.duplicated().any():
        raise ValueError('Conflicting diagnoses for a participant')
    counts = subjects.target.value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError('Insufficient labeled participants for classification')
    output = frame.copy()
    output['split'] = 'highrisk'
    if counts.min() < 10:
        output.loc[output.target.isin([0, 1]), 'split'] = 'cv'
        return output, 'nested_group_cv'
    train, remainder = train_test_split(subjects, test_size=.30, stratify=subjects.target, random_state=SEED)
    validation, test = train_test_split(remainder, test_size=.5, stratify=remainder.target, random_state=SEED)
    sets = {'train': set(train.subject_id), 'validation': set(validation.subject_id), 'test': set(test.subject_id)}
    for split, ids in sets.items():
        output.loc[output.subject_id.isin(ids), 'split'] = split
    for left in sets:
        for right in sets:
            if left != right and sets[left] & sets[right]:
                raise AssertionError('Participant leakage')
    if output.groupby('subject_id').split.nunique().gt(1).any():
        raise AssertionError('Repeated participant in multiple partitions')
    return output, 'holdout'


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    schemas = json.loads((RESULTS / 'feature_schemas.json').read_text())
    specs, summaries = [], {}
    experiments = [('early_stage', 'early_biomarkers'), ('multiple_speech', 'multiple_speech_training'), ('oxford_voice', 'oxford_voice')]
    if 'kaggle_voice' in schemas:
        audio = pd.read_csv(ROOT / schemas['kaggle_voice']['path'])
        if not audio.empty and audio.subject_id.notna().all():
            experiments.append(('kaggle_voice', 'kaggle_voice'))
        else:
            summaries['kaggle_voice'] = dict(status='excluded_from_modeling',
                reason='No documented recording-to-participant mapping; a recording split could leak speakers.',
                available_processed_audio_rows=len(audio), missing_subject_ids=int(audio.subject_id.isna().sum()))
    for experiment, source in experiments:
        if source not in schemas:
            summaries[experiment] = dict(status='unavailable_source')
            continue
        frame = pd.read_csv(ROOT / schemas[source]['path'])
        split, evaluation = assign_splits(frame)
        if source == 'multiple_speech_training' and 'multiple_speech_external' in schemas:
            external = pd.read_csv(ROOT / schemas['multiple_speech_external']['path'])
            external['split'] = 'external'
            if set(split.subject_id) & set(external.subject_id):
                raise AssertionError('Independent collection ID collision')
            split = pd.concat([split, external], ignore_index=True)
        features = schemas[source]['features']
        # Never retain a wholly unavailable predictor or invent arbitrary fill values.
        training = split.loc[split.split.isin(['train', 'cv'])]
        retained = [column for column in features if training[column].notna().any()]
        if not retained:
            raise ValueError('No usable training features for ' + experiment)
        path = RESULTS / f'experiment_{experiment}.csv'
        split.to_csv(path, index=False)
        spec = dict(name=experiment, path=str(path.relative_to(ROOT)), features=retained,
            evaluation=evaluation, target='target', group='subject_id', selection_eligible=experiment == 'early_stage')
        specs.append(spec)
        part_counts = {}
        for part, subset in split.groupby('split'):
            subjects = subset[['subject_id', 'diagnosis']].drop_duplicates()
            part_counts[part] = dict(records=len(subset), participants=subset.subject_id.nunique(),
                participant_classes={str(k): int(v) for k,v in subjects.diagnosis.value_counts().items()})
        summaries[experiment] = dict(evaluation=evaluation, feature_count=len(retained), partitions=part_counts,
            limitations='Only eight healthy Oxford participants: use four-fold outer / three-fold inner grouped CV instead of ~1 healthy person per holdout.' if evaluation == 'nested_group_cv' else '70/15/15 by participants, stratified diagnosis. Small validation/test counts yield unstable point estimates.')
        print(experiment, evaluation, json.dumps(part_counts))
    (RESULTS / 'experiment_specs.json').write_text(json.dumps(specs, indent=2))
    (RESULTS / 'split_summary.json').write_text(json.dumps(summaries, indent=2))
    text = '''SUBJECT-LEVEL SPLITTING REPORT
Seed 42. Holdout proportions are 70%/15%/15% of unique participants, not rows. Stratify binary diagnosis at the participant table; map all repeated recordings back to their participant partition. All six classifiers use identical partitions and grouped inner folds.
If a source has fewer than 10 participants in either binary class, use nested grouped CV rather than an unreliable three-way split. Oxford has eight healthy participants; four-fold outer CV supplies OOF estimates, three-fold inner CV chooses parameters. Outer-fold models are retained in the model bundle; a full-data deployment fit has no independent test claim.
UCI301 original train collection is re-split into training/validation/testing participants. Its independent PD-only test collection retains split=external and a distinct original-collection ID namespace. Do not use it for threshold selection, fitting or tuning. Only sensitivity is identifiable there.
RBD split=highrisk, never used for training, hyperparameter selection, validation/test accuracy or PD classification labels. Exploratory scores are not diagnoses or longitudinal conversion predictions.
No imputation/scaling/feature selection occurs before splitting. Model pipelines fit median imputation and StandardScaler inside each inner training fold; only chosen training-only pipelines score validation/test. Features wholly unavailable in training are not retained. No feature selection is used.
Kaggle is ineligible for subject-disjoint evaluation without an evidence-backed participant mapping. Place that mapping in RESULTS/kaggle_subject_mapping.csv (recording_id,subject_id; optional documented disease_stage) and rerun extraction/cleaning/merging/splitting/training. A filename-derived mapping is not acceptable. Subject mappings do not establish independence from the UCI cohorts.
A common-feature cross-source model is not available: the source-specific acoustic measurements have no verified cross-source equivalence. Separate within-source performance is not evidence of cross-source transfer or early-disease performance in unknown-stage cohorts.
'''
    print(text + '\n' + json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
