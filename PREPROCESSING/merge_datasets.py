"""Merge compatible provenance/labels, not unverified acoustic measurements."""
import json
from itertools import combinations
import pandas as pd
from inspect_data import ROOT, RESULTS
from clean_data import METADATA


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    schemas = json.loads((RESULTS / 'feature_schemas.json').read_text())
    tables = {name: pd.read_csv(ROOT / item['path']) for name, item in schemas.items()}
    registry = []
    for name, frame in tables.items():
        data = frame[METADATA].copy()
        data['source_specific_dataset'] = schemas[name]['path']
        registry.append(data)
    if not registry:
        raise ValueError('No source-specific data available: run cleaning first')
    master = pd.concat(registry, ignore_index=True)
    if master.duplicated(['dataset_source', 'recording_id']).any():
        raise ValueError('Duplicate provenance keys: investigate before merging')
    # Retain the subset calculations; the source-specific tables already store these rows.
    early = tables.get('early_biomarkers')
    if early is not None:
        early_stage = early.loc[early.diagnosis.isin([0, 1])]
        early_patients = early.loc[early.diagnosis.eq(1)]
    healthy = master.loc[master.diagnosis.eq(0)]
    matrix = []
    for left, right in combinations(schemas, 2):
        overlap = sorted(set(schemas[left]['features']) & set(schemas[right]['features']))
        same_source = schemas[left]['source'] == schemas[right]['source']
        matrix.append(dict(left=left, right=right, exact_standardized_name_overlap=';'.join(overlap),
            retained_common_features=';'.join(overlap) if same_source else '',
            decision='Same UCI301 protocol/features; external PD-only collection kept separate for evaluation' if same_source else 'Separate: no verified equivalent feature extraction and units across these sources'))
    compatibility = pd.DataFrame(matrix)
    report = '''HARMONIZATION AND MERGE REPORT
Decision: harmonize provenance and label semantics; retain acoustic predictors in source-specific tables.
The in-memory metadata registry is NOT a pooled acoustic training matrix. Its source_specific_dataset column points to the corresponding features. No heterogeneous columns are zero-filled or imputed across sources. No pooled performance is reported.

Observed schemas:
- UCI392: 24 connected-speech timing/respiration descriptors, separately measured during reading and monologue. No sustained-vowel jitter/shimmer/F0 features. Both tasks describe the SAME participant and remain in one subject-level row.
- UCI301: 26 Praat-style acoustic descriptors across 26 heterogeneous tasks/participant. Published columns include local jitter/shimmer percentages, autocorrelation, NTH/HTN, pitch and period statistics. The independent test collection uses only sustained vowels and has no controls; it is not pooled with training.
- UCI174: 22 MDVP/dynamical descriptors. Jitter(%) values are published on a different numeric scale from UCI301's local jitter; common-looking names do not establish equivalent normalization or extraction. NTH/HTN and NHR/HNR are not silently equated. Even mean/min/max pitch candidates lack documented cross-tool measurement settings and task equivalence in the supplied dictionaries. They remain source-specific rather than claiming validated measurement harmonization.
- Kaggle: 44 newly extracted librosa descriptors (MFCC summaries, spectral/energy/zero-crossing measures, confidence-filtered pYIN F0). Confidence-filtered F0 is not assumed equivalent to MDVP/Praat whole-record pitch; frame F0/RMS is not cycle jitter/shimmer. No participant mapping is present.

Successful feature alignment: UCI301 training and independent external collection have the SAME documented 26-feature ordering/definitions. They are concatenated only in the UCI301 experiment CSV with distinct collection namespaces; the external collection is excluded from fit/tuning/selection.
Successful cross-source alignment: subject provenance, binary diagnosis, stage semantics and recording locators only. No verified cross-source acoustic model is claimed. Across-source comparison therefore means separate within-source performance, NOT validated transfer. A pooled acoustic model / leave-one-source-out evaluation is intentionally unavailable rather than invented.

Candidate features and units are preserved in feature_schemas.json and original UCI metadata JSON. Fricative decay is printed promile/min for reading but ?/min for monologue; neither unit conversion nor task exchangeability is inferred. Other early units (ms, per minute, dB) remain explicit in feature names. Oxford source units retained verbatim; UCI301 relative measurements remain percentages, absolute jitter/period in seconds, pitch in Hz, noise harmonic levels in dB. No cross-source scaling replaces unit verification.

Labels: diagnosis 0=Healthy,1=confirmed PD,3=RBD; target 0/1 excludes RBD. disease_stage_label 0=Healthy,1=Early-stage PD,2=Established PD,3=RBD,Unknown=PD stage unavailable. No source provides justified established-stage labels for this project. Unknown-stage PD is not relabeled established or early.

The UCI392 source-specific table retains early PD, healthy subjects and RBD rows. The binary experiment contains early PD plus healthy subjects; RBD rows receive explicitly exploratory scores, not supervised labels.
Healthy controls remain in their source-specific tables rather than a pooled acoustic matrix. Kaggle participant counts and overlap with UCI studies are unknown. Namespacing avoids ID collisions but is not evidence that different studies have no overlapping people; no cross-source models are trained.
Raw observations are never deduplicated across studies merely because numeric feature values resemble one another. Exact within-table duplicates and byte/decoded-identical audio are handled in cleaning with audit evidence.

Sources:
https://archive.ics.uci.edu/dataset/392/early+biomarkers+of+parkinson+s+disease+based+on+natural+connected+speech
https://archive.ics.uci.edu/dataset/301/parkinson+speech+dataset+with+multiple+types+of+sound+recordings
https://archive.ics.uci.edu/dataset/174/parkinsons
https://www.kaggle.com/datasets/ucimachinelearning/parkinsons-voice-dataset
'''
    print(report)
    print('Master provenance registry:', len(master), 'records; no unverifiable cross-source acoustic merge.')
    print('Healthy registry:', len(healthy), 'records.')


if __name__ == '__main__':
    main()
