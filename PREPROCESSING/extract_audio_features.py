#!/usr/bin/env python3
"""Extract auditable acoustic features without inventing participant identities.

Run from any directory: ``python PREPROCESSING/extract_audio_features.py``.
Raw DATA is read-only. Mapping recording_id values must exactly match POSIX paths
relative to DATA/raw/KAGGLE_Voice (including the original folder names). Optional
mapping columns are recording_id, subject_id, and disease_stage; supplied subject
IDs are source-namespaced, never derived from audio filenames. Unknown identity
makes a recording ineligible for participant-disjoint modelling, not extraction.

No jitter/shimmer estimates are produced: frame-level F0 is not a sequence of
glottal periods, and frame RMS is not cycle-level amplitude. Outliers are retained;
nonfinite features are written as missing values. Duplicate recordings are audited
but never removed here. ``inspect_audio_inventory`` reads container metadata only,
without decoding samples, resampling, or computing acoustic features.
"""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_SOURCE = "KAGGLE_Voice"
SAMPLE_RATE = 22050
FRAME_LENGTH = 2048
HOP_LENGTH = 512
MIN_DURATION_SECONDS = 0.25
SILENCE_PEAK_THRESHOLD = 1e-7
PITCH_PROBABILITY_THRESHOLD = 0.8
AUDIO_SUFFIXES = {".wav", ".wave", ".flac", ".ogg", ".oga", ".mp3", ".m4a", ".aif", ".aiff", ".au"}
AUDIO_FEATURES = [
    f"mfcc_{index:02d}_{stat}" for index in range(1, 14) for stat in ("mean", "std")
] + [
    f"{name}_{stat}"
    for name in ("spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "spectral_flatness", "rms", "zcr")
    for stat in ("mean", "std")
] + ["f0_mean", "f0_std", "f0_median", "f0_min", "f0_max", "voiced_fraction"]
METADATA_COLUMNS = [
    "recording_id", "relative_path", "filename", "folder_label", "folder_labels",
    "dataset_source", "diagnosis", "target", "disease_stage", "subject_id",
    "participant_mapping_present", "original_channels", "original_sample_rate",
    "original_frames", "original_duration_seconds", "file_size_bytes", "raw_sha256",
    "decoded_audio_sha256", "processed_sample_rate", "extraction_status", "issues",
]


def _audio_paths(data_dir):
    if not data_dir.exists():
        return []
    return sorted(path for path in data_dir.rglob("*")
                  if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)


def _load_mapping(mapping_path):
    mapping = {}
    issues = []
    if not mapping_path.exists():
        return mapping, issues
    try:
        with mapping_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {"recording_id", "subject_id"}.issubset(reader.fieldnames or []):
                return {}, ["Participant mapping must contain recording_id and subject_id columns."]
            conflicts = set()
            for line, row in enumerate(reader, start=2):
                recording_id = (row.get("recording_id") or "").strip()
                subject_id = (row.get("subject_id") or "").strip()
                stage = (row.get("disease_stage") or "").strip()
                if not recording_id or not subject_id:
                    issues.append(f"Mapping row {line}: missing recording_id or subject_id; ignored.")
                    continue
                entry = {"subject_id": subject_id, "disease_stage": stage}
                if recording_id in mapping and mapping[recording_id] != entry:
                    conflicts.add(recording_id)
                mapping[recording_id] = entry
            for recording_id in sorted(conflicts):
                mapping.pop(recording_id, None)
                issues.append(f"Conflicting participant mapping for {recording_id}; identity left missing.")
    except (OSError, UnicodeError, csv.Error) as exc:
        return {}, [f"Could not read participant mapping: {type(exc).__name__}: {exc}"]
    return mapping, issues


def _metadata_row(path, data_dir, mapping):
    relative = path.relative_to(data_dir)
    recording_id = relative.as_posix()
    labels = relative.parts[:-1]
    diagnoses = {0 if label.strip().casefold() == "healthy" else 1
                 for label in labels
                 if label.strip().casefold() in {"healthy", "parkinson", "parkinsons", "pd"}}
    diagnosis = next(iter(diagnoses)) if len(diagnoses) == 1 else None
    entry = mapping.get(recording_id, {})
    subject = entry.get("subject_id", "")
    if subject and not subject.startswith(f"{DATASET_SOURCE}:"):
        subject = f"{DATASET_SOURCE}:{subject}"
    row = dict.fromkeys(METADATA_COLUMNS)
    row.update({
        "recording_id": recording_id, "relative_path": recording_id,
        "filename": path.name, "folder_label": path.parent.name,
        "folder_labels": "/".join(labels), "dataset_source": DATASET_SOURCE,
        "diagnosis": diagnosis, "target": diagnosis,
        "disease_stage": entry.get("disease_stage") or ("Not applicable" if diagnosis == 0 else "Unknown"),
        "subject_id": subject, "participant_mapping_present": bool(subject),
    })
    issues = []
    if not subject:
        issues.append("missing_participant_id: no usable explicit recording-to-subject mapping")
    if not diagnoses:
        issues.append("unknown_diagnosis_folder")
    elif len(diagnoses) > 1:
        issues.append("conflicting_diagnosis_folders")
    try:
        row["file_size_bytes"] = path.stat().st_size
    except OSError as exc:
        issues.append(f"file_stat_error: {exc}")
    try:
        import soundfile as sf
        info = sf.info(str(path))
        row.update({"original_channels": int(info.channels), "original_sample_rate": int(info.samplerate),
                    "original_frames": int(info.frames), "original_duration_seconds": float(info.duration)})
        if info.duration < MIN_DURATION_SECONDS:
            issues.append("short_clip: duration below 0.25 seconds")
        if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
            issues.append("invalid_audio_metadata")
    except Exception as exc:
        issues.append(f"audio_metadata_error: {type(exc).__name__}: {exc}")
    return row, issues


def inspect_audio_inventory(data_dir=None, mapping_path=None):
    """Return JSON-safe metadata without sample decoding or feature extraction.

    Silence and decoded-content duplicates cannot be determined by this function;
    their assessment is explicitly deferred to extraction. All paths are relative
    to the supplied audio directory. No files or directories are created.
    """
    data_dir = Path(data_dir) if data_dir is not None else ROOT / "DATA" / "raw" / DATASET_SOURCE
    mapping_path = Path(mapping_path) if mapping_path is not None else ROOT / "RESULTS" / "kaggle_subject_mapping.csv"
    mapping, mapping_issues = _load_mapping(mapping_path)
    files = []
    for path in _audio_paths(data_dir):
        row, issues = _metadata_row(path, data_dir, mapping)
        files.append({key: value for key, value in row.items()
                      if key not in {"raw_sha256", "decoded_audio_sha256", "processed_sample_rate", "extraction_status", "issues"}}
                     | {"issues": issues})
    return {
        "dataset_source": DATASET_SOURCE, "data_directory": str(data_dir),
        "directory_exists": data_dir.exists(), "audio_file_count": len(files),
        "folder_label_counts": dict(Counter(row["folder_label"] for row in files)),
        "channel_counts": dict(Counter(str(row["original_channels"]) for row in files)),
        "sample_rate_counts": dict(Counter(str(row["original_sample_rate"]) for row in files)),
        "missing_participant_ids": sum(not row["subject_id"] for row in files),
        "participant_mapping_path": str(mapping_path), "participant_mapping_exists": mapping_path.exists(),
        "mapping_issues": mapping_issues, "files": files,
        "inspection_limitations": ["Metadata only: silence and decoded-content duplicates not assessed.",
                                   "Filename numbering never establishes participant identity.",
                                   "Folder diagnosis does not establish disease stage."],
    }


def _raw_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summarize(values, name, features):
    import numpy as np
    features[f"{name}_mean"] = float(np.mean(values))
    features[f"{name}_std"] = float(np.std(values))


def _extract_features(samples, sample_rate, issues):
    import librosa
    import numpy as np

    mono = librosa.to_mono(samples.T)
    if sample_rate != SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    features = {}
    mfcc = librosa.feature.mfcc(y=mono, sr=SAMPLE_RATE, n_mfcc=13,
                               n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    for index, coefficients in enumerate(mfcc, start=1):
        _summarize(coefficients, f"mfcc_{index:02d}", features)
    magnitude = np.abs(librosa.stft(mono, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
    _summarize(librosa.feature.spectral_centroid(S=magnitude, sr=SAMPLE_RATE), "spectral_centroid", features)
    _summarize(librosa.feature.spectral_bandwidth(S=magnitude, sr=SAMPLE_RATE), "spectral_bandwidth", features)
    _summarize(librosa.feature.spectral_rolloff(S=magnitude, sr=SAMPLE_RATE, roll_percent=0.85), "spectral_rolloff", features)
    _summarize(librosa.feature.spectral_flatness(S=magnitude), "spectral_flatness", features)
    rms = librosa.feature.rms(y=mono, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    _summarize(rms, "rms", features)
    _summarize(librosa.feature.zero_crossing_rate(mono, frame_length=FRAME_LENGTH,
                                                 hop_length=HOP_LENGTH), "zcr", features)
    try:
        f0, voiced, probability = librosa.pyin(mono, fmin=50.0, fmax=600.0, sr=SAMPLE_RATE,
                                             frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        energy_threshold = max(1e-5, float(np.max(rms)) * 0.05)
        reliable = (voiced & np.isfinite(f0) & np.isfinite(probability)
                    & (probability >= PITCH_PROBABILITY_THRESHOLD) & (rms > energy_threshold))
        features["voiced_fraction"] = float(np.mean(reliable))
        if np.any(reliable):
            pitch = f0[reliable]
            _summarize(pitch, "f0", features)
            features.update(f0_median=float(np.median(pitch)), f0_min=float(np.min(pitch)),
                            f0_max=float(np.max(pitch)))
        else:
            issues.append("no_reliable_voiced_frames: F0 statistics left missing")
    except Exception as exc:
        issues.append(f"pitch_estimation_error: {type(exc).__name__}: {exc}")
    for name, value in features.items():
        if not np.isfinite(value):
            features[name] = None
            issues.append(f"nonfinite_feature: {name}; stored as missing, not zero")
    return features


def _extract_recording(path, row, issues):
    """Keep the row even when decoding or feature extraction fails."""
    try:
        row["raw_sha256"] = _raw_sha256(path)
        import numpy as np
        import soundfile as sf
        samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        frames, channels = samples.shape
        row.update(original_channels=int(channels), original_sample_rate=int(sample_rate),
                   original_frames=int(frames), original_duration_seconds=float(frames / sample_rate))
        digest = hashlib.sha256()
        digest.update(f"decoded-float32-le-v1:{sample_rate}:{frames}:{channels}\n".encode("ascii"))
        digest.update(memoryview(samples.astype("<f4", copy=False)).cast("B"))
        row["decoded_audio_sha256"] = digest.hexdigest()
        if not frames or not channels or sample_rate <= 0:
            row["extraction_status"] = "corrupt"
            issues.append("empty_or_invalid_decoded_audio")
            return
        if not np.all(np.isfinite(samples)):
            row["extraction_status"] = "corrupt"
            issues.append("nonfinite_audio_samples: features left missing")
            return
        if np.max(np.abs(samples)) <= SILENCE_PEAK_THRESHOLD:
            row["extraction_status"] = "silent"
            issues.append("silent_clip: peak amplitude <= 1e-7; features left missing")
            return
        if frames / sample_rate < MIN_DURATION_SECONDS:
            row["extraction_status"] = "short"
            if not any(issue.startswith("short_clip") for issue in issues):
                issues.append("short_clip: duration below 0.25 seconds")
            return
    except ImportError as exc:
        row["extraction_status"] = "dependency_error"
        issues.append(f"audio_dependency_error: {type(exc).__name__}: {exc}; audio validity not determined")
        return
    except Exception as exc:
        row["extraction_status"] = "corrupt"
        issues.append(f"audio_decode_error: {type(exc).__name__}: {exc}")
        return
    try:
        features = _extract_features(samples, sample_rate, issues)
        row.update(features)
        row["processed_sample_rate"] = SAMPLE_RATE
        row["extraction_status"] = "ok" if all(row.get(name) is not None for name in AUDIO_FEATURES) else "partial"
    except Exception as exc:
        row["extraction_status"] = "feature_error"
        issues.append(f"feature_extraction_error: {type(exc).__name__}: {exc}")


def _duplicate_groups(rows, column):
    groups = defaultdict(list)
    for row in rows:
        if row.get(column):
            groups[row[column]].append(row["recording_id"])
    return [{"sha256": digest, "recording_ids": recordings}
            for digest, recordings in sorted(groups.items()) if len(recordings) > 1]


def main():
    data_dir = ROOT / "DATA" / "raw" / DATASET_SOURCE
    results_dir = ROOT / "RESULTS" / "_pipeline"
    mapping_path = ROOT / "RESULTS" / "kaggle_subject_mapping.csv"
    mapping, mapping_issues = _load_mapping(mapping_path)
    rows = []
    logs = []
    for path in _audio_paths(data_dir):
        row, issues = _metadata_row(path, data_dir, mapping)
        _extract_recording(path, row, issues)
        row["issues"] = json.dumps(issues, ensure_ascii=False)
        rows.append(row)
        logs.append({"recording_id": row["recording_id"], "extraction_status": row["extraction_status"], "issues": issues})
    raw_duplicates = _duplicate_groups(rows, "raw_sha256")
    decoded_duplicates = _duplicate_groups(rows, "decoded_audio_sha256")
    matched_recordings = {row["recording_id"] for row in rows}
    report = {
        "dataset_source": DATASET_SOURCE,
        "input_directory": str(data_dir), "input_directory_exists": data_dir.exists(),
        "audio_file_count": len(rows), "status": "complete" if rows else "no_audio_files",
        "extraction_status_counts": dict(Counter(row["extraction_status"] for row in rows)),
        "metadata_columns": METADATA_COLUMNS, "feature_columns": AUDIO_FEATURES,
        "participant_mapping_path": str(mapping_path), "participant_mapping_exists": mapping_path.exists(),
        "mapping_issues": mapping_issues,
        "unmatched_mapping_recording_ids": sorted(set(mapping) - matched_recordings),
        "recordings_missing_participant_ids": sum(not row["subject_id"] for row in rows),
        "identity_policy": "Only explicit mapping IDs are used, prefixed KAGGLE_Voice: unless already prefixed. Filename numbers are not participants. Missing IDs exclude recordings from participant-disjoint modelling.",
        "diagnosis_policy": "Only Healthy=0 and Parkinson/Parkinsons/PD=1 ancestor folder labels (case-insensitive). Other or conflicting folders have missing diagnosis and target. RBD is not a positive label.",
        "stage_policy": "Optional mapping disease_stage is retained verbatim. Otherwise healthy is Not applicable, all others Unknown; confirmed PD is not assumed early-stage.",
        "processing": {"mono": True, "sample_rate_hz": SAMPLE_RATE, "resampling": "librosa.resample",
                       "frame_length": FRAME_LENGTH, "hop_length": HOP_LENGTH,
                       "mfcc_count": 13, "spectral_rolloff_fraction": 0.85,
                       "pitch_method": "librosa.pyin", "pitch_range_hz": [50, 600],
                       "pitch_voicing_rule": "pyin voiced flag, finite F0/probability, probability >= 0.8, frame RMS > max(1e-5, 5% of maximum frame RMS)",
                       "voiced_fraction_denominator": "all centered analysis frames; missing if pitch estimation fails",
                       "minimum_duration_seconds": MIN_DURATION_SECONDS,
                       "silence_peak_threshold": SILENCE_PEAK_THRESHOLD,
                       "short_or_silent_policy": "Retain metadata and hashes, leave acoustic features missing.",
                       "summary_std": "population standard deviation (ddof=0)",
                       "outlier_policy": "No clipping; nonfinite feature values become missing, never fabricated zero."},
        "jitter_shimmer": {"computed": False, "reliable_measurements_available": False,
                           "reason": "No validated glottal pulse extraction is performed. Frame-level F0 and RMS cannot provide pitch-period jitter or cycle-amplitude shimmer. These columns are intentionally omitted, not filled with zeros."},
        "source_integrity": {"raw_files_modified": False, "excluded_source_files": [],
                             "raw_hash": "SHA256 of original file bytes",
                             "decoded_hash": "SHA256 of ASCII decoded-float32-le-v1:sample_rate:frames:channels plus newline followed by original-rate, interleaved-channel little-endian float32 decoded samples; before mono conversion/resampling",
                             "raw_byte_duplicate_groups": raw_duplicates,
                             "decoded_content_duplicate_groups": decoded_duplicates,
                             "duplicate_policy": "All copies retained; these are evidence of recording duplication, not participant identity. Parent cleaning must prevent duplicate-content leakage."},
        "files": logs,
        "limitations": ["Folder labels are source labels, not independently verified clinical diagnoses.",
                        "No participant mapping is inferred from filename, row order, or acoustic similarity.",
                        "Pitch is constrained to 50–600 Hz and confidence-filtered; unvoiced or unreliable F0 is missing.",
                        "No verified early-stage labels are invented; no demographic or clinical scores become audio features.",
                        "No audio files means an empty metadata-and-feature-header CSV, never synthetic recordings."],
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "kaggle_audio_features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS + AUDIO_FEATURES)
        writer.writeheader()
        writer.writerows(rows)
    with (results_dir / "audio_extraction_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"Audio extraction: {len(rows)} recordings; {report['recordings_missing_participant_ids']} without participant IDs.")


if __name__ == "__main__":
    main()
