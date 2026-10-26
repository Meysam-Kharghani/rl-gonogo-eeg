#!/usr/bin/env python3
# FORMALIZED HISTORICAL SOURCE SNAPSHOT
# Evidence status: STOPPED_BY_DESIGN; NOT_PART_OF_FINAL_EVIDENCE
# This file preserves a historical project stage with public-facing terminology standardized for release.
# The SHA-256 of the original executed/development source is recorded in the corresponding provenance registry.

"""Re-extract trial EEG features with the direct code-9 response marker.

This imports the repository's validated extractor so channel selection, windows,
baselines, condition-event mapping, and metadata handling remain identical. Only
the response lock changes:

  direct_marker: use the nearest code-9 event within --max-error-ms;
  hybrid: use that marker when available, otherwise target onset + behavioral RT.

Run this only after the marker-sensitivity audit, and compare its response-locked
results with the original metadata-locked extraction. Cue/target features are also
written for schema compatibility but should be numerically unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from common_utils import ensure_dir, import_module  # noqa: E402


def process_subject(src, subject: str, set_path: Path, meta_sub: pd.DataFrame,
                    target_delay: float, mode: str, max_error_ms: float):
    raw, ev = src.read_raw_and_events(set_path)
    cond = ev[ev["code"].isin(src.CONDITION_CODES)].reset_index(drop=True)
    response_events = ev[ev["code"] == src.RESPONSE_CODE].reset_index(drop=True)
    response_onsets = response_events["onset_sec"].astype(float).to_numpy() if not response_events.empty else np.array([])
    channels = {name: src.get_channel(raw, name) for name in ["Fz", "Cz", "Pz", "C3", "C4"]}
    rows, qc = [], []
    n = min(len(meta_sub), len(cond))
    for i in range(n):
        m = meta_sub.iloc[i]
        cue_onset = float(cond.iloc[i]["onset_sec"])
        target_onset = cue_onset + target_delay
        responded = int(m["response_present_numeric"]) == 1 and pd.notna(m["rt_sec"])
        metadata_onset = target_onset + float(m["rt_sec"]) if responded else np.nan
        marker, err = src.nearest_response_marker(response_onsets, metadata_onset, max_abs_diff_sec=max_error_ms / 1000.0)
        if not responded:
            chosen = None; source = "no_response"
        elif np.isfinite(marker):
            chosen = marker; source = "direct_marker"
        elif mode == "hybrid":
            chosen = metadata_onset; source = "metadata_fallback"
        else:
            chosen = None; source = "missing_marker"
        base = {
            "subject_id": subject, "subject_id_norm": src.normalize_subject_id(subject),
            "group": "high" if subject.startswith("H") else "low", "trial_ordinal": i + 1,
            "trial_index": m["trial_index"], "trial_index_numeric": pd.to_numeric(m["trial_index"], errors="coerce"),
            "cue_type": src.normalize_label(m["cue_type"]), "trial_class": src.normalize_label(m["trial_class"]),
            "response_type": src.normalize_label(m.get("response_type", "")),
            "response_present": int(m["response_present_numeric"]),
            "rt_response_from_target_ms": float(m["rt_response_from_target_ms"]) if pd.notna(m["rt_response_from_target_ms"]) else np.nan,
            "eeg_condition_code": str(cond.iloc[i]["code"]), "cue_onset_sec": cue_onset,
            "target_onset_sec": target_onset, "response_onset_from_metadata_sec": metadata_onset,
            "response_marker_onset_sec": marker, "response_marker_error_ms": err,
            "response_lock_onset_sec": chosen if chosen is not None else np.nan,
            "response_lock_source": source,
        }
        feat = src.extract_trial_features(raw, m, cue_onset, chosen, channels, target_delay)
        rows.append({**base, **feat})
        qc.append({"subject_id": subject, "trial_ordinal": i + 1, "responded": responded,
                   "metadata_onset_sec": metadata_onset, "marker_onset_sec": marker,
                   "marker_error_ms": err, "response_lock_source": source})
    summary = {"subject_id": subject, "n_trials": n, "n_response_events": len(response_events),
               "n_direct_marker": sum(x["response_lock_source"] == "direct_marker" for x in qc),
               "n_metadata_fallback": sum(x["response_lock_source"] == "metadata_fallback" for x in qc),
               "n_missing_marker": sum(x["response_lock_source"] == "missing_marker" for x in qc),
               "median_abs_marker_error_ms": float(np.nanmedian(np.abs([x["marker_error_ms"] for x in qc])))}
    return pd.DataFrame(rows), pd.DataFrame(qc), summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-extractor", required=True, help="Repository scripts/03_extract_trial_level_eeg_features.py")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--eeg-root", required=True)
    ap.add_argument("--latents", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["direct_marker", "hybrid"], default="direct_marker")
    ap.add_argument("--max-error-ms", type=float, default=20.0)
    ap.add_argument("--target-delay-sec", type=float, default=0.5)
    args = ap.parse_args()

    out = ensure_dir(args.out)
    src = import_module(args.source_extractor, "source_eeg_extractor")
    metadata = src.load_metadata(Path(args.metadata))
    set_by_subject = src.choose_preferred_set_files(src.find_set_files(Path(args.eeg_root)))
    all_f, all_q, summaries = [], [], []
    for subject in sorted(metadata["subject_id_norm"].dropna().unique()):
        if subject not in set_by_subject:
            summaries.append({"subject_id": subject, "error": "missing_eeg_set_file"}); continue
        try:
            f, q, s = process_subject(src, subject, set_by_subject[subject],
                                      metadata[metadata["subject_id_norm"].eq(subject)].copy(),
                                      args.target_delay_sec, args.mode, args.max_error_ms)
            all_f.append(f); all_q.append(q); summaries.append(s)
            print(subject, len(f), flush=True)
        except Exception as exc:
            summaries.append({"subject_id": subject, "error": f"{type(exc).__name__}: {exc}"})
    features = pd.concat(all_f, ignore_index=True) if all_f else pd.DataFrame()
    qc = pd.concat(all_q, ignore_index=True) if all_q else pd.DataFrame()
    if args.latents and not features.empty:
        lat = src.load_latents(Path(args.latents))
        features = features.merge(lat, on=["subject_id_norm", "trial_index_numeric"], how="left", suffixes=("", "_latent"))
    features.to_csv(out / "eeg_trial_neural_features_direct_marker.csv", index=False)
    qc.to_csv(out / "eeg_direct_marker_lock_qc.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "eeg_direct_marker_by_subject.csv", index=False)
    lines = ["# Direct-marker extraction", "", f"Mode: {args.mode}", f"Maximum marker discrepancy: {args.max_error_ms} ms", "",
             "Compare all primary response-locked effects and decoding results with the metadata-locked table. Cue/target results are not a new analysis."]
    (out / "DIRECT_MARKER_EXTRACTION_README.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
