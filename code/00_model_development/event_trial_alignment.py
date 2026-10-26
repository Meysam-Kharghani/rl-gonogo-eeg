#!/usr/bin/env python3
# FORMALIZED HISTORICAL SOURCE SNAPSHOT
# Evidence status: HISTORICAL_OR_SUPERSEDED_DEVELOPMENT_EVIDENCE
# This file preserves a historical project stage with public-facing terminology standardized for release.
# The SHA-256 of the original executed/development source is recorded in the corresponding provenance registry.

"""
EEG event-code and trial-alignment audit for the EEG-computational study.

This script is non-destructive. It verifies whether the EEGLAB event sequence
matches the trial-level metadata table and evaluates whether the condition
events are closer to cue onset or target onset by comparing response-marker
latencies with behavioral RTs.

Expected default condition-code mapping, inferred from per-subject event counts:
1 = up / nogo
2 = up / go
3 = bi / nogo
4 = bi / go
5 = down / nogo
6 = down / go
9 = response marker candidate

Outputs:
- eeg_trial_alignment_by_subject.csv
- eeg_trial_alignment_rows.csv
- eeg_response_alignment_rows.csv
- eeg_event_trial_alignment_report.md

Run from controlled_inputs:
python code/final/run_eeg_event_trial_alignment_v1.py \
  --base . \
  --metadata data/raw/csv/ALL_SUBJECTS_metadata_features_merged_ORDERED.xlsx \
  --eeg_root "data/raw data/Groups_eventlistdefined" \
  --out results/eeg_event_trial_alignment_v1
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


DEFAULT_CONDITION_MAP = {
    "1": {"cue_type": "up", "trial_class": "nogo"},
    "2": {"cue_type": "up", "trial_class": "go"},
    "3": {"cue_type": "bi", "trial_class": "nogo"},
    "4": {"cue_type": "bi", "trial_class": "go"},
    "5": {"cue_type": "down", "trial_class": "nogo"},
    "6": {"cue_type": "down", "trial_class": "go"},
}


def normalize_subject_id(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    m = re.search(r"([HLhl])\s*0*([0-9]+)", s)
    if m:
        return f"{m.group(1).upper()}{int(m.group(2))}"
    return s.upper()


def normalize_label(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def parse_event_code(desc: object) -> str:
    """
    Parse MNE annotation descriptions into simple numeric event code strings.
    Works for descriptions like '1', 'S  1', 'boundary', etc.
    """
    s = str(desc).strip()
    if re.fullmatch(r"\d+", s):
        return str(int(s))
    m = re.search(r"(?<!\d)(\d+)(?!\d)", s)
    if m:
        return str(int(m.group(1)))
    return s


def infer_subject_group_from_path(path: Path) -> Tuple[str, str]:
    parts = [p for p in path.parts]
    subject = ""
    for part in reversed(parts):
        if re.fullmatch(r"[HLhl]\d+", part.strip()):
            subject = normalize_subject_id(part)
            break
    if not subject:
        m = re.search(r"([HLhl]\d+)", path.name)
        if m:
            subject = normalize_subject_id(m.group(1))
    group = ""
    lower_parts = [p.lower() for p in parts]
    if any("high" in p for p in lower_parts) or subject.startswith("H"):
        group = "high"
    elif any("low" in p for p in lower_parts) or subject.startswith("L"):
        group = "low"
    return subject, group


def find_set_files(eeg_root: Path) -> List[Path]:
    return sorted(eeg_root.rglob("*.set"))


def choose_preferred_set_files(files: List[Path]) -> Dict[str, Path]:
    by_subj: Dict[str, List[Path]] = defaultdict(list)
    for p in files:
        subject, _ = infer_subject_group_from_path(p)
        if subject:
            by_subj[subject].append(p)

    selected: Dict[str, Path] = {}
    for subject, candidates in sorted(by_subj.items()):
        def rank(p: Path) -> Tuple[int, str]:
            name = p.name.lower()
            if name.endswith("_ch_filt_ref_elist.set"):
                return (0, str(p))
            if "ica" not in name and "pruned" not in name:
                return (1, str(p))
            return (2, str(p))
        selected[subject] = sorted(candidates, key=rank)[0]
    return selected


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = ["subject_id", "trial_index", "trial_class", "cue_type", "response_present", "rt_response_from_target_ms"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata table is missing required columns: {missing}")

    out = df.copy()
    out["subject_id_norm"] = out["subject_id"].map(normalize_subject_id)
    out["cue_type_norm"] = out["cue_type"].map(normalize_label)
    out["trial_class_norm"] = out["trial_class"].map(normalize_label)
    out["trial_index_numeric"] = pd.to_numeric(out["trial_index"], errors="coerce")
    out["response_present_numeric"] = pd.to_numeric(out["response_present"], errors="coerce").fillna(0).astype(int)
    out["rt_sec"] = pd.to_numeric(out["rt_response_from_target_ms"], errors="coerce") / 1000.0
    out = out.sort_values(["subject_id_norm", "trial_index_numeric"]).reset_index(drop=True)
    return out


def load_events_from_set(path: Path):
    import mne  # local import so dependency errors are clear at runtime

    raw = mne.io.read_raw_eeglab(str(path), preload=False, verbose="ERROR")
    rows = []
    for onset, duration, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        rows.append({
            "onset_sec": float(onset),
            "duration_sec": float(duration),
            "description": str(desc),
            "code": parse_event_code(desc),
        })
    ev = pd.DataFrame(rows).sort_values("onset_sec").reset_index(drop=True)
    return raw, ev


def trial_alignment_for_subject(
    subject: str,
    set_path: Path,
    meta_sub: pd.DataFrame,
    condition_map: Dict[str, dict],
    response_code: str,
    target_delay_sec: float,
    response_window_sec: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw, ev = load_events_from_set(set_path)

    condition_codes = set(condition_map.keys())
    cond = ev[ev["code"].isin(condition_codes)].copy().reset_index(drop=True)
    resp = ev[ev["code"] == response_code].copy().reset_index(drop=True)

    n_meta = len(meta_sub)
    n_cond = len(cond)
    n_compare = min(n_meta, n_cond)

    align_rows = []
    for i in range(n_compare):
        m = meta_sub.iloc[i]
        c = cond.iloc[i]
        code = str(c["code"])
        expected = condition_map.get(code, {})
        eeg_cue = expected.get("cue_type", "")
        eeg_trial_class = expected.get("trial_class", "")

        cue_match = normalize_label(m["cue_type"]) == eeg_cue
        class_match = normalize_label(m["trial_class"]) == eeg_trial_class
        align_rows.append({
            "subject_id": subject,
            "trial_ordinal": i + 1,
            "metadata_trial_index": m["trial_index"],
            "eeg_condition_code": code,
            "eeg_condition_onset_sec": c["onset_sec"],
            "metadata_cue_type": normalize_label(m["cue_type"]),
            "eeg_cue_type": eeg_cue,
            "cue_match": bool(cue_match),
            "metadata_trial_class": normalize_label(m["trial_class"]),
            "eeg_trial_class": eeg_trial_class,
            "trial_class_match": bool(class_match),
            "both_match": bool(cue_match and class_match),
        })

    # Response-marker timing check.
    response_rows = []
    resp_onsets = list(resp["onset_sec"].astype(float).values)
    cond_onsets = list(cond["onset_sec"].astype(float).values)

    for i in range(n_compare):
        m = meta_sub.iloc[i]
        c_onset = float(cond_onsets[i])
        next_cond = float(cond_onsets[i + 1]) if i + 1 < len(cond_onsets) else c_onset + response_window_sec
        upper = min(next_cond, c_onset + response_window_sec)
        candidates = [x for x in resp_onsets if c_onset <= x <= upper]

        rt_sec = m["rt_sec"]
        responded = int(m["response_present_numeric"]) == 1 and pd.notna(rt_sec)

        best_target = np.nan
        best_cue = np.nan
        target_diff = np.nan
        cue_diff = np.nan
        n_candidates = len(candidates)

        if responded and candidates:
            expected_target = float(rt_sec)
            expected_cue = float(target_delay_sec + rt_sec)

            # Match response marker under two timing hypotheses.
            diffs_target = [(abs((x - c_onset) - expected_target), x) for x in candidates]
            diffs_cue = [(abs((x - c_onset) - expected_cue), x) for x in candidates]
            target_diff, best_target = min(diffs_target, key=lambda z: z[0])
            cue_diff, best_cue = min(diffs_cue, key=lambda z: z[0])

        response_rows.append({
            "subject_id": subject,
            "trial_ordinal": i + 1,
            "metadata_trial_index": m["trial_index"],
            "responded": bool(responded),
            "rt_sec": float(rt_sec) if pd.notna(rt_sec) else np.nan,
            "eeg_condition_onset_sec": c_onset,
            "n_response_candidates_after_condition": n_candidates,
            "best_response_onset_target_hypothesis_sec": best_target,
            "abs_error_target_hypothesis_ms": target_diff * 1000.0 if pd.notna(target_diff) else np.nan,
            "best_response_onset_cue_hypothesis_sec": best_cue,
            "abs_error_cue_hypothesis_ms": cue_diff * 1000.0 if pd.notna(cue_diff) else np.nan,
        })

    align_df = pd.DataFrame(align_rows)
    resp_df = pd.DataFrame(response_rows)

    n_both = int(align_df["both_match"].sum()) if not align_df.empty else 0
    match_rate = n_both / n_compare if n_compare else np.nan

    target_mae = float(resp_df["abs_error_target_hypothesis_ms"].dropna().median()) if resp_df["abs_error_target_hypothesis_ms"].notna().any() else np.nan
    cue_mae = float(resp_df["abs_error_cue_hypothesis_ms"].dropna().median()) if resp_df["abs_error_cue_hypothesis_ms"].notna().any() else np.nan

    if pd.notna(target_mae) and pd.notna(cue_mae):
        if target_mae + 10 < cue_mae:
            timing = "condition_event_closer_to_target"
        elif cue_mae + 10 < target_mae:
            timing = "condition_event_closer_to_cue"
        else:
            timing = "ambiguous"
    else:
        timing = "not_evaluable"

    summary = {
        "subject_id": subject,
        "set_file": set_path.name,
        "n_metadata_trials": int(n_meta),
        "n_condition_events": int(n_cond),
        "n_response_events": int(len(resp)),
        "n_compared_trials": int(n_compare),
        "n_exact_condition_matches": int(n_both),
        "condition_sequence_match_rate": float(match_rate) if pd.notna(match_rate) else np.nan,
        "median_response_abs_error_target_hypothesis_ms": target_mae,
        "median_response_abs_error_cue_hypothesis_ms": cue_mae,
        "inferred_condition_event_timing": timing,
        "sfreq_hz": float(raw.info.get("sfreq", np.nan)),
        "n_channels": int(len(raw.ch_names)),
    }

    return align_df, resp_df, summary


def write_report(out_dir: Path, summary: pd.DataFrame) -> None:
    lines = []
    lines.append("# EEG event-code and trial-alignment report")
    lines.append("")
    lines.append("This report checks whether EEG condition events align with the trial-level metadata.")
    lines.append("")
    if summary.empty:
        lines.append("No subjects were processed.")
    else:
        n = len(summary)
        exact_all = int((summary["condition_sequence_match_rate"] == 1.0).sum())
        readable = int(summary["error"].isna().sum()) if "error" in summary.columns else n
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Subjects processed: {n}")
        lines.append(f"- Subjects without processing errors: {readable}")
        lines.append(f"- Subjects with perfect condition-sequence alignment: {exact_all}/{n}")
        lines.append(f"- Condition event counts observed: {sorted(summary['n_condition_events'].dropna().astype(int).unique().tolist())}")
        lines.append(f"- Response event counts range: {int(summary['n_response_events'].min())}–{int(summary['n_response_events'].max())}")
        timing_counts = summary["inferred_condition_event_timing"].value_counts(dropna=False).to_dict()
        lines.append(f"- Condition timing inference counts: {timing_counts}")
        lines.append("")
        lines.append("## Interpretation")
        lines.append("")
        if exact_all == n:
            lines.append("- The six condition codes align exactly with the metadata trial sequence for every subject.")
        else:
            lines.append("- Some subjects do not show perfect condition-code alignment; inspect `eeg_trial_alignment_rows.csv` before epoching.")
        lines.append("- Code 9 behaves as a response-marker candidate, but the timing check should be used before response-locked analyses.")
        lines.append("")
        lines.append("## Default condition-code map used")
        lines.append("")
        lines.append("| Code | Cue | Trial class |")
        lines.append("|---:|---|---|")
        lines.append("| 1 | up | nogo |")
        lines.append("| 2 | up | go |")
        lines.append("| 3 | bi | nogo |")
        lines.append("| 4 | bi | go |")
        lines.append("| 5 | down | nogo |")
        lines.append("| 6 | down | go |")
        lines.append("| 9 | response marker candidate | |")
        lines.append("")
        lines.append("## Recommended next step")
        lines.append("")
        lines.append("If condition alignment is perfect, run a first neural-dynamics extraction using predefined windows:")
        lines.append("cue-locked CNV/N2/P3, target-locked N2/P300, response-preparation RP/mu-beta features, and error-locked ERN/Pe/theta.")
        lines.append("Keep this as an exploratory neural validation branch until cross-validated or permutation-corrected results are available.")
    lines.append("")
    (out_dir / "eeg_event_trial_alignment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default=".")
    ap.add_argument("--metadata", type=str, required=True)
    ap.add_argument("--eeg_root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--condition_map_json", type=str, default="")
    ap.add_argument("--response_code", type=str, default="9")
    ap.add_argument("--target_delay_sec", type=float, default=0.5)
    ap.add_argument("--response_window_sec", type=float, default=2.0)
    args = ap.parse_args()

    base = Path(args.base).expanduser().resolve()
    metadata_path = (base / args.metadata).resolve() if not Path(args.metadata).is_absolute() else Path(args.metadata).resolve()
    eeg_root = (base / args.eeg_root).resolve() if not Path(args.eeg_root).is_absolute() else Path(args.eeg_root).resolve()
    out_dir = (base / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    condition_map = DEFAULT_CONDITION_MAP
    if args.condition_map_json:
        condition_map = json.loads(Path(args.condition_map_json).read_text(encoding="utf-8"))

    metadata = load_metadata(metadata_path)
    set_by_subject = choose_preferred_set_files(find_set_files(eeg_root))

    all_align = []
    all_resp = []
    summaries = []

    subjects = sorted(metadata["subject_id_norm"].dropna().unique().tolist())
    for subject in subjects:
        meta_sub = metadata[metadata["subject_id_norm"] == subject].copy()
        if subject not in set_by_subject:
            summaries.append({
                "subject_id": subject,
                "set_file": "",
                "error": "missing_set_file",
                "n_metadata_trials": len(meta_sub),
                "n_condition_events": np.nan,
                "n_response_events": np.nan,
                "n_compared_trials": 0,
                "n_exact_condition_matches": 0,
                "condition_sequence_match_rate": np.nan,
                "median_response_abs_error_target_hypothesis_ms": np.nan,
                "median_response_abs_error_cue_hypothesis_ms": np.nan,
                "inferred_condition_event_timing": "not_evaluable",
                "sfreq_hz": np.nan,
                "n_channels": np.nan,
            })
            continue

        try:
            align_df, resp_df, summary = trial_alignment_for_subject(
                subject=subject,
                set_path=set_by_subject[subject],
                meta_sub=meta_sub,
                condition_map=condition_map,
                response_code=str(args.response_code),
                target_delay_sec=float(args.target_delay_sec),
                response_window_sec=float(args.response_window_sec),
            )
            summary["error"] = np.nan
            all_align.append(align_df)
            all_resp.append(resp_df)
            summaries.append(summary)
        except Exception as exc:
            summaries.append({
                "subject_id": subject,
                "set_file": set_by_subject[subject].name,
                "error": f"{type(exc).__name__}: {exc}",
                "n_metadata_trials": len(meta_sub),
                "n_condition_events": np.nan,
                "n_response_events": np.nan,
                "n_compared_trials": 0,
                "n_exact_condition_matches": 0,
                "condition_sequence_match_rate": np.nan,
                "median_response_abs_error_target_hypothesis_ms": np.nan,
                "median_response_abs_error_cue_hypothesis_ms": np.nan,
                "inferred_condition_event_timing": "not_evaluable",
                "sfreq_hz": np.nan,
                "n_channels": np.nan,
            })

    align_out = pd.concat(all_align, ignore_index=True) if all_align else pd.DataFrame()
    resp_out = pd.concat(all_resp, ignore_index=True) if all_resp else pd.DataFrame()
    summary_out = pd.DataFrame(summaries)

    summary_out.to_csv(out_dir / "eeg_trial_alignment_by_subject.csv", index=False)
    align_out.to_csv(out_dir / "eeg_trial_alignment_rows.csv", index=False)
    resp_out.to_csv(out_dir / "eeg_response_alignment_rows.csv", index=False)
    write_report(out_dir, summary_out)

    print(f"Wrote outputs to: {out_dir}")
    if not summary_out.empty:
        print(summary_out[["subject_id", "condition_sequence_match_rate", "n_condition_events", "n_response_events", "inferred_condition_event_timing", "error"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
