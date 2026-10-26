#!/usr/bin/env python3
"""Trial-wise EEG feature extraction for the EEG-computational study neural-dynamics analysis.

Condition event codes 1-6 identify cue type and Go/NoGo trial class. The
condition marker defines cue onset, target onset is represented as cue onset
plus 0.5 s, and response timing is reconstructed from behavioral response time
relative to target onset. Event code 9 is retained as a response-marker
quality-control candidate.

The program extracts trial-level ERP features and related quality-control and
summary outputs from processed EEGLAB datasets. Optional computational latent
inputs are used only for downstream summary analyses and are not required for
ERP feature extraction."""

from __future__ import annotations

import argparse
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


CONDITION_MAP = {
    "1": {"cue_type": "up", "trial_class": "nogo"},
    "2": {"cue_type": "up", "trial_class": "go"},
    "3": {"cue_type": "bi", "trial_class": "nogo"},
    "4": {"cue_type": "bi", "trial_class": "go"},
    "5": {"cue_type": "down", "trial_class": "nogo"},
    "6": {"cue_type": "down", "trial_class": "go"},
}

CONDITION_CODES = set(CONDITION_MAP.keys())
RESPONSE_CODE = "9"


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
    required = [
        "subject_id", "trial_index", "trial_class", "cue_type",
        "response_present", "rt_response_from_target_ms"
    ]
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
    if "response_type" in out.columns:
        out["response_type_norm"] = out["response_type"].map(normalize_label)
    else:
        out["response_type_norm"] = ""
    out = out.sort_values(["subject_id_norm", "trial_index_numeric"]).reset_index(drop=True)
    return out


def load_latents(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    if "subject_id" in df.columns:
        df["subject_id_norm"] = df["subject_id"].map(normalize_subject_id)
    elif "subj" in df.columns:
        df["subject_id_norm"] = df["subj"].map(normalize_subject_id)
    else:
        raise ValueError("Latent file must contain subject_id or subj.")

    if "trial_index" in df.columns:
        df["trial_index_numeric"] = pd.to_numeric(df["trial_index"], errors="coerce")
    elif "trial" in df.columns:
        df["trial_index_numeric"] = pd.to_numeric(df["trial"], errors="coerce")
    else:
        raise ValueError("Latent file must contain trial_index or trial.")

    # Avoid duplicate raw columns after merge.
    reserved = {"subject_id", "subj", "trial", "trial_index"}
    keep_cols = [c for c in df.columns if c not in reserved]
    return df[keep_cols].copy()


def read_raw_and_events(path: Path):
    import mne
    raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose="ERROR")
    events = []
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        events.append({
            "onset_sec": float(onset),
            "description": str(desc),
            "code": parse_event_code(desc),
        })
    ev = pd.DataFrame(events).sort_values("onset_sec").reset_index(drop=True)
    return raw, ev


def get_channel(raw, name: str) -> Optional[str]:
    lower = {ch.lower(): ch for ch in raw.ch_names}
    return lower.get(name.lower())


def get_data_uv(raw, channel: str, t0: float, t1: float) -> Optional[np.ndarray]:
    sfreq = float(raw.info["sfreq"])
    start = int(round(t0 * sfreq))
    stop = int(round(t1 * sfreq))
    if stop <= start:
        return None
    if start < 0 or stop > raw.n_times:
        return None
    data = raw.get_data(picks=[channel], start=start, stop=stop)[0]
    # MNE returns EEG data in Volts for standard EEG readers; convert to microvolts.
    return data * 1e6


def baseline_corrected_segment(
    raw,
    channel: str,
    event_time: float,
    window: Tuple[float, float],
    baseline: Optional[Tuple[float, float]],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    sfreq = float(raw.info["sfreq"])
    t0 = event_time + window[0]
    t1 = event_time + window[1]
    data = get_data_uv(raw, channel, t0, t1)
    if data is None:
        return None
    times = np.arange(len(data)) / sfreq + window[0]

    if baseline is not None:
        b0 = event_time + baseline[0]
        b1 = event_time + baseline[1]
        bdata = get_data_uv(raw, channel, b0, b1)
        if bdata is not None and len(bdata) > 0:
            data = data - np.nanmean(bdata)
    return data, times


def extract_window_features(
    raw,
    event_time: Optional[float],
    channel: Optional[str],
    window: Tuple[float, float],
    baseline: Optional[Tuple[float, float]],
    prefix: str,
) -> Dict[str, float]:
    out = {
        f"{prefix}_mean_uv": np.nan,
        f"{prefix}_min_uv": np.nan,
        f"{prefix}_max_uv": np.nan,
        f"{prefix}_minlat_ms": np.nan,
        f"{prefix}_maxlat_ms": np.nan,
        f"{prefix}_slope_uv_per_s": np.nan,
    }
    if event_time is None or channel is None or not np.isfinite(event_time):
        return out

    seg = baseline_corrected_segment(raw, channel, float(event_time), window, baseline)
    if seg is None:
        return out

    data, times = seg
    if len(data) < 2:
        return out

    out[f"{prefix}_mean_uv"] = float(np.nanmean(data))
    min_idx = int(np.nanargmin(data))
    max_idx = int(np.nanargmax(data))
    out[f"{prefix}_min_uv"] = float(data[min_idx])
    out[f"{prefix}_max_uv"] = float(data[max_idx])
    out[f"{prefix}_minlat_ms"] = float(times[min_idx] * 1000.0)
    out[f"{prefix}_maxlat_ms"] = float(times[max_idx] * 1000.0)

    valid = np.isfinite(data)
    if valid.sum() >= 3:
        try:
            out[f"{prefix}_slope_uv_per_s"] = float(np.polyfit(times[valid], data[valid], deg=1)[0])
        except Exception:
            pass
    return out


def extract_trial_features(
    raw,
    trial_row: pd.Series,
    cue_onset: float,
    response_onset: Optional[float],
    channels: Dict[str, Optional[str]],
    target_delay_sec: float,
) -> Dict[str, float]:
    target_onset = cue_onset + target_delay_sec

    features: Dict[str, float] = {}

    # Cue-locked neural preparation and cue evaluation.
    features.update(extract_window_features(raw, cue_onset, channels.get("Cz"), (0.10, 0.20), (-0.20, 0.00), "cue_cnv_early_cz_100_200"))
    features.update(extract_window_features(raw, cue_onset, channels.get("Cz"), (0.20, 0.50), (-0.20, 0.00), "cue_cnv_late_cz_200_500"))
    features.update(extract_window_features(raw, cue_onset, channels.get("Fz"), (0.20, 0.35), (-0.20, 0.00), "cue_n2_fz_200_350"))
    features.update(extract_window_features(raw, cue_onset, channels.get("Cz"), (0.20, 0.35), (-0.20, 0.00), "cue_n2_cz_200_350"))
    features.update(extract_window_features(raw, cue_onset, channels.get("Cz"), (0.25, 0.45), (-0.20, 0.00), "cue_p3_cz_250_450"))

    # Target-locked stimulus evaluation and conflict/selection.
    features.update(extract_window_features(raw, target_onset, channels.get("Fz"), (0.20, 0.35), (-0.20, 0.00), "target_n2_fz_200_350"))
    features.update(extract_window_features(raw, target_onset, channels.get("Cz"), (0.20, 0.35), (-0.20, 0.00), "target_n2_cz_200_350"))
    features.update(extract_window_features(raw, target_onset, channels.get("Cz"), (0.25, 0.50), (-0.20, 0.00), "target_p300_cz_250_500"))

    # Response preparation and response-locked performance monitoring.
    if response_onset is not None and np.isfinite(response_onset):
        features.update(extract_window_features(raw, response_onset, channels.get("Cz"), (-0.30, 0.00), (-0.50, -0.30), "resp_rp_cz_m300_0"))
        features.update(extract_window_features(raw, response_onset, channels.get("Fz"), (0.05, 0.15), (-0.20, 0.00), "resp_ern_fz_50_150"))
        features.update(extract_window_features(raw, response_onset, channels.get("Fz"), (0.20, 0.40), (-0.20, 0.00), "resp_pe_fz_200_400"))
        features.update(extract_window_features(raw, response_onset, channels.get("Cz"), (0.20, 0.40), (-0.20, 0.00), "resp_pe_cz_200_400"))
    else:
        for prefix in ["resp_rp_cz_m300_0", "resp_ern_fz_50_150", "resp_pe_fz_200_400", "resp_pe_cz_200_400"]:
            features.update({
                f"{prefix}_mean_uv": np.nan,
                f"{prefix}_min_uv": np.nan,
                f"{prefix}_max_uv": np.nan,
                f"{prefix}_minlat_ms": np.nan,
                f"{prefix}_maxlat_ms": np.nan,
                f"{prefix}_slope_uv_per_s": np.nan,
            })

    return features


def nearest_response_marker(
    response_onsets: np.ndarray,
    predicted_response_time: Optional[float],
    max_abs_diff_sec: float = 0.15,
) -> Tuple[float, float]:
    if predicted_response_time is None or not np.isfinite(predicted_response_time) or len(response_onsets) == 0:
        return np.nan, np.nan
    idx = int(np.argmin(np.abs(response_onsets - predicted_response_time)))
    marker_time = float(response_onsets[idx])
    diff_ms = (marker_time - float(predicted_response_time)) * 1000.0
    if abs(diff_ms) > max_abs_diff_sec * 1000.0:
        return np.nan, np.nan
    return marker_time, diff_ms


def process_subject(
    subject: str,
    set_path: Path,
    meta_sub: pd.DataFrame,
    target_delay_sec: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    raw, ev = read_raw_and_events(set_path)

    condition_events = ev[ev["code"].isin(CONDITION_CODES)].copy().reset_index(drop=True)
    response_events = ev[ev["code"] == RESPONSE_CODE].copy().reset_index(drop=True)
    response_onsets = response_events["onset_sec"].astype(float).values if not response_events.empty else np.array([])

    channels = {name: get_channel(raw, name) for name in ["Fz", "Cz", "Pz", "C3", "C4"]}

    n_trials = min(len(meta_sub), len(condition_events))
    rows = []
    qc_rows = []

    for i in range(n_trials):
        m = meta_sub.iloc[i]
        cond = condition_events.iloc[i]
        cue_onset = float(cond["onset_sec"])
        target_onset = cue_onset + target_delay_sec
        code = str(cond["code"])

        responded = int(m["response_present_numeric"]) == 1 and pd.notna(m["rt_sec"])
        response_onset_from_metadata = target_onset + float(m["rt_sec"]) if responded else np.nan
        marker_onset, marker_error_ms = nearest_response_marker(response_onsets, response_onset_from_metadata)

        response_onset = response_onset_from_metadata if responded else None

        expected = CONDITION_MAP.get(code, {})
        cue_match = normalize_label(m["cue_type"]) == expected.get("cue_type", "")
        class_match = normalize_label(m["trial_class"]) == expected.get("trial_class", "")

        base_row = {
            "subject_id": subject,
            "subject_id_norm": normalize_subject_id(subject),
            "group": "high" if subject.startswith("H") else "low",
            "trial_ordinal": i + 1,
            "trial_index": m["trial_index"],
            "trial_index_numeric": pd.to_numeric(m["trial_index"], errors="coerce"),
            "cue_type": normalize_label(m["cue_type"]),
            "trial_class": normalize_label(m["trial_class"]),
            "response_type": normalize_label(m.get("response_type", "")),
            "response_present": int(m["response_present_numeric"]),
            "rt_response_from_target_ms": float(m["rt_response_from_target_ms"]) if pd.notna(m["rt_response_from_target_ms"]) else np.nan,
            "eeg_condition_code": code,
            "cue_onset_sec": cue_onset,
            "target_onset_sec": target_onset,
            "response_onset_from_metadata_sec": response_onset_from_metadata,
            "response_marker_onset_sec": marker_onset,
            "response_marker_error_ms": marker_error_ms,
            "condition_cue_match": bool(cue_match),
            "condition_trial_class_match": bool(class_match),
        }

        feature_row = extract_trial_features(raw, m, cue_onset, response_onset, channels, target_delay_sec)
        rows.append({**base_row, **feature_row})

        qc_rows.append({
            "subject_id": subject,
            "trial_ordinal": i + 1,
            "responded": bool(responded),
            "rt_sec": float(m["rt_sec"]) if pd.notna(m["rt_sec"]) else np.nan,
            "response_onset_from_metadata_sec": response_onset_from_metadata,
            "nearest_response_marker_sec": marker_onset,
            "response_marker_error_ms": marker_error_ms,
        })

    summary = {
        "subject_id": subject,
        "set_file": set_path.name,
        "n_metadata_trials": int(len(meta_sub)),
        "n_condition_events": int(len(condition_events)),
        "n_response_events": int(len(response_events)),
        "n_extracted_trials": int(n_trials),
        "sfreq_hz": float(raw.info["sfreq"]),
        "n_channels": int(len(raw.ch_names)),
        "channels_found": ",".join([f"{k}:{v}" for k, v in channels.items()]),
        "condition_match_rate": float(np.mean([r["condition_cue_match"] and r["condition_trial_class_match"] for r in rows])) if rows else np.nan,
        "median_response_marker_abs_error_ms": float(np.nanmedian(np.abs([q["response_marker_error_ms"] for q in qc_rows]))) if qc_rows else np.nan,
    }

    return pd.DataFrame(rows), pd.DataFrame(qc_rows), summary


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan, dtype=float)
    ok = np.isfinite(pvals)
    if ok.sum() == 0:
        return qvals
    p = pvals[ok]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    qvals[ok] = out
    return qvals


def within_subject_demean(df: pd.DataFrame, col: str, subject_col: str = "subject_id") -> pd.Series:
    x = pd.to_numeric(df[col], errors="coerce")
    return x - x.groupby(df[subject_col]).transform("mean")


def encoding_screen(df: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats

    candidate_latents = [
        "p_press", "actor_qdiff", "control", "belief_nogo", "associability",
        "prediction_error", "pe", "surprise", "policy", "value", "qdiff"
    ]
    latent_cols = [c for c in candidate_latents if c in df.columns]

    feature_keywords = [
        "cue_cnv", "cue_n2", "cue_p3", "target_n2", "target_p300",
        "resp_rp", "resp_ern", "resp_pe"
    ]
    neural_cols = [
        c for c in df.columns
        if any(k in c for k in feature_keywords)
        and (c.endswith("_mean_uv") or c.endswith("_min_uv") or c.endswith("_max_uv") or c.endswith("_slope_uv_per_s"))
    ]

    rows = []
    for latent in latent_cols:
        x = within_subject_demean(df, latent)
        for neural in neural_cols:
            y = within_subject_demean(df, neural)
            mask = np.isfinite(x) & np.isfinite(y)
            n = int(mask.sum())
            if n < 50:
                continue
            r, p = stats.pearsonr(x[mask], y[mask])
            # OLS slope for y ~ x after subject demeaning.
            denom = float(np.dot(x[mask], x[mask]))
            beta = float(np.dot(x[mask], y[mask]) / denom) if denom > 0 else np.nan
            rows.append({
                "latent_variable": latent,
                "neural_feature": neural,
                "n_trials": n,
                "within_subject_r": float(r),
                "p_value": float(p),
                "beta_y_per_x": beta,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_fdr_bh"] = fdr_bh(out["p_value"].values)
        out = out.sort_values(["q_value_fdr_bh", "p_value", "latent_variable", "neural_feature"]).reset_index(drop=True)
    return out


def summarize_by_group(features: pd.DataFrame) -> pd.DataFrame:
    neural_cols = [
        c for c in features.columns
        if (
            c.endswith("_mean_uv") or c.endswith("_min_uv") or c.endswith("_max_uv")
            or c.endswith("_slope_uv_per_s")
        )
    ]
    rows = []
    for group, gdf in features.groupby("group"):
        for col in neural_cols:
            x = pd.to_numeric(gdf[col], errors="coerce")
            rows.append({
                "group": group,
                "feature": col,
                "n": int(x.notna().sum()),
                "mean": float(x.mean()) if x.notna().any() else np.nan,
                "sd": float(x.std()) if x.notna().sum() > 1 else np.nan,
                "median": float(x.median()) if x.notna().any() else np.nan,
            })
    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    features: pd.DataFrame,
    subject_summary: pd.DataFrame,
    encoding: Optional[pd.DataFrame],
    latents_supplied: bool,
) -> None:
    lines = []
    lines.append("# EEG neural-dynamics extraction report")
    lines.append("")
    lines.append("This report summarizes trial-wise EEG features extracted from validated cue, target, and response timings.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Subjects processed: {subject_summary['subject_id'].nunique() if not subject_summary.empty else 0}")
    lines.append(f"- Trial-wise feature rows: {len(features)}")
    if not subject_summary.empty:
        lines.append(f"- Extracted trials per subject: {int(subject_summary['n_extracted_trials'].min())}–{int(subject_summary['n_extracted_trials'].max())}")
        lines.append(f"- Median response-marker absolute error: {float(subject_summary['median_response_marker_abs_error_ms'].median()):.3f} ms")
        lines.append(f"- Condition match rate range: {float(subject_summary['condition_match_rate'].min()):.3f}–{float(subject_summary['condition_match_rate'].max()):.3f}")
    lines.append("")
    lines.append("## Feature families")
    lines.append("")
    lines.append("- Cue-locked: CNV, N2, and P3 windows.")
    lines.append("- Target-locked: N2 and P300 windows.")
    lines.append("- Response-locked: readiness potential, ERN, and Pe windows.")
    lines.append("")
    lines.append("## Computational encoding screen")
    lines.append("")
    if not latents_supplied:
        lines.append("- No latent file was supplied, so no computational encoding screen was run.")
    elif encoding is None or encoding.empty:
        lines.append("- A latent file was supplied, but no eligible latent-feature pairs were available for screening.")
    else:
        lines.append(f"- Encoding pairs tested: {len(encoding)}")
        sig = encoding[encoding["q_value_fdr_bh"] < 0.05]
        lines.append(f"- FDR q < 0.05 pairs: {len(sig)}")
        lines.append("")
        lines.append("Descriptive feature-association summary:")
        lines.append("")
        lines.append("| Latent | Neural feature | r | p | q |")
        lines.append("|---|---|---:|---:|---:|")
        for _, row in encoding.head(15).iterrows():
            lines.append(
                f"| {row['latent_variable']} | {row['neural_feature']} | "
                f"{row['within_subject_r']:.4f} | {row['p_value']:.3g} | {row['q_value_fdr_bh']:.3g} |"
            )
    lines.append("")
    lines.append("## Interpretation constraints")
    lines.append("")
    lines.append("These outputs provide descriptive feature-extraction and quality-control summaries. Inferential interpretation should rely on")
    lines.append("predefined contrasts, participant-level aggregation, and appropriate multiple-comparison control across the evaluated time, channel, and feature families.")
    lines.append("")
    lines.append("## Analysis scope")
    lines.append("")
    lines.append("The exported trial-wise features support downstream analyses defined separately from this extraction workflow.")
    lines.append("Primary inferential analyses should use predefined feature sets, participant-level evaluation, and the corresponding multiple-comparison procedures.")
    lines.append("")
    (out_dir / "eeg_neural_dynamics_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default=".")
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--eeg_root", type=str, required=True)
    parser.add_argument("--latents", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--target_delay_sec", type=float, default=0.5)
    args = parser.parse_args()

    base = Path(args.base).expanduser().resolve()
    metadata_path = (base / args.metadata).resolve() if not Path(args.metadata).is_absolute() else Path(args.metadata).resolve()
    eeg_root = (base / args.eeg_root).resolve() if not Path(args.eeg_root).is_absolute() else Path(args.eeg_root).resolve()
    out_dir = (base / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    latent_path = None
    if args.latents:
        latent_path = (base / args.latents).resolve() if not Path(args.latents).is_absolute() else Path(args.latents).resolve()

    metadata = load_metadata(metadata_path)
    latents = load_latents(latent_path) if latent_path else None
    set_by_subject = choose_preferred_set_files(find_set_files(eeg_root))

    all_features = []
    all_qc = []
    summaries = []

    subjects = sorted(metadata["subject_id_norm"].dropna().unique().tolist())
    for subject in subjects:
        if subject not in set_by_subject:
            summaries.append({
                "subject_id": subject,
                "error": "missing_eeg_set_file",
                "n_extracted_trials": 0,
            })
            continue
        meta_sub = metadata[metadata["subject_id_norm"] == subject].copy()
        try:
            feat, qc, summ = process_subject(subject, set_by_subject[subject], meta_sub, args.target_delay_sec)
            summ["error"] = ""
            all_features.append(feat)
            all_qc.append(qc)
            summaries.append(summ)
            print(f"{subject}: extracted {len(feat)} trials")
        except Exception as exc:
            summaries.append({
                "subject_id": subject,
                "error": f"{type(exc).__name__}: {exc}",
                "n_extracted_trials": 0,
            })
            print(f"{subject}: ERROR {type(exc).__name__}: {exc}")

    features = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()
    qc = pd.concat(all_qc, ignore_index=True) if all_qc else pd.DataFrame()
    subject_summary = pd.DataFrame(summaries)

    if latents is not None and not features.empty:
        # Merge EEG trial-wise features with computational latent variables using
        # normalized subject IDs and numeric trial indices. These keys are robust
        # to source-specific subject label formatting.
        features["subject_id_norm"] = features["subject_id"].map(normalize_subject_id)
        features["trial_index_numeric"] = pd.to_numeric(features["trial_index"], errors="coerce")

        if "subject_id_norm" not in latents.columns or "trial_index_numeric" not in latents.columns:
            raise ValueError("Latent table must contain normalized merge keys: subject_id_norm and trial_index_numeric.")

        features = features.merge(
            latents,
            on=["subject_id_norm", "trial_index_numeric"],
            how="left",
            suffixes=("", "_latent"),
        )

    features.to_csv(out_dir / "eeg_trial_neural_features.csv", index=False)
    qc.to_csv(out_dir / "eeg_response_marker_qc.csv", index=False)
    subject_summary.to_csv(out_dir / "eeg_neural_extraction_by_subject.csv", index=False)

    group_summary = summarize_by_group(features) if not features.empty else pd.DataFrame()
    group_summary.to_csv(out_dir / "eeg_neural_feature_summary_by_group.csv", index=False)

    encoding = None
    if latents is not None and not features.empty:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            encoding = encoding_screen(features)
        encoding.to_csv(out_dir / "eeg_computational_encoding_screen.csv", index=False)

    write_report(out_dir, features, subject_summary, encoding, latents is not None)

    print(f"Wrote outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
