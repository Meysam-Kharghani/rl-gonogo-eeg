#!/usr/bin/env python3
# FORMALIZED HISTORICAL SOURCE SNAPSHOT
# Evidence status: HISTORICAL_OR_SUPERSEDED_DEVELOPMENT_EVIDENCE
# This file preserves a historical project stage with public-facing terminology standardized for release.
# The SHA-256 of the original executed/development source is recorded in the corresponding provenance registry.

"""
ERP waveform and difference-wave analysis for the EEG-computational study.

This script performs a non-destructive subject-level ERP waveform analysis from
validated EEGLAB .set/.fdt files.

Event mapping assumed from prior audit:
1 = up / nogo
2 = up / go
3 = bi / nogo
4 = bi / go
5 = down / nogo
6 = down / go
9 = response marker

Timing:
- codes 1–6 mark cue onset
- target onset = cue onset + 0.5 s
- response onset is reconstructed as target onset + behavioral RT

Outputs:
- erp_subject_waveforms_long.csv
- erp_group_waveforms_long.csv
- erp_subject_window_features.csv
- erp_window_contrast_stats.csv
- erp_timepoint_contrast_stats.csv
- erp_significant_time_ranges.csv
- figures/*.png
- ERP_WAVEFORM_ANALYSIS_REPORT.md

Example:
python code/final/run_eeg_erp_waveform_analysis_v1.py \
  --base . \
  --metadata data/raw/csv/ALL_SUBJECTS_metadata_features_merged_ORDERED.xlsx \
  --eeg_root "data/raw data/Groups_eventlistdefined" \
  --out results/eeg_erp_waveform_analysis_v1
"""

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
from scipy import stats


CONDITION_MAP = {
    "1": {"cue_type": "up", "trial_class": "nogo"},
    "2": {"cue_type": "up", "trial_class": "go"},
    "3": {"cue_type": "bi", "trial_class": "nogo"},
    "4": {"cue_type": "bi", "trial_class": "go"},
    "5": {"cue_type": "down", "trial_class": "nogo"},
    "6": {"cue_type": "down", "trial_class": "go"},
}
CONDITION_CODES = set(CONDITION_MAP.keys())

OUTCOME_ORDER = ["go_hit", "go_miss", "nogo_cr", "nogo_fa"]
OUTCOME_LABELS = {
    "go_hit": "Go Hit",
    "go_miss": "Go Miss",
    "nogo_cr": "NoGo Correct Rejection",
    "nogo_fa": "NoGo False Alarm",
}

LOCK_WINDOWS = {
    "cue": (-0.2, 0.8),
    "target": (-0.2, 0.8),
    "response": (-0.5, 0.6),
}
BASELINES = {
    "cue": (-0.2, 0.0),
    "target": (-0.2, 0.0),
    "response": (-0.5, -0.3),
}
CHANNELS = ["Fz", "Cz", "Pz"]

WINDOW_SPECS = [
    {"lock": "cue", "channel": "Cz", "name": "cue_cnv_200_500", "t0": 0.200, "t1": 0.500, "measure": "mean"},
    {"lock": "cue", "channel": "Cz", "name": "cue_p3_250_450", "t0": 0.250, "t1": 0.450, "measure": "mean"},
    {"lock": "cue", "channel": "Pz", "name": "cue_p3_pz_250_450", "t0": 0.250, "t1": 0.450, "measure": "mean"},
    {"lock": "cue", "channel": "Fz", "name": "cue_n2_fz_200_350", "t0": 0.200, "t1": 0.350, "measure": "mean"},
    {"lock": "target", "channel": "Fz", "name": "target_n2_fz_200_350", "t0": 0.200, "t1": 0.350, "measure": "mean"},
    {"lock": "target", "channel": "Cz", "name": "target_n2_cz_200_350", "t0": 0.200, "t1": 0.350, "measure": "mean"},
    {"lock": "target", "channel": "Pz", "name": "target_p300_pz_250_500", "t0": 0.250, "t1": 0.500, "measure": "mean"},
    {"lock": "target", "channel": "Cz", "name": "target_p300_cz_250_500", "t0": 0.250, "t1": 0.500, "measure": "mean"},
    {"lock": "response", "channel": "Cz", "name": "resp_rp_cz_m300_0", "t0": -0.300, "t1": 0.000, "measure": "mean"},
    {"lock": "response", "channel": "Fz", "name": "resp_ern_fz_50_150", "t0": 0.050, "t1": 0.150, "measure": "mean"},
    {"lock": "response", "channel": "Fz", "name": "resp_ern_fz_min_50_150", "t0": 0.050, "t1": 0.150, "measure": "min"},
    {"lock": "response", "channel": "Fz", "name": "resp_pe_fz_200_400", "t0": 0.200, "t1": 0.400, "measure": "mean"},
    {"lock": "response", "channel": "Cz", "name": "resp_pe_cz_200_400", "t0": 0.200, "t1": 0.400, "measure": "mean"},
    {"lock": "response", "channel": "Pz", "name": "resp_pe_pz_200_400", "t0": 0.200, "t1": 0.400, "measure": "mean"},
]

CONTRAST_SPECS = [
    {"name": "nogo_cr_minus_nogo_fa", "a": "nogo_cr", "b": "nogo_fa", "locks": ["cue", "target"], "interpretation": "successful inhibition minus inhibitory-control failure"},
    {"name": "nogo_fa_minus_go_hit", "a": "nogo_fa", "b": "go_hit", "locks": ["response"], "interpretation": "error monitoring: false alarm minus correct go response"},
    {"name": "go_hit_minus_go_miss", "a": "go_hit", "b": "go_miss", "locks": ["cue", "target"], "interpretation": "successful go response minus go omission"},
]


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
    parts = list(path.parts)
    subject = ""
    for part in reversed(parts):
        if re.fullmatch(r"[HLhl]\d+", part.strip()):
            subject = normalize_subject_id(part)
            break
    if not subject:
        m = re.search(r"([HLhl]\d+)", path.name)
        if m:
            subject = normalize_subject_id(m.group(1))
    group = "high" if subject.startswith("H") else ("low" if subject.startswith("L") else "")
    return subject, group


def choose_preferred_set_files(eeg_root: Path) -> Dict[str, Path]:
    files = sorted(eeg_root.rglob("*.set"))
    by_subject: Dict[str, List[Path]] = defaultdict(list)
    for p in files:
        subject, _ = infer_subject_group_from_path(p)
        if subject:
            by_subject[subject].append(p)

    selected = {}
    for subject, candidates in sorted(by_subject.items()):
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
    out["trial_index_numeric"] = pd.to_numeric(out["trial_index"], errors="coerce")
    out["trial_class_norm"] = out["trial_class"].map(normalize_label)
    out["cue_type_norm"] = out["cue_type"].map(normalize_label)
    out["response_present_numeric"] = pd.to_numeric(out["response_present"], errors="coerce").fillna(0).astype(int)
    out["rt_sec"] = pd.to_numeric(out["rt_response_from_target_ms"], errors="coerce") / 1000.0

    if "response_type" in out.columns:
        out["response_type_norm"] = out["response_type"].map(normalize_label)
    else:
        out["response_type_norm"] = ""

    def outcome(row):
        tc = row["trial_class_norm"]
        press = int(row["response_present_numeric"])
        rt = row["rt_sec"]
        rt_ok = pd.notna(rt)
        rtype = row["response_type_norm"]
        if rtype in {"hit", "go_hit"}:
            return "go_hit"
        if rtype in {"miss", "go_miss"}:
            return "go_miss"
        if rtype in {"cr", "correct_rejection", "correct rejection", "nogo_cr"}:
            return "nogo_cr"
        if rtype in {"fa", "false_alarm", "false alarm", "nogo_fa"}:
            return "nogo_fa"
        if tc == "go" and press == 1 and rt_ok:
            return "go_hit"
        if tc == "go" and press == 0:
            return "go_miss"
        if tc == "nogo" and press == 0:
            return "nogo_cr"
        if tc == "nogo" and press == 1 and rt_ok:
            return "nogo_fa"
        return "unknown"

    out["outcome"] = out.apply(outcome, axis=1)
    out = out.sort_values(["subject_id_norm", "trial_index_numeric"]).reset_index(drop=True)
    return out


def read_raw_events(path: Path):
    import mne
    raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose="ERROR")
    rows = []
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        rows.append({
            "onset_sec": float(onset),
            "description": str(desc),
            "code": parse_event_code(desc),
        })
    events = pd.DataFrame(rows).sort_values("onset_sec").reset_index(drop=True)
    return raw, events


def get_channel(raw, name: str) -> Optional[str]:
    lower = {ch.lower(): ch for ch in raw.ch_names}
    return lower.get(name.lower())


def standard_times(sfreq: float, lock: str) -> np.ndarray:
    t0, t1 = LOCK_WINDOWS[lock]
    n = int(round((t1 - t0) * sfreq))
    return np.arange(n, dtype=float) / sfreq + t0


def get_segment_uv(raw, channel: str, event_time: float, lock: str) -> Optional[np.ndarray]:
    sfreq = float(raw.info["sfreq"])
    t0, t1 = LOCK_WINDOWS[lock]
    start = int(round((event_time + t0) * sfreq))
    stop = start + int(round((t1 - t0) * sfreq))
    if start < 0 or stop > raw.n_times or stop <= start:
        return None
    data = raw.get_data(picks=[channel], start=start, stop=stop)[0] * 1e6

    b0, b1 = BASELINES[lock]
    bs = int(round((event_time + b0) * sfreq))
    be = int(round((event_time + b1) * sfreq))
    if bs >= 0 and be <= raw.n_times and be > bs:
        bdata = raw.get_data(picks=[channel], start=bs, stop=be)[0] * 1e6
        data = data - np.nanmean(bdata)
    return data


def process_subject(subject: str, set_path: Path, meta_sub: pd.DataFrame, target_delay_sec: float) -> Tuple[List[dict], dict]:
    raw, events = read_raw_events(set_path)
    sfreq = float(raw.info["sfreq"])
    condition_events = events[events["code"].isin(CONDITION_CODES)].copy().reset_index(drop=True)

    chans = {ch: get_channel(raw, ch) for ch in CHANNELS}
    n_trials = min(len(meta_sub), len(condition_events))

    accum: Dict[Tuple[str, str, str], dict] = {}
    trial_counts = CounterLike()

    for i in range(n_trials):
        m = meta_sub.iloc[i]
        cond = condition_events.iloc[i]
        cue_onset = float(cond["onset_sec"])
        target_onset = cue_onset + target_delay_sec
        responded = int(m["response_present_numeric"]) == 1 and pd.notna(m["rt_sec"])
        response_onset = target_onset + float(m["rt_sec"]) if responded else None
        outcome = str(m["outcome"])

        if outcome not in OUTCOME_ORDER:
            continue

        event_times = {"cue": cue_onset, "target": target_onset}
        if response_onset is not None:
            event_times["response"] = response_onset

        for lock, event_time in event_times.items():
            for ch_name, ch_actual in chans.items():
                if ch_actual is None:
                    continue
                seg = get_segment_uv(raw, ch_actual, float(event_time), lock)
                if seg is None:
                    continue
                key = (lock, ch_name, outcome)
                if key not in accum:
                    accum[key] = {"sum": np.zeros_like(seg, dtype=float), "count": 0}
                # All valid segments should have the same length for a given lock.
                if len(seg) == len(accum[key]["sum"]):
                    accum[key]["sum"] += seg
                    accum[key]["count"] += 1
                trial_counts.add((lock, outcome))

    rows = []
    for (lock, channel, outcome), obj in accum.items():
        count = int(obj["count"])
        if count <= 0:
            continue
        mean_wave = obj["sum"] / count
        times = standard_times(sfreq, lock)
        if len(times) != len(mean_wave):
            n = min(len(times), len(mean_wave))
            times = times[:n]
            mean_wave = mean_wave[:n]
        for t, amp in zip(times, mean_wave):
            rows.append({
                "subject_id": subject,
                "group": "high" if subject.startswith("H") else "low",
                "lock": lock,
                "channel": channel,
                "outcome": outcome,
                "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
                "time_sec": float(t),
                "time_ms": float(t * 1000.0),
                "amplitude_uv": float(amp),
                "n_trials": count,
            })

    summary = {
        "subject_id": subject,
        "group": "high" if subject.startswith("H") else "low",
        "set_file": set_path.name,
        "sfreq_hz": sfreq,
        "n_channels": int(len(raw.ch_names)),
        "n_metadata_trials": int(len(meta_sub)),
        "n_condition_events": int(len(condition_events)),
        "n_compared_trials": int(n_trials),
        "channels_found": ",".join([f"{k}:{v}" for k, v in chans.items()]),
    }
    for outcome in OUTCOME_ORDER:
        summary[f"n_{outcome}"] = int((meta_sub["outcome"] == outcome).sum())
    return rows, summary


class CounterLike:
    def __init__(self):
        self.d = defaultdict(int)
    def add(self, key):
        self.d[key] += 1


def compute_group_waveforms(subject_waveforms: pd.DataFrame) -> pd.DataFrame:
    if subject_waveforms.empty:
        return pd.DataFrame()
    group_cols = ["group", "lock", "channel", "outcome", "outcome_label", "time_sec", "time_ms"]
    g = subject_waveforms.groupby(group_cols)["amplitude_uv"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"])
    out = out.rename(columns={"mean": "mean_amplitude_uv", "std": "sd_amplitude_uv", "count": "n_subjects"})
    return out


def extract_window_value(wave_df: pd.DataFrame, spec: dict) -> float:
    d = wave_df[
        (wave_df["lock"] == spec["lock"]) &
        (wave_df["channel"] == spec["channel"]) &
        (wave_df["time_sec"] >= spec["t0"]) &
        (wave_df["time_sec"] <= spec["t1"])
    ]
    if d.empty:
        return np.nan
    x = pd.to_numeric(d["amplitude_uv"], errors="coerce").dropna()
    if x.empty:
        return np.nan
    if spec["measure"] == "mean":
        return float(x.mean())
    if spec["measure"] == "min":
        return float(x.min())
    if spec["measure"] == "max":
        return float(x.max())
    raise ValueError(f"Unknown measure: {spec['measure']}")


def compute_subject_window_features(subject_waveforms: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subject, group, outcome), d in subject_waveforms.groupby(["subject_id", "group", "outcome"]):
        row = {
            "subject_id": subject,
            "group": group,
            "outcome": outcome,
            "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
        }
        n_trials = d.groupby(["lock", "channel"])["n_trials"].max()
        for spec in WINDOW_SPECS:
            row[spec["name"]] = extract_window_value(d, spec)
            trial_key = (spec["lock"], spec["channel"])
            if trial_key in n_trials.index:
                row[f"{spec['name']}_n_trials"] = int(n_trials.loc[trial_key])
            else:
                row[f"{spec['name']}_n_trials"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


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


def contrast_window_stats(window_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for contrast in CONTRAST_SPECS:
        a = contrast["a"]
        b = contrast["b"]
        for spec in WINDOW_SPECS:
            if spec["lock"] not in contrast["locks"]:
                continue
            feat = spec["name"]
            wide = window_features.pivot_table(index=["subject_id", "group"], columns="outcome", values=feat, aggfunc="mean").reset_index()
            if a not in wide.columns or b not in wide.columns:
                continue
            wide["difference"] = wide[a] - wide[b]
            valid = wide.dropna(subset=["difference"]).copy()
            if len(valid) >= 10:
                vals = valid["difference"].astype(float).values
                t, p = stats.ttest_1samp(vals, 0.0)
                ci = stats.t.interval(0.95, df=len(vals)-1, loc=float(np.mean(vals)), scale=float(stats.sem(vals))) if len(vals) > 1 else (np.nan, np.nan)
                rows.append({
                    "contrast": contrast["name"],
                    "interpretation": contrast["interpretation"],
                    "window_feature": feat,
                    "lock": spec["lock"],
                    "channel": spec["channel"],
                    "measure": spec["measure"],
                    "n_subjects": int(len(vals)),
                    "mean_difference_uv": float(np.mean(vals)),
                    "ci95_low_uv": float(ci[0]),
                    "ci95_high_uv": float(ci[1]),
                    "t_value": float(t),
                    "p_value": float(p),
                    "test": "paired_one_sample_difference",
                })

            # group-specific contrast summaries
            for group, gd in valid.groupby("group"):
                if len(gd) >= 8:
                    vals = gd["difference"].astype(float).values
                    t, p = stats.ttest_1samp(vals, 0.0)
                    ci = stats.t.interval(0.95, df=len(vals)-1, loc=float(np.mean(vals)), scale=float(stats.sem(vals))) if len(vals) > 1 else (np.nan, np.nan)
                    rows.append({
                        "contrast": f"{contrast['name']}_{group}",
                        "interpretation": contrast["interpretation"],
                        "window_feature": feat,
                        "lock": spec["lock"],
                        "channel": spec["channel"],
                        "measure": spec["measure"],
                        "n_subjects": int(len(vals)),
                        "mean_difference_uv": float(np.mean(vals)),
                        "ci95_low_uv": float(ci[0]),
                        "ci95_high_uv": float(ci[1]),
                        "t_value": float(t),
                        "p_value": float(p),
                        "test": f"paired_one_sample_difference_within_{group}",
                    })

            # high-low difference of the contrast
            if set(valid["group"].dropna().unique()) >= {"high", "low"}:
                high = valid.loc[valid["group"] == "high", "difference"].astype(float).dropna().values
                low = valid.loc[valid["group"] == "low", "difference"].astype(float).dropna().values
                if len(high) >= 8 and len(low) >= 8:
                    t, p = stats.ttest_ind(high, low, equal_var=False)
                    rows.append({
                        "contrast": f"{contrast['name']}_high_minus_low",
                        "interpretation": f"group difference in {contrast['interpretation']}",
                        "window_feature": feat,
                        "lock": spec["lock"],
                        "channel": spec["channel"],
                        "measure": spec["measure"],
                        "n_subjects": int(len(high) + len(low)),
                        "mean_difference_uv": float(np.mean(high) - np.mean(low)),
                        "ci95_low_uv": np.nan,
                        "ci95_high_uv": np.nan,
                        "t_value": float(t),
                        "p_value": float(p),
                        "test": "welch_group_difference_of_contrast",
                    })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_fdr_bh"] = fdr_bh(out["p_value"].values)
        out = out.sort_values(["q_value_fdr_bh", "p_value"]).reset_index(drop=True)
    return out


def timepoint_contrast_stats(subject_waveforms: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    ranges = []

    for contrast in CONTRAST_SPECS:
        a = contrast["a"]
        b = contrast["b"]
        for lock in contrast["locks"]:
            for channel in CHANNELS:
                d = subject_waveforms[
                    (subject_waveforms["lock"] == lock) &
                    (subject_waveforms["channel"] == channel) &
                    (subject_waveforms["outcome"].isin([a, b]))
                ].copy()
                if d.empty:
                    continue
                wide = d.pivot_table(index=["subject_id", "group", "time_ms"], columns="outcome", values="amplitude_uv", aggfunc="mean").reset_index()
                if a not in wide.columns or b not in wide.columns:
                    continue
                wide["difference"] = wide[a] - wide[b]
                for time_ms, td in wide.groupby("time_ms"):
                    vals = td["difference"].dropna().astype(float).values
                    if len(vals) < 10:
                        continue
                    t, p = stats.ttest_1samp(vals, 0.0)
                    rows.append({
                        "contrast": contrast["name"],
                        "lock": lock,
                        "channel": channel,
                        "time_ms": float(time_ms),
                        "n_subjects": int(len(vals)),
                        "mean_difference_uv": float(np.mean(vals)),
                        "t_value": float(t),
                        "p_value": float(p),
                    })

    out = pd.DataFrame(rows)
    if out.empty:
        return out, pd.DataFrame()

    # FDR within each contrast-lock-channel series.
    q_all = []
    for keys, g in out.groupby(["contrast", "lock", "channel"]):
        g = g.copy()
        g["q_value_fdr_bh_within_series"] = fdr_bh(g["p_value"].values)
        q_all.append(g)
    out = pd.concat(q_all, ignore_index=True)
    out = out.sort_values(["contrast", "lock", "channel", "time_ms"]).reset_index(drop=True)

    # Convert significant timepoints to contiguous ranges.
    for keys, g in out.groupby(["contrast", "lock", "channel"]):
        sig = g[g["q_value_fdr_bh_within_series"] < 0.05].copy()
        if sig.empty:
            continue
        times = np.sort(sig["time_ms"].unique())
        if len(times) == 0:
            continue
        # Sampling step can have slight floating-point variation.
        diffs = np.diff(times)
        step = float(np.median(diffs)) if len(diffs) else 4.0
        start = times[0]
        prev = times[0]
        for t in times[1:]:
            if abs((t - prev) - step) > step * 1.5:
                if prev - start >= 20:
                    seg = g[(g["time_ms"] >= start) & (g["time_ms"] <= prev)]
                    ranges.append({
                        "contrast": keys[0],
                        "lock": keys[1],
                        "channel": keys[2],
                        "start_ms": float(start),
                        "end_ms": float(prev),
                        "duration_ms": float(prev - start),
                        "mean_difference_uv": float(seg["mean_difference_uv"].mean()),
                        "min_q": float(seg["q_value_fdr_bh_within_series"].min()),
                    })
                start = t
            prev = t
        if prev - start >= 20:
            seg = g[(g["time_ms"] >= start) & (g["time_ms"] <= prev)]
            ranges.append({
                "contrast": keys[0],
                "lock": keys[1],
                "channel": keys[2],
                "start_ms": float(start),
                "end_ms": float(prev),
                "duration_ms": float(prev - start),
                "mean_difference_uv": float(seg["mean_difference_uv"].mean()),
                "min_q": float(seg["q_value_fdr_bh_within_series"].min()),
            })

    return out, pd.DataFrame(ranges)


def make_figures(group_waveforms: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Outcome waveform figures by group, lock, channel.
    for lock in ["cue", "target", "response"]:
        for channel in CHANNELS:
            for group in ["high", "low"]:
                d = group_waveforms[
                    (group_waveforms["lock"] == lock) &
                    (group_waveforms["channel"] == channel) &
                    (group_waveforms["group"] == group)
                ].copy()
                if d.empty:
                    continue
                fig, ax = plt.subplots(figsize=(8, 4.8))
                for outcome in OUTCOME_ORDER:
                    od = d[d["outcome"] == outcome].sort_values("time_ms")
                    if od.empty:
                        continue
                    ax.plot(od["time_ms"], od["mean_amplitude_uv"], label=OUTCOME_LABELS.get(outcome, outcome), linewidth=1.8)
                    if od["sem"].notna().any():
                        ax.fill_between(
                            od["time_ms"].values,
                            (od["mean_amplitude_uv"] - od["sem"]).values,
                            (od["mean_amplitude_uv"] + od["sem"]).values,
                            alpha=0.15,
                        )
                ax.axvline(0, linestyle="--", linewidth=1)
                if lock == "cue":
                    ax.axvline(500, linestyle=":", linewidth=1)
                    ax.text(505, ax.get_ylim()[1] * 0.90, "target", fontsize=8)
                ax.set_title(f"{group.capitalize()} group — {lock}-locked ERP at {channel}")
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Amplitude (µV)")
                ax.legend(frameon=False, fontsize=8)
                fig.tight_layout()
                fig.savefig(fig_dir / f"erp_{group}_{lock}_{channel}_outcomes.png", dpi=200)
                plt.close(fig)

    # Difference wave figures.
    for contrast in CONTRAST_SPECS:
        for lock in contrast["locks"]:
            for channel in CHANNELS:
                fig, ax = plt.subplots(figsize=(8, 4.8))
                plotted = False
                for group in ["high", "low"]:
                    ga = group_waveforms[
                        (group_waveforms["lock"] == lock) &
                        (group_waveforms["channel"] == channel) &
                        (group_waveforms["group"] == group) &
                        (group_waveforms["outcome"] == contrast["a"])
                    ].sort_values("time_ms")
                    gb = group_waveforms[
                        (group_waveforms["lock"] == lock) &
                        (group_waveforms["channel"] == channel) &
                        (group_waveforms["group"] == group) &
                        (group_waveforms["outcome"] == contrast["b"])
                    ].sort_values("time_ms")
                    if ga.empty or gb.empty:
                        continue
                    merged = ga[["time_ms", "mean_amplitude_uv"]].merge(
                        gb[["time_ms", "mean_amplitude_uv"]],
                        on="time_ms",
                        suffixes=("_a", "_b"),
                    )
                    if merged.empty:
                        continue
                    merged["diff"] = merged["mean_amplitude_uv_a"] - merged["mean_amplitude_uv_b"]
                    ax.plot(merged["time_ms"], merged["diff"], label=group.capitalize(), linewidth=1.8)
                    plotted = True
                if not plotted:
                    plt.close(fig)
                    continue
                ax.axhline(0, linewidth=1)
                ax.axvline(0, linestyle="--", linewidth=1)
                if lock == "cue":
                    ax.axvline(500, linestyle=":", linewidth=1)
                ax.set_title(f"{contrast['name']} — {lock}-locked at {channel}")
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Difference amplitude (µV)")
                ax.legend(frameon=False)
                fig.tight_layout()
                fig.savefig(fig_dir / f"diff_{contrast['name']}_{lock}_{channel}.png", dpi=200)
                plt.close(fig)


def write_report(
    out_dir: Path,
    subject_summary: pd.DataFrame,
    window_stats: pd.DataFrame,
    sig_ranges: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# ERP waveform and difference-wave analysis report")
    lines.append("")
    lines.append("This analysis reconstructs subject-level ERP waveforms from the validated EEG event sequence and compares outcome-defined waveform families.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Subjects processed: {len(subject_summary)}")
    if not subject_summary.empty:
        lines.append(f"- Sampling rates: {sorted(subject_summary['sfreq_hz'].dropna().unique().tolist())}")
        lines.append(f"- Compared trials per subject: {int(subject_summary['n_compared_trials'].min())}–{int(subject_summary['n_compared_trials'].max())}")
        for outcome in OUTCOME_ORDER:
            col = f"n_{outcome}"
            if col in subject_summary.columns:
                lines.append(f"- {OUTCOME_LABELS[outcome]} trials per subject: {int(subject_summary[col].min())}–{int(subject_summary[col].max())}")
    lines.append("")
    lines.append("## Predefined window contrasts")
    lines.append("")
    lines.append(f"- Tested rows: {len(window_stats)}")
    lines.append(f"- FDR q < 0.05 rows: {int((window_stats['q_value_fdr_bh'] < 0.05).sum()) if not window_stats.empty else 0}")
    if not window_stats.empty:
        top = window_stats.head(20)
        lines.append("")
        lines.append("| Contrast | Window | Test | n | Mean difference (µV) | 95% CI | p | q |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for _, row in top.iterrows():
            lines.append(
                f"| {row['contrast']} | {row['window_feature']} | {row['test']} | {int(row['n_subjects'])} | "
                f"{row['mean_difference_uv']:.3f} | [{row['ci95_low_uv']:.3f}, {row['ci95_high_uv']:.3f}] | "
                f"{row['p_value']:.3g} | {row['q_value_fdr_bh']:.3g} |"
            )
    lines.append("")
    lines.append("## Time-resolved significant ranges")
    lines.append("")
    if sig_ranges.empty:
        lines.append("- No FDR-corrected time ranges of at least 20 ms were detected.")
    else:
        lines.append(f"- Significant ranges detected: {len(sig_ranges)}")
        lines.append("")
        lines.append("| Contrast | Lock | Channel | Time range (ms) | Mean difference (µV) | min q |")
        lines.append("|---|---|---|---:|---:|---:|")
        for _, row in sig_ranges.head(30).iterrows():
            lines.append(
                f"| {row['contrast']} | {row['lock']} | {row['channel']} | "
                f"{row['start_ms']:.0f} to {row['end_ms']:.0f} | {row['mean_difference_uv']:.3f} | {row['min_q']:.3g} |"
            )
    lines.append("")
    lines.append("## Interpretation guardrail")
    lines.append("")
    lines.append("These waveform analyses are designed to formalize ERP patterns visible in the original figures.")
    lines.append("Predefined window effects and FDR-corrected time ranges are more defensible than visual waveform separation alone.")
    lines.append("Use group-specific and time-resolved results to decide whether a neural-dynamics figure should be added to the manuscript.")
    lines.append("")
    (out_dir / "ERP_WAVEFORM_ANALYSIS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default=".")
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--eeg_root", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--target_delay_sec", type=float, default=0.5)
    args = parser.parse_args()

    base = Path(args.base).expanduser().resolve()
    metadata_path = (base / args.metadata).resolve() if not Path(args.metadata).is_absolute() else Path(args.metadata).resolve()
    eeg_root = (base / args.eeg_root).resolve() if not Path(args.eeg_root).is_absolute() else Path(args.eeg_root).resolve()
    out_dir = (base / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_path)
    set_by_subject = choose_preferred_set_files(eeg_root)

    all_rows = []
    summaries = []
    subjects = sorted(metadata["subject_id_norm"].dropna().unique().tolist())

    for subject in subjects:
        if subject not in set_by_subject:
            summaries.append({"subject_id": subject, "error": "missing_set_file"})
            print(f"{subject}: missing .set file")
            continue
        meta_sub = metadata[metadata["subject_id_norm"] == subject].copy()
        try:
            rows, summary = process_subject(subject, set_by_subject[subject], meta_sub, args.target_delay_sec)
            summary["error"] = ""
            all_rows.extend(rows)
            summaries.append(summary)
            print(f"{subject}: waveforms extracted ({len(rows)} rows)")
        except Exception as exc:
            summaries.append({"subject_id": subject, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{subject}: ERROR {type(exc).__name__}: {exc}")

    subject_waveforms = pd.DataFrame(all_rows)
    subject_summary = pd.DataFrame(summaries)
    group_waveforms = compute_group_waveforms(subject_waveforms)
    window_features = compute_subject_window_features(subject_waveforms)
    window_stats = contrast_window_stats(window_features)
    timepoint_stats, sig_ranges = timepoint_contrast_stats(subject_waveforms)

    subject_waveforms.to_csv(out_dir / "erp_subject_waveforms_long.csv", index=False)
    group_waveforms.to_csv(out_dir / "erp_group_waveforms_long.csv", index=False)
    subject_summary.to_csv(out_dir / "erp_subject_processing_summary.csv", index=False)
    window_features.to_csv(out_dir / "erp_subject_window_features.csv", index=False)
    window_stats.to_csv(out_dir / "erp_window_contrast_stats.csv", index=False)
    timepoint_stats.to_csv(out_dir / "erp_timepoint_contrast_stats.csv", index=False)
    sig_ranges.to_csv(out_dir / "erp_significant_time_ranges.csv", index=False)

    make_figures(group_waveforms, out_dir)
    write_report(out_dir, subject_summary, window_stats, sig_ranges)

    print(f"Wrote ERP waveform outputs to: {out_dir}")
    if not window_stats.empty:
        print(window_stats.head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
