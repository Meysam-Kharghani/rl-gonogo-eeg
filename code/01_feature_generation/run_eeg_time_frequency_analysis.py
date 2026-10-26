#!/usr/bin/env python3
"""Trial-wise EEG time-frequency feature extraction for EEG-computational study.

The program derives band-power features from processed EEGLAB datasets using
band-pass filtering, forward-backward zero-phase filtering, and Hilbert
amplitude envelopes. Event codes 1-6 define cue conditions, target onset is
represented as cue onset plus 0.5 s, and response onset is reconstructed from
behavioral response time.

The output includes trial-level time-frequency features and aggregate summary
tables. Optional computational latent inputs support downstream association
summaries and are not required for feature extraction."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal, stats


CONDITION_CODES = {"1", "2", "3", "4", "5", "6"}

BANDS = {
    "theta": (4.0, 7.0),
    "alpha": (8.0, 12.0),
    "mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# Feature specification:
# lock: cue, target, response
# baseline is relative to the same event unless otherwise noted.
FEATURE_SPECS = [
    # Cue/preparatory control
    {"name": "cue_theta_fz_200_500", "band": "theta", "channel": "Fz", "lock": "cue", "t0": 0.200, "t1": 0.500, "b0": -0.200, "b1": 0.000},
    {"name": "cue_theta_cz_200_500", "band": "theta", "channel": "Cz", "lock": "cue", "t0": 0.200, "t1": 0.500, "b0": -0.200, "b1": 0.000},
    {"name": "cue_alpha_pz_200_500", "band": "alpha", "channel": "Pz", "lock": "cue", "t0": 0.200, "t1": 0.500, "b0": -0.200, "b1": 0.000},
    {"name": "cue_beta_cz_200_500", "band": "beta", "channel": "Cz", "lock": "cue", "t0": 0.200, "t1": 0.500, "b0": -0.200, "b1": 0.000},

    # Target processing/conflict
    {"name": "target_theta_fz_200_500", "band": "theta", "channel": "Fz", "lock": "target", "t0": 0.200, "t1": 0.500, "b0": -0.200, "b1": 0.000},
    {"name": "target_theta_cz_200_500", "band": "theta", "channel": "Cz", "lock": "target", "t0": 0.200, "t1": 0.500, "b0": -0.200, "b1": 0.000},
    {"name": "target_alpha_pz_250_500", "band": "alpha", "channel": "Pz", "lock": "target", "t0": 0.250, "t1": 0.500, "b0": -0.200, "b1": 0.000},
    {"name": "target_beta_cz_250_500", "band": "beta", "channel": "Cz", "lock": "target", "t0": 0.250, "t1": 0.500, "b0": -0.200, "b1": 0.000},

    # Response preparation / motor execution
    {"name": "resp_mu_c3_m400_0", "band": "mu", "channel": "C3", "lock": "response", "t0": -0.400, "t1": 0.000, "b0": -0.700, "b1": -0.500},
    {"name": "resp_mu_cz_m400_0", "band": "mu", "channel": "Cz", "lock": "response", "t0": -0.400, "t1": 0.000, "b0": -0.700, "b1": -0.500},
    {"name": "resp_mu_c4_m400_0", "band": "mu", "channel": "C4", "lock": "response", "t0": -0.400, "t1": 0.000, "b0": -0.700, "b1": -0.500},
    {"name": "resp_beta_c3_m400_0", "band": "beta", "channel": "C3", "lock": "response", "t0": -0.400, "t1": 0.000, "b0": -0.700, "b1": -0.500},
    {"name": "resp_beta_cz_m400_0", "band": "beta", "channel": "Cz", "lock": "response", "t0": -0.400, "t1": 0.000, "b0": -0.700, "b1": -0.500},
    {"name": "resp_beta_c4_m400_0", "band": "beta", "channel": "C4", "lock": "response", "t0": -0.400, "t1": 0.000, "b0": -0.700, "b1": -0.500},

    # Performance monitoring and post-response processing
    {"name": "resp_theta_fz_0_200", "band": "theta", "channel": "Fz", "lock": "response", "t0": 0.000, "t1": 0.200, "b0": -0.200, "b1": 0.000},
    {"name": "resp_theta_fz_0_400", "band": "theta", "channel": "Fz", "lock": "response", "t0": 0.000, "t1": 0.400, "b0": -0.200, "b1": 0.000},
    {"name": "resp_theta_cz_0_400", "band": "theta", "channel": "Cz", "lock": "response", "t0": 0.000, "t1": 0.400, "b0": -0.200, "b1": 0.000},
    {"name": "resp_beta_cz_200_500", "band": "beta", "channel": "Cz", "lock": "response", "t0": 0.200, "t1": 0.500, "b0": -0.200, "b1": 0.000},
]

CONTRASTS = [
    {"name": "nogo_fa_minus_go_hit", "a": "nogo_fa", "b": "go_hit", "features": [
        "resp_theta_fz_0_200", "resp_theta_fz_0_400", "resp_theta_cz_0_400",
        "resp_mu_c3_m400_0", "resp_mu_cz_m400_0", "resp_mu_c4_m400_0",
        "resp_beta_c3_m400_0", "resp_beta_cz_m400_0", "resp_beta_c4_m400_0",
        "resp_beta_cz_200_500",
    ]},
    {"name": "nogo_cr_minus_nogo_fa", "a": "nogo_cr", "b": "nogo_fa", "features": [
        "target_theta_fz_200_500", "target_theta_cz_200_500",
        "target_alpha_pz_250_500", "target_beta_cz_250_500",
    ]},
    {"name": "go_hit_minus_go_miss", "a": "go_hit", "b": "go_miss", "features": [
        "cue_theta_fz_200_500", "cue_theta_cz_200_500",
        "cue_alpha_pz_200_500", "cue_beta_cz_200_500",
        "target_theta_fz_200_500", "target_alpha_pz_250_500",
    ]},
]

ENCODING_TESTS = [
    {"latent": "control_current", "features": ["cue_theta_fz_200_500", "cue_theta_cz_200_500", "cue_alpha_pz_200_500", "cue_beta_cz_200_500"], "subset": "all"},
    {"latent": "p_press", "features": ["resp_mu_c3_m400_0", "resp_mu_cz_m400_0", "resp_mu_c4_m400_0", "resp_beta_c3_m400_0", "resp_beta_cz_m400_0", "resp_beta_c4_m400_0"], "subset": "response"},
    {"latent": "actor_qdiff_current", "features": ["target_theta_fz_200_500", "target_theta_cz_200_500", "target_alpha_pz_250_500", "target_beta_cz_250_500"], "subset": "all"},
]


def normalize_subject_id(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    m = re.search(r"([HLhl])\s*0*([0-9]+)", s)
    if m:
        return f"{m.group(1).upper()}{int(m.group(2))}"
    return s.upper()


def norm_text(x: object) -> str:
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


def infer_subject_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        if re.fullmatch(r"[HLhl]\d+", part.strip()):
            return normalize_subject_id(part)
    m = re.search(r"([HLhl]\d+)", path.name)
    return normalize_subject_id(m.group(1)) if m else ""


def choose_preferred_set_files(eeg_root: Path) -> Dict[str, Path]:
    files = sorted(eeg_root.rglob("*.set"))
    by_subject = defaultdict(list)
    for p in files:
        subject = infer_subject_from_path(p)
        if subject:
            by_subject[subject].append(p)

    selected = {}
    for subject, candidates in sorted(by_subject.items()):
        def rank(p: Path):
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
        raise ValueError(f"Metadata table missing columns: {missing}")

    out = df.copy()
    out["subject_id_norm"] = out["subject_id"].map(normalize_subject_id)
    out["trial_index_numeric"] = pd.to_numeric(out["trial_index"], errors="coerce")
    out["trial_class_norm"] = out["trial_class"].map(norm_text)
    out["cue_type_norm"] = out["cue_type"].map(norm_text)
    out["response_present_numeric"] = pd.to_numeric(out["response_present"], errors="coerce").fillna(0).astype(int)
    out["rt_sec"] = pd.to_numeric(out["rt_response_from_target_ms"], errors="coerce") / 1000.0
    if "response_type" in out.columns:
        out["response_type_norm"] = out["response_type"].map(norm_text)
    else:
        out["response_type_norm"] = ""

    def outcome(row):
        rtype = row["response_type_norm"]
        tc = row["trial_class_norm"]
        press = int(row["response_present_numeric"])
        rt_ok = pd.notna(row["rt_sec"])
        if rtype in {"hit", "go_hit"}:
            return "go_hit"
        if rtype in {"miss", "go_miss"}:
            return "go_miss"
        if rtype in {"cr", "correct rejection", "correct_rejection", "nogo_cr"}:
            return "nogo_cr"
        if rtype in {"fa", "false alarm", "false_alarm", "nogo_fa"}:
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


def load_latents(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    if "subject_id" in df.columns:
        df["subject_id_norm"] = df["subject_id"].map(normalize_subject_id)
    elif "subj" in df.columns:
        df["subject_id_norm"] = df["subj"].map(normalize_subject_id)
    else:
        raise ValueError("Latent file needs subject_id or subj.")

    if "trial_index" in df.columns:
        df["latent_trial_index_numeric"] = pd.to_numeric(df["trial_index"], errors="coerce")
    elif "trial" in df.columns:
        df["latent_trial_index_numeric"] = pd.to_numeric(df["trial"], errors="coerce")
    else:
        raise ValueError("Latent file needs trial_index or trial.")

    keep = ["subject_id_norm", "latent_trial_index_numeric"]
    for c in ["p_press", "control_current", "actor_qdiff_current", "belief_nogo_current", "associability_current", "is_nogo", "action_press"]:
        if c in df.columns:
            keep.append(c)
    return df[keep].copy()


def read_raw_events(path: Path):
    import mne
    raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose="ERROR")
    events = []
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        events.append({"onset_sec": float(onset), "code": parse_event_code(desc)})
    ev = pd.DataFrame(events).sort_values("onset_sec").reset_index(drop=True)
    return raw, ev


def get_channel(raw, name: str) -> Optional[str]:
    lower = {ch.lower(): ch for ch in raw.ch_names}
    return lower.get(name.lower())


def band_log_power(data_uv: np.ndarray, sfreq: float, band: Tuple[float, float]) -> np.ndarray:
    low, high = band
    nyq = sfreq / 2.0
    high = min(high, nyq - 1.0)
    sos = signal.butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
    filtered = signal.sosfiltfilt(sos, data_uv)
    analytic = signal.hilbert(filtered)
    power = np.abs(analytic) ** 2
    return np.log(power + 1e-12)


def segment_mean(arr: np.ndarray, sfreq: float, event_time: float, t0: float, t1: float) -> float:
    start = int(round((event_time + t0) * sfreq))
    stop = int(round((event_time + t1) * sfreq))
    if start < 0 or stop > len(arr) or stop <= start:
        return np.nan
    return float(np.nanmean(arr[start:stop]))


def process_subject(subject: str, set_path: Path, meta_sub: pd.DataFrame, target_delay_sec: float) -> Tuple[pd.DataFrame, dict]:
    raw, ev = read_raw_events(set_path)
    sfreq = float(raw.info["sfreq"])
    condition_events = ev[ev["code"].isin(CONDITION_CODES)].copy().reset_index(drop=True)

    # Precompute log power series for every required channel-band pair.
    required_pairs = sorted({(spec["channel"], spec["band"]) for spec in FEATURE_SPECS})
    power_cache: Dict[Tuple[str, str], np.ndarray] = {}
    missing_channels = []

    for ch_name, band_name in required_pairs:
        ch = get_channel(raw, ch_name)
        if ch is None:
            missing_channels.append(ch_name)
            continue
        data_uv = raw.get_data(picks=[ch])[0] * 1e6
        power_cache[(ch_name, band_name)] = band_log_power(data_uv, sfreq, BANDS[band_name])

    rows = []
    n_trials = min(len(meta_sub), len(condition_events))

    for i in range(n_trials):
        m = meta_sub.iloc[i]
        cond = condition_events.iloc[i]
        cue_onset = float(cond["onset_sec"])
        target_onset = cue_onset + target_delay_sec
        responded = int(m["response_present_numeric"]) == 1 and pd.notna(m["rt_sec"])
        response_onset = target_onset + float(m["rt_sec"]) if responded else np.nan

        event_times = {
            "cue": cue_onset,
            "target": target_onset,
            "response": response_onset,
        }

        row = {
            "subject_id": subject,
            "subject_id_norm": subject,
            "group": "high" if subject.startswith("H") else "low",
            "trial_index": m["trial_index"],
            "trial_index_numeric": pd.to_numeric(m["trial_index"], errors="coerce"),
            "cue_type": m["cue_type_norm"],
            "trial_class": m["trial_class_norm"],
            "response_present": int(m["response_present_numeric"]),
            "response_type": m["response_type_norm"],
            "outcome": m["outcome"],
            "rt_response_from_target_ms": m["rt_response_from_target_ms"],
        }

        for spec in FEATURE_SPECS:
            key = (spec["channel"], spec["band"])
            arr = power_cache.get(key)
            et = event_times.get(spec["lock"], np.nan)
            if arr is None or not np.isfinite(et):
                row[spec["name"]] = np.nan
                continue
            active = segment_mean(arr, sfreq, et, spec["t0"], spec["t1"])
            base = segment_mean(arr, sfreq, et, spec["b0"], spec["b1"])
            row[spec["name"]] = active - base if np.isfinite(active) and np.isfinite(base) else np.nan

        rows.append(row)

    summary = {
        "subject_id": subject,
        "group": "high" if subject.startswith("H") else "low",
        "set_file": set_path.name,
        "sfreq_hz": sfreq,
        "n_compared_trials": int(n_trials),
        "missing_channels": ",".join(sorted(set(missing_channels))),
    }
    for outcome in ["go_hit", "go_miss", "nogo_cr", "nogo_fa"]:
        summary[f"n_{outcome}"] = int((meta_sub["outcome"] == outcome).sum())

    return pd.DataFrame(rows), summary


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


def contrast_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for contrast in CONTRASTS:
        a, b = contrast["a"], contrast["b"]
        for feature in contrast["features"]:
            if feature not in df.columns:
                continue
            wide = df.pivot_table(index=["subject_id", "group"], columns="outcome", values=feature, aggfunc="mean").reset_index()
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
                    "feature": feature,
                    "test": "paired_subject_difference",
                    "n_subjects": int(len(vals)),
                    "mean_difference_log_power": float(np.mean(vals)),
                    "ci95_low": float(ci[0]),
                    "ci95_high": float(ci[1]),
                    "t_value": float(t),
                    "p_value": float(p),
                })

            for group, gd in valid.groupby("group"):
                if len(gd) >= 8:
                    vals = gd["difference"].astype(float).values
                    t, p = stats.ttest_1samp(vals, 0.0)
                    ci = stats.t.interval(0.95, df=len(vals)-1, loc=float(np.mean(vals)), scale=float(stats.sem(vals))) if len(vals) > 1 else (np.nan, np.nan)
                    rows.append({
                        "contrast": f"{contrast['name']}_{group}",
                        "feature": feature,
                        "test": f"paired_subject_difference_within_{group}",
                        "n_subjects": int(len(vals)),
                        "mean_difference_log_power": float(np.mean(vals)),
                        "ci95_low": float(ci[0]),
                        "ci95_high": float(ci[1]),
                        "t_value": float(t),
                        "p_value": float(p),
                    })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_fdr_bh"] = fdr_bh(out["p_value"].values)
        out = out.sort_values(["q_value_fdr_bh", "p_value"]).reset_index(drop=True)
    return out


def choose_latent_offset(features: pd.DataFrame, latents: pd.DataFrame) -> int:
    candidates = [-2, -1, 0, 1, 2]
    rows = []
    for off in candidates:
        tmp = latents.copy()
        tmp["trial_index_numeric"] = tmp["latent_trial_index_numeric"] + off
        merged = features[["subject_id_norm", "trial_index_numeric", "response_present", "trial_class"]].merge(
            tmp.drop(columns=["latent_trial_index_numeric"]),
            on=["subject_id_norm", "trial_index_numeric"],
            how="left",
        )
        scores = []
        matched = int(merged["p_press"].notna().sum()) if "p_press" in merged.columns else 0
        if "action_press" in merged.columns:
            m = merged["action_press"].notna()
            if m.any():
                scores.append((
                    pd.to_numeric(merged.loc[m, "response_present"], errors="coerce").astype(int).values
                    == pd.to_numeric(merged.loc[m, "action_press"], errors="coerce").astype(int).values
                ).mean())
        if "is_nogo" in merged.columns:
            m = merged["is_nogo"].notna()
            if m.any():
                scores.append((
                    (merged.loc[m, "trial_class"].map(norm_text) == "nogo").astype(int).values
                    == pd.to_numeric(merged.loc[m, "is_nogo"], errors="coerce").astype(int).values
                ).mean())
        rows.append({"offset": off, "matched": matched, "score": float(np.mean(scores)) if scores else np.nan})
    score_df = pd.DataFrame(rows).sort_values(["score", "matched"], ascending=False)
    return int(score_df.iloc[0]["offset"])


def merge_latents(features: pd.DataFrame, latents: Optional[pd.DataFrame]) -> pd.DataFrame:
    if latents is None:
        return features
    offset = choose_latent_offset(features, latents)
    tmp = latents.copy()
    tmp["trial_index_numeric"] = tmp["latent_trial_index_numeric"] + offset
    tmp = tmp.drop(columns=["latent_trial_index_numeric"])
    merged = features.merge(tmp, on=["subject_id_norm", "trial_index_numeric"], how="left")
    merged["latent_trial_index_offset_used"] = offset
    return merged


def subject_level_encoding(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for test in ENCODING_TESTS:
        latent = test["latent"]
        if latent not in df.columns:
            continue
        subset = test["subset"]
        d = df.copy()
        if subset == "response":
            d = d[pd.to_numeric(d["response_present"], errors="coerce") == 1].copy()

        for feature in test["features"]:
            if feature not in d.columns:
                continue
            rs = []
            for subject, g in d.groupby("subject_id"):
                x = pd.to_numeric(g[latent], errors="coerce")
                y = pd.to_numeric(g[feature], errors="coerce")
                mask = np.isfinite(x) & np.isfinite(y)
                if int(mask.sum()) < 25:
                    continue
                if np.nanstd(x[mask]) <= 1e-12 or np.nanstd(y[mask]) <= 1e-12:
                    continue
                r, _ = stats.pearsonr(x[mask], y[mask])
                rs.append(np.arctanh(np.clip(float(r), -0.999999, 0.999999)))
            if len(rs) >= 10:
                z = np.array(rs)
                t, p = stats.ttest_1samp(z, 0.0)
                ci = stats.t.interval(0.95, df=len(z)-1, loc=float(np.mean(z)), scale=float(stats.sem(z))) if len(z) > 1 else (np.nan, np.nan)
                rows.append({
                    "latent": latent,
                    "feature": feature,
                    "subset": subset,
                    "n_subjects": int(len(z)),
                    "mean_r": float(np.tanh(np.mean(z))),
                    "ci95_low_r": float(np.tanh(ci[0])) if np.isfinite(ci[0]) else np.nan,
                    "ci95_high_r": float(np.tanh(ci[1])) if np.isfinite(ci[1]) else np.nan,
                    "t_value": float(t),
                    "p_value": float(p),
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_fdr_bh"] = fdr_bh(out["p_value"].values)
        out = out.sort_values(["q_value_fdr_bh", "p_value"]).reset_index(drop=True)
    return out


def summary_by_subject(df: pd.DataFrame) -> pd.DataFrame:
    feature_names = [spec["name"] for spec in FEATURE_SPECS]
    rows = []
    for (subject, group), g in df.groupby(["subject_id", "group"]):
        row = {"subject_id": subject, "group": group, "n_trials": int(len(g))}
        for outcome in ["go_hit", "go_miss", "nogo_cr", "nogo_fa"]:
            row[f"n_{outcome}"] = int((g["outcome"] == outcome).sum())
        for feat in feature_names:
            if feat in g.columns:
                row[f"{feat}_mean"] = float(pd.to_numeric(g[feat], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(out_dir: Path, features: pd.DataFrame, subj_summary: pd.DataFrame, contrasts: pd.DataFrame, encoding: pd.DataFrame, latents_supplied: bool) -> None:
    lines = []
    lines.append("# EEG time-frequency neural-dynamics report")
    lines.append("")
    lines.append("This analysis extracts baseline-corrected log band-power features using band-pass filtering and Hilbert envelopes.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Trial rows: {len(features)}")
    lines.append(f"- Subjects: {features['subject_id'].nunique() if not features.empty else 0}")
    if "latent_trial_index_offset_used" in features.columns:
        vals = sorted(features["latent_trial_index_offset_used"].dropna().unique().tolist())
        lines.append(f"- Latent trial-index offset used: {vals}")
    lines.append("")
    lines.append("## Band-power contrasts")
    lines.append("")
    lines.append(f"- Tested contrast rows: {len(contrasts)}")
    lines.append(f"- FDR q < 0.05 rows: {int((contrasts['q_value_fdr_bh'] < 0.05).sum()) if not contrasts.empty else 0}")
    if not contrasts.empty:
        lines.append("")
        lines.append("| Contrast | Feature | n | Mean log-power difference | 95% CI | p | q |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for _, row in contrasts.head(20).iterrows():
            lines.append(
                f"| {row['contrast']} | {row['feature']} | {int(row['n_subjects'])} | "
                f"{row['mean_difference_log_power']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
                f"{row['p_value']:.3g} | {row['q_value_fdr_bh']:.3g} |"
            )
    lines.append("")
    lines.append("## Computational-state encoding")
    lines.append("")
    if not latents_supplied:
        lines.append("- No latent file supplied; encoding screen was skipped.")
    else:
        lines.append(f"- Tested encoding rows: {len(encoding)}")
        lines.append(f"- FDR q < 0.05 rows: {int((encoding['q_value_fdr_bh'] < 0.05).sum()) if not encoding.empty else 0}")
        if not encoding.empty:
            lines.append("")
            lines.append("| Latent | Feature | subset | mean r | 95% CI | p | q |")
            lines.append("|---|---|---|---:|---:|---:|---:|")
            for _, row in encoding.head(20).iterrows():
                lines.append(
                    f"| {row['latent']} | {row['feature']} | {row['subset']} | "
                    f"{row['mean_r']:.4f} | [{row['ci95_low_r']:.4f}, {row['ci95_high_r']:.4f}] | "
                    f"{row['p_value']:.3g} | {row['q_value_fdr_bh']:.3g} |"
                )
    lines.append("")
    lines.append("## Interpretation constraints")
    lines.append("")
    lines.append("This targeted time-frequency analysis is interpreted using participant-level inference and FDR correction.")
    lines.append("Reported effects characterize neural dynamics associated with control, motor preparation, or performance monitoring without implying causal mechanisms.")
    lines.append("")
    (out_dir / "TF_NEURAL_DYNAMICS_REPORT.md").write_text("\\n".join(lines), encoding="utf-8")


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
    set_by_subject = choose_preferred_set_files(eeg_root)

    feature_tables = []
    subject_summaries = []

    for subject in sorted(metadata["subject_id_norm"].dropna().unique()):
        if subject not in set_by_subject:
            print(f"{subject}: missing .set")
            continue
        meta_sub = metadata[metadata["subject_id_norm"] == subject].copy()
        try:
            feats, summ = process_subject(subject, set_by_subject[subject], meta_sub, args.target_delay_sec)
            feature_tables.append(feats)
            subject_summaries.append(summ)
            print(f"{subject}: extracted {len(feats)} trials")
        except Exception as exc:
            subject_summaries.append({"subject_id": subject, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{subject}: ERROR {type(exc).__name__}: {exc}")

    features = pd.concat(feature_tables, ignore_index=True) if feature_tables else pd.DataFrame()
    features = merge_latents(features, latents)
    subj_summary = pd.DataFrame(subject_summaries)
    band_summary = summary_by_subject(features) if not features.empty else pd.DataFrame()
    contrasts = contrast_stats(features) if not features.empty else pd.DataFrame()
    encoding = subject_level_encoding(features) if latents is not None and not features.empty else pd.DataFrame()

    features.to_csv(out_dir / "tf_trial_bandpower_features.csv", index=False)
    subj_summary.to_csv(out_dir / "tf_subject_processing_summary.csv", index=False)
    band_summary.to_csv(out_dir / "tf_subject_bandpower_summary.csv", index=False)
    contrasts.to_csv(out_dir / "tf_contrast_stats.csv", index=False)
    encoding.to_csv(out_dir / "tf_computational_encoding_subject_level.csv", index=False)
    write_report(out_dir, features, subj_summary, contrasts, encoding, latents is not None)

    print(f"Wrote outputs to: {out_dir}")
    if not contrasts.empty:
        print(contrasts.head(15).to_string(index=False))
    if not encoding.empty:
        print(encoding.head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
