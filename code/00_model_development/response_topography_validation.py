#!/usr/bin/env python3
# FORMALIZED HISTORICAL SOURCE SNAPSHOT
# Evidence status: HISTORICAL_OR_SUPERSEDED_DEVELOPMENT_EVIDENCE
# This file preserves a historical project stage with public-facing terminology standardized for release.
# The SHA-256 of the original executed/development source is recorded in the corresponding provenance registry.

"""
All-channel EEG topography analysis for the EEG-computational study.

Purpose
-------
This script tests whether the strongest neural effects have plausible scalp
topographies:
1. Response-locked false alarm vs Go hit:
   - ERN window: 50–150 ms after response
   - Pe window: 200–400 ms after response
   - theta power: 0–400 ms after response
   - beta power: 200–500 ms after response
2. Target-locked successful inhibition vs false alarm:
   - late positive / P300-like window: 400–470 ms after target
   - broader P300 window: 250–500 ms after target

Why useful
----------
Previous analyses showed strong ERN/Pe/theta/beta and decoding effects.
This script asks whether those effects are spatially plausible:
- ERN/theta should be more fronto-central.
- Pe/late positivity should be more centro-parietal.
- target late positive activity should often be centro-parietal.

Inputs
------
- EEGLAB .set/.fdt files under Groups_eventlistdefined
- trial-level metadata Excel

Outputs
-------
- eeg_topography_subject_features.csv
- eeg_topography_contrast_stats.csv
- eeg_topography_significant_channels.csv
- figures/topomap_*.png
- EEG_TOPOGRAPHY_REPORT.md

Example
-------
python code/final/run_eeg_topography_analysis_v1.py \
  --base . \
  --metadata data/raw/csv/ALL_SUBJECTS_metadata_features_merged_ORDERED.xlsx \
  --eeg_root "data/raw data/Groups_eventlistdefined" \
  --out results/eeg_topography_analysis_v1 \
  --min_trials 10
"""

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

ERP_FEATURE_SPECS = [
    {
        "feature": "response_ern_mean_50_150",
        "lock": "response",
        "t0": 0.050,
        "t1": 0.150,
        "b0": -0.200,
        "b1": 0.000,
        "measure": "mean",
        "contrast": "nogo_fa_minus_go_hit",
    },
    {
        "feature": "response_ern_min_50_150",
        "lock": "response",
        "t0": 0.050,
        "t1": 0.150,
        "b0": -0.200,
        "b1": 0.000,
        "measure": "min",
        "contrast": "nogo_fa_minus_go_hit",
    },
    {
        "feature": "response_pe_mean_200_400",
        "lock": "response",
        "t0": 0.200,
        "t1": 0.400,
        "b0": -0.200,
        "b1": 0.000,
        "measure": "mean",
        "contrast": "nogo_fa_minus_go_hit",
    },
    {
        "feature": "target_late_positive_mean_400_470",
        "lock": "target",
        "t0": 0.400,
        "t1": 0.470,
        "b0": -0.200,
        "b1": 0.000,
        "measure": "mean",
        "contrast": "nogo_cr_minus_nogo_fa",
    },
    {
        "feature": "target_p300_mean_250_500",
        "lock": "target",
        "t0": 0.250,
        "t1": 0.500,
        "b0": -0.200,
        "b1": 0.000,
        "measure": "mean",
        "contrast": "nogo_cr_minus_nogo_fa",
    },
]

TF_FEATURE_SPECS = [
    {
        "feature": "response_theta_logpower_0_400",
        "band": "theta",
        "freq": (4.0, 7.0),
        "lock": "response",
        "t0": 0.000,
        "t1": 0.400,
        "b0": -0.200,
        "b1": 0.000,
        "contrast": "nogo_fa_minus_go_hit",
    },
    {
        "feature": "response_beta_logpower_200_500",
        "band": "beta",
        "freq": (13.0, 30.0),
        "lock": "response",
        "t0": 0.200,
        "t1": 0.500,
        "b0": -0.200,
        "b1": 0.000,
        "contrast": "nogo_fa_minus_go_hit",
    },
]


CONTRAST_MAP = {
    "nogo_fa_minus_go_hit": ("nogo_fa", "go_hit"),
    "nogo_cr_minus_nogo_fa": ("nogo_cr", "nogo_fa"),
}


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
        raise ValueError(f"Metadata missing required columns: {missing}")

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


def read_raw_events(path: Path):
    import mne
    raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose="ERROR")
    events = []
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        events.append({"onset_sec": float(onset), "code": parse_event_code(desc)})
    ev = pd.DataFrame(events).sort_values("onset_sec").reset_index(drop=True)
    return raw, ev


def segment_values(data: np.ndarray, sfreq: float, event_time: float, t0: float, t1: float) -> Optional[np.ndarray]:
    start = int(round((event_time + t0) * sfreq))
    stop = int(round((event_time + t1) * sfreq))
    if start < 0 or stop > data.shape[-1] or stop <= start:
        return None
    return data[..., start:stop]


def summarize_segment(seg: np.ndarray, measure: str) -> np.ndarray:
    if measure == "mean":
        return np.nanmean(seg, axis=-1)
    if measure == "min":
        return np.nanmin(seg, axis=-1)
    if measure == "max":
        return np.nanmax(seg, axis=-1)
    raise ValueError(f"Unknown measure: {measure}")


def band_log_power_matrix(data_uv: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    nyq = sfreq / 2.0
    high = min(high, nyq - 1.0)
    sos = signal.butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
    filt = signal.sosfiltfilt(sos, data_uv, axis=-1)
    analytic = signal.hilbert(filt, axis=-1)
    power = np.abs(analytic) ** 2
    return np.log(power + 1e-12)


def process_subject(subject: str, set_path: Path, meta_sub: pd.DataFrame, target_delay_sec: float) -> Tuple[pd.DataFrame, dict]:
    raw, ev = read_raw_events(set_path)
    sfreq = float(raw.info["sfreq"])
    ch_names = list(raw.ch_names)
    data_uv = raw.get_data() * 1e6
    condition_events = ev[ev["code"].isin(CONDITION_CODES)].copy().reset_index(drop=True)

    n_trials = min(len(meta_sub), len(condition_events))

    # Precompute TF matrices needed.
    tf_cache = {}
    for spec in TF_FEATURE_SPECS:
        band = spec["band"]
        if band not in tf_cache:
            low, high = spec["freq"]
            tf_cache[band] = band_log_power_matrix(data_uv, sfreq, low, high)

    # Accumulate feature values by outcome, feature, channel.
    accum = defaultdict(list)

    for i in range(n_trials):
        m = meta_sub.iloc[i]
        cond = condition_events.iloc[i]

        outcome = str(m["outcome"])
        if outcome not in {"go_hit", "go_miss", "nogo_cr", "nogo_fa"}:
            continue

        cue_onset = float(cond["onset_sec"])
        target_onset = cue_onset + target_delay_sec
        responded = int(m["response_present_numeric"]) == 1 and pd.notna(m["rt_sec"])
        response_onset = target_onset + float(m["rt_sec"]) if responded else np.nan

        event_times = {
            "cue": cue_onset,
            "target": target_onset,
            "response": response_onset,
        }

        for spec in ERP_FEATURE_SPECS:
            et = event_times.get(spec["lock"], np.nan)
            if not np.isfinite(et):
                continue
            active = segment_values(data_uv, sfreq, et, spec["t0"], spec["t1"])
            base = segment_values(data_uv, sfreq, et, spec["b0"], spec["b1"])
            if active is None or base is None:
                continue
            active_summary = summarize_segment(active, spec["measure"])
            baseline_mean = np.nanmean(base, axis=-1)
            values = active_summary - baseline_mean
            for ch, val in zip(ch_names, values):
                accum[(outcome, spec["feature"], ch)].append(float(val))

        for spec in TF_FEATURE_SPECS:
            et = event_times.get(spec["lock"], np.nan)
            if not np.isfinite(et):
                continue
            mat = tf_cache[spec["band"]]
            active = segment_values(mat, sfreq, et, spec["t0"], spec["t1"])
            base = segment_values(mat, sfreq, et, spec["b0"], spec["b1"])
            if active is None or base is None:
                continue
            values = np.nanmean(active, axis=-1) - np.nanmean(base, axis=-1)
            for ch, val in zip(ch_names, values):
                accum[(outcome, spec["feature"], ch)].append(float(val))

    rows = []
    for (outcome, feature, channel), vals in accum.items():
        clean = np.asarray(vals, dtype=float)
        clean = clean[np.isfinite(clean)]
        rows.append({
            "subject_id": subject,
            "group": "high" if subject.startswith("H") else "low",
            "outcome": outcome,
            "feature": feature,
            "channel": channel,
            "n_trials": int(len(clean)),
            "mean_value": float(np.mean(clean)) if len(clean) else np.nan,
        })

    summary = {
        "subject_id": subject,
        "group": "high" if subject.startswith("H") else "low",
        "set_file": set_path.name,
        "sfreq_hz": sfreq,
        "n_channels": int(len(ch_names)),
        "channels": ",".join(ch_names),
        "n_metadata_trials": int(len(meta_sub)),
        "n_condition_events": int(len(condition_events)),
        "n_compared_trials": int(n_trials),
    }
    for outcome in ["go_hit", "go_miss", "nogo_cr", "nogo_fa"]:
        summary[f"n_{outcome}"] = int((meta_sub["outcome"] == outcome).sum())

    return pd.DataFrame(rows), summary


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    ok = np.isfinite(pvals)
    if ok.sum() == 0:
        return out
    p = pvals[ok]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    temp = np.empty(n)
    temp[order] = q
    out[ok] = temp
    return out


def contrast_stats(subject_features: pd.DataFrame, min_trials: int) -> pd.DataFrame:
    rows = []

    feature_to_contrast = {}
    for spec in ERP_FEATURE_SPECS:
        feature_to_contrast[spec["feature"]] = spec["contrast"]
    for spec in TF_FEATURE_SPECS:
        feature_to_contrast[spec["feature"]] = spec["contrast"]

    for feature, contrast in feature_to_contrast.items():
        a, b = CONTRAST_MAP[contrast]
        d = subject_features[subject_features["feature"] == feature].copy()

        wide_value = d.pivot_table(index=["subject_id", "group", "channel"], columns="outcome", values="mean_value", aggfunc="mean").reset_index()
        wide_n = d.pivot_table(index=["subject_id", "group", "channel"], columns="outcome", values="n_trials", aggfunc="mean").reset_index()

        if a not in wide_value.columns or b not in wide_value.columns:
            continue

        # Add n columns.
        wide = wide_value.copy()
        if a in wide_n.columns:
            wide[f"n_{a}"] = wide_n[a]
        else:
            wide[f"n_{a}"] = np.nan
        if b in wide_n.columns:
            wide[f"n_{b}"] = wide_n[b]
        else:
            wide[f"n_{b}"] = np.nan

        wide = wide[
            (pd.to_numeric(wide[f"n_{a}"], errors="coerce") >= min_trials) &
            (pd.to_numeric(wide[f"n_{b}"], errors="coerce") >= min_trials)
        ].copy()

        wide["difference"] = wide[a] - wide[b]

        for channel, g in wide.groupby("channel"):
            vals = pd.to_numeric(g["difference"], errors="coerce").dropna().values
            if len(vals) < 8:
                continue
            t, p = stats.ttest_1samp(vals, 0.0)
            ci = stats.t.interval(0.95, df=len(vals)-1, loc=float(np.mean(vals)), scale=float(stats.sem(vals))) if len(vals) > 1 else (np.nan, np.nan)
            rows.append({
                "contrast": contrast,
                "feature": feature,
                "channel": channel,
                "n_subjects": int(len(vals)),
                "mean_difference": float(np.mean(vals)),
                "ci95_low": float(ci[0]),
                "ci95_high": float(ci[1]),
                "t_value": float(t),
                "p_value": float(p),
                "positive_subjects": int(np.sum(vals > 0)),
                "negative_subjects": int(np.sum(vals < 0)),
                "min_trials": int(min_trials),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        # FDR within each feature.
        parts = []
        for feature, g in out.groupby("feature"):
            g = g.copy()
            g["q_value_fdr_bh_within_feature"] = fdr_bh(g["p_value"].values)
            parts.append(g)
        out = pd.concat(parts, ignore_index=True)
        out = out.sort_values(["feature", "q_value_fdr_bh_within_feature", "p_value"]).reset_index(drop=True)

    return out


def significant_channels(stats_df: pd.DataFrame) -> pd.DataFrame:
    if stats_df.empty:
        return pd.DataFrame()

    rows = []
    for feature, g in stats_df.groupby("feature"):
        sig = g[g["q_value_fdr_bh_within_feature"] < 0.05].copy()
        rows.append({
            "feature": feature,
            "contrast": g["contrast"].iloc[0],
            "n_channels_tested": int(g["channel"].nunique()),
            "n_channels_q_lt_05": int(sig["channel"].nunique()),
            "best_channel": g.sort_values("p_value").iloc[0]["channel"],
            "best_channel_mean_difference": float(g.sort_values("p_value").iloc[0]["mean_difference"]),
            "best_channel_p": float(g.sort_values("p_value").iloc[0]["p_value"]),
            "best_channel_q": float(g.sort_values("p_value").iloc[0]["q_value_fdr_bh_within_feature"]),
            "positive_channels": ",".join(sig.loc[sig["mean_difference"] > 0, "channel"].astype(str).tolist()),
            "negative_channels": ",".join(sig.loc[sig["mean_difference"] < 0, "channel"].astype(str).tolist()),
        })
    return pd.DataFrame(rows).sort_values(["n_channels_q_lt_05", "best_channel_q"], ascending=[False, True]).reset_index(drop=True)


def make_topomaps(stats_df: pd.DataFrame, out_dir: Path, ch_names: List[str]) -> None:
    if stats_df.empty:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Create standard montage info. Only channels with known positions will plot.
    montage = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore")

    for feature, g in stats_df.groupby("feature"):
        channel_to_value = dict(zip(g["channel"].astype(str), g["mean_difference"].astype(float)))
        values = np.array([channel_to_value.get(ch, np.nan) for ch in ch_names], dtype=float)
        if np.isfinite(values).sum() < 4:
            continue

        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        im, cn = mne.viz.plot_topomap(
            values,
            info,
            axes=ax,
            show=False,
            contours=6,
            names=None,
            sensors=True,
        )
        cbar = plt.colorbar(im, ax=ax, shrink=0.75)
        cbar.set_label("Mean contrast difference")
        contrast = g["contrast"].iloc[0]
        ax.set_title(f"{feature}\n{contrast}")
        fig.tight_layout()
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", feature)
        fig.savefig(fig_dir / f"topomap_{safe_name}.png", dpi=220)
        plt.close(fig)


def write_report(out_dir: Path, summaries: pd.DataFrame, stats_df: pd.DataFrame, sig_df: pd.DataFrame, min_trials: int) -> None:
    lines = []
    lines.append("# EEG topography analysis report")
    lines.append("")
    lines.append("This report maps the scalp distribution of the strongest ERP/time-frequency effects.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Subjects processed: {len(summaries)}")
    lines.append(f"- Minimum trials per condition for contrasts: {min_trials}")
    if not summaries.empty:
        lines.append(f"- Channels per subject: {int(summaries['n_channels'].min())}–{int(summaries['n_channels'].max())}")
        for outcome in ["go_hit", "go_miss", "nogo_cr", "nogo_fa"]:
            col = f"n_{outcome}"
            if col in summaries.columns:
                lines.append(f"- {outcome} trials per subject: {int(summaries[col].min())}–{int(summaries[col].max())}")
    lines.append("")
    lines.append("## Significant-channel summary")
    lines.append("")
    if sig_df.empty:
        lines.append("- No feature summaries were generated.")
    else:
        lines.append("| Feature | Contrast | channels q<.05 | best channel | best mean diff | best q |")
        lines.append("|---|---|---:|---|---:|---:|")
        for _, row in sig_df.iterrows():
            lines.append(
                f"| {row['feature']} | {row['contrast']} | {int(row['n_channels_q_lt_05'])} | "
                f"{row['best_channel']} | {row['best_channel_mean_difference']:.4f} | {row['best_channel_q']:.3g} |"
            )
    lines.append("")
    lines.append("## Top channel-level effects")
    lines.append("")
    if stats_df.empty:
        lines.append("- No channel-level tests were generated.")
    else:
        lines.append("| Feature | Contrast | Channel | n | Mean difference | 95% CI | p | q |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for _, row in stats_df.sort_values(["q_value_fdr_bh_within_feature", "p_value"]).head(40).iterrows():
            lines.append(
                f"| {row['feature']} | {row['contrast']} | {row['channel']} | {int(row['n_subjects'])} | "
                f"{row['mean_difference']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
                f"{row['p_value']:.3g} | {row['q_value_fdr_bh_within_feature']:.3g} |"
            )
    lines.append("")
    lines.append("## Interpretation guardrail")
    lines.append("")
    lines.append("- Topography supports anatomical/plausibility checks; it should not be treated as source localization.")
    lines.append("- Use the maps to support claims such as fronto-central error monitoring or centro-parietal late positivity only when the spatial pattern is coherent.")
    lines.append("- If topographies are noisy or dominated by isolated channels, keep them in Supplementary.")
    lines.append("")
    (out_dir / "EEG_TOPOGRAPHY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default=".")
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--eeg_root", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--target_delay_sec", type=float, default=0.5)
    parser.add_argument("--min_trials", type=int, default=10)
    args = parser.parse_args()

    base = Path(args.base).expanduser().resolve()
    metadata_path = (base / args.metadata).resolve() if not Path(args.metadata).is_absolute() else Path(args.metadata).resolve()
    eeg_root = (base / args.eeg_root).resolve() if not Path(args.eeg_root).is_absolute() else Path(args.eeg_root).resolve()
    out_dir = (base / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_path)
    set_by_subject = choose_preferred_set_files(eeg_root)

    all_features = []
    summaries = []
    ch_names_master = None

    for subject in sorted(metadata["subject_id_norm"].dropna().unique()):
        if subject not in set_by_subject:
            print(f"{subject}: missing .set")
            continue

        meta_sub = metadata[metadata["subject_id_norm"] == subject].copy()
        try:
            feats, summ = process_subject(subject, set_by_subject[subject], meta_sub, args.target_delay_sec)
            all_features.append(feats)
            summaries.append(summ)
            if ch_names_master is None and "channels" in summ:
                ch_names_master = summ["channels"].split(",")
            print(f"{subject}: extracted topography features ({len(feats)} rows)")
        except Exception as exc:
            summaries.append({
                "subject_id": subject,
                "group": "high" if subject.startswith("H") else "low",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"{subject}: ERROR {type(exc).__name__}: {exc}")

    subject_features = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()
    summaries_df = pd.DataFrame(summaries)
    stats_df = contrast_stats(subject_features, min_trials=int(args.min_trials)) if not subject_features.empty else pd.DataFrame()
    sig_df = significant_channels(stats_df) if not stats_df.empty else pd.DataFrame()

    subject_features.to_csv(out_dir / "eeg_topography_subject_features.csv", index=False)
    summaries_df.to_csv(out_dir / "eeg_topography_subject_processing_summary.csv", index=False)
    stats_df.to_csv(out_dir / "eeg_topography_contrast_stats.csv", index=False)
    sig_df.to_csv(out_dir / "eeg_topography_significant_channels.csv", index=False)

    if ch_names_master is None and not subject_features.empty:
        ch_names_master = sorted(subject_features["channel"].dropna().unique().tolist())

    if ch_names_master:
        make_topomaps(stats_df, out_dir, ch_names_master)

    write_report(out_dir, summaries_df, stats_df, sig_df, int(args.min_trials))

    print(f"Wrote outputs to: {out_dir}")
    if not sig_df.empty:
        print(sig_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
