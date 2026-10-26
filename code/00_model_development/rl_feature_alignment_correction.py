#!/usr/bin/env python3
# FORMALIZED HISTORICAL SOURCE SNAPSHOT
# Evidence status: HISTORICAL_OR_SUPERSEDED_DEVELOPMENT_EVIDENCE
# This file preserves a historical project stage with public-facing terminology standardized for release.
# The SHA-256 of the original executed/development source is recorded in the corresponding provenance registry.

"""
Repair latent-trial alignment and rerun EEG computational encoding.

Purpose
-------
The EEG trial-wise extraction can be valid while the latent merge is off by one
trial if the latent file uses zero-based trial indices and the metadata table
uses one-based trial indices. This script does not re-read EEG. It takes the
existing extracted EEG trial-feature table, removes any previously merged latent
columns, re-merges the latent table after automatically selecting the trial-index
offset that best matches cue type, NoGo status, and response, and then reruns
trial-level and subject-level encoding screens.

Example
-------
python code/final/fix_eeg_latent_alignment_and_encoding_v1.py \
  --features results/eeg_neural_dynamics_v1_1/eeg_trial_neural_features.csv \
  --latents outputs/ddm_latents/trial_latents_for_ddm_M6_seed2026.csv \
  --out results/eeg_neural_dynamics_v1_2_corrected
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


KNOWN_LATENT_COLUMNS = {
    "subject_index", "group_latent", "cue_idx", "cue_type_latent",
    "is_nogo", "action_press", "correct", "error", "fa",
    "p_press", "belief_nogo_current", "associability_current",
    "control_current", "actor_qdiff_current",
    "cnv_z", "p300_z", "ern_strength_z", "pe_strength_z",
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


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "subject_id_norm" not in df.columns:
        df["subject_id_norm"] = df["subject_id"].map(normalize_subject_id)
    if "trial_index_numeric" not in df.columns:
        df["trial_index_numeric"] = pd.to_numeric(df["trial_index"], errors="coerce")
    # Remove previously merged latent columns so the corrected merge is clean.
    drop_cols = [c for c in df.columns if c in KNOWN_LATENT_COLUMNS]
    return df.drop(columns=drop_cols, errors="ignore")


def load_latents(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "subject_id" in df.columns:
        df["subject_id_norm"] = df["subject_id"].map(normalize_subject_id)
    elif "subj" in df.columns:
        df["subject_id_norm"] = df["subj"].map(normalize_subject_id)
    else:
        raise ValueError("Latent file must contain `subject_id` or `subj`.")

    if "trial_index" in df.columns:
        df["latent_trial_index_numeric"] = pd.to_numeric(df["trial_index"], errors="coerce")
    elif "trial" in df.columns:
        df["latent_trial_index_numeric"] = pd.to_numeric(df["trial"], errors="coerce")
    else:
        raise ValueError("Latent file must contain `trial_index` or `trial`.")

    # Rename non-key behavioral descriptors to make their source explicit.
    rename = {}
    if "group" in df.columns:
        rename["group"] = "group_latent"
    if "cue_type" in df.columns:
        rename["cue_type"] = "cue_type_latent"
    df = df.rename(columns=rename)

    # Keep only useful latent columns and keys.
    keep = ["subject_id_norm", "latent_trial_index_numeric"]
    for c in df.columns:
        if c in KNOWN_LATENT_COLUMNS and c not in keep:
            keep.append(c)

    # Preserve any common latent names that may not be in the known list.
    for c in df.columns:
        if any(k in c.lower() for k in ["press", "qdiff", "control", "belief", "associability", "surprise", "prediction_error"]):
            if c not in keep and c not in {"subject_id", "subj", "trial", "trial_index"}:
                keep.append(c)

    return df[keep].copy()


def score_merge(features: pd.DataFrame, latents: pd.DataFrame, offset: int) -> Dict[str, float]:
    tmp = latents.copy()
    tmp["trial_index_numeric"] = tmp["latent_trial_index_numeric"] + offset
    merged = features[["subject_id_norm", "trial_index_numeric", "cue_type", "trial_class", "response_present"]].merge(
        tmp.drop(columns=["latent_trial_index_numeric"]),
        on=["subject_id_norm", "trial_index_numeric"],
        how="left",
    )
    matched_rows = int(merged["p_press"].notna().sum()) if "p_press" in merged.columns else int(merged.iloc[:, -1].notna().sum())

    scores = []
    details = {"offset": offset, "matched_rows": matched_rows}

    if "cue_type_latent" in merged.columns:
        m = merged["cue_type_latent"].notna()
        if m.any():
            cue_match = (merged.loc[m, "cue_type"].map(normalize_label) == merged.loc[m, "cue_type_latent"].map(normalize_label)).mean()
            details["cue_match"] = float(cue_match)
            scores.append(cue_match)

    if "is_nogo" in merged.columns:
        m = merged["is_nogo"].notna()
        if m.any():
            meta_nogo = (merged.loc[m, "trial_class"].map(normalize_label) == "nogo").astype(int)
            nogo_match = (meta_nogo.values == pd.to_numeric(merged.loc[m, "is_nogo"], errors="coerce").astype(int).values).mean()
            details["nogo_match"] = float(nogo_match)
            scores.append(nogo_match)

    if "action_press" in merged.columns:
        m = merged["action_press"].notna()
        if m.any():
            press_match = (
                pd.to_numeric(merged.loc[m, "response_present"], errors="coerce").astype(float).values
                == pd.to_numeric(merged.loc[m, "action_press"], errors="coerce").astype(float).values
            ).mean()
            details["press_match"] = float(press_match)
            scores.append(press_match)

    details["mean_behavioral_match"] = float(np.mean(scores)) if scores else np.nan
    return details


def choose_offset(features: pd.DataFrame, latents: pd.DataFrame, offsets: List[int]) -> Tuple[int, pd.DataFrame]:
    rows = [score_merge(features, latents, off) for off in offsets]
    scores = pd.DataFrame(rows)
    scores = scores.sort_values(["mean_behavioral_match", "matched_rows"], ascending=False).reset_index(drop=True)
    best_offset = int(scores.iloc[0]["offset"])
    return best_offset, scores


def corrected_merge(features: pd.DataFrame, latents: pd.DataFrame, offset: int) -> pd.DataFrame:
    tmp = latents.copy()
    tmp["trial_index_numeric"] = tmp["latent_trial_index_numeric"] + offset
    tmp = tmp.drop(columns=["latent_trial_index_numeric"])
    return features.merge(tmp, on=["subject_id_norm", "trial_index_numeric"], how="left")


def neural_feature_columns(df: pd.DataFrame) -> List[str]:
    feature_keywords = [
        "cue_cnv", "cue_n2", "cue_p3", "target_n2", "target_p300",
        "resp_rp", "resp_ern", "resp_pe"
    ]
    suffixes = ("_mean_uv", "_min_uv", "_max_uv", "_slope_uv_per_s")
    return [
        c for c in df.columns
        if any(k in c for k in feature_keywords) and c.endswith(suffixes)
    ]


def latent_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "p_press", "control_current", "actor_qdiff_current",
        "belief_nogo_current", "associability_current",
    ]
    out = []
    for c in candidates:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().sum() > 50 and x.std(skipna=True) > 1e-12:
                out.append(c)
    return out


def within_subject_demean(df: pd.DataFrame, col: str) -> pd.Series:
    x = pd.to_numeric(df[col], errors="coerce")
    return x - x.groupby(df["subject_id_norm"]).transform("mean")


def trial_level_encoding(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lats = latent_columns(df)
    neus = neural_feature_columns(df)

    for latent in lats:
        x = within_subject_demean(df, latent)
        for neural in neus:
            y = within_subject_demean(df, neural)
            mask = np.isfinite(x) & np.isfinite(y)
            n = int(mask.sum())
            if n < 50:
                continue
            r, p = stats.pearsonr(x[mask], y[mask])
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
        out = out.sort_values(["q_value_fdr_bh", "p_value"]).reset_index(drop=True)
    return out


def subject_level_encoding(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lats = latent_columns(df)
    neus = neural_feature_columns(df)

    for latent in lats:
        for neural in neus:
            subject_rs = []
            for _, g in df.groupby("subject_id_norm"):
                x = pd.to_numeric(g[latent], errors="coerce")
                y = pd.to_numeric(g[neural], errors="coerce")
                mask = np.isfinite(x) & np.isfinite(y)
                if int(mask.sum()) < 20:
                    continue
                if float(np.nanstd(x[mask])) <= 1e-12 or float(np.nanstd(y[mask])) <= 1e-12:
                    continue
                r, _ = stats.pearsonr(x[mask], y[mask])
                if np.isfinite(r):
                    subject_rs.append(float(r))

            if len(subject_rs) < 10:
                continue

            z = np.arctanh(np.clip(np.array(subject_rs), -0.999999, 0.999999))
            t, p = stats.ttest_1samp(z, 0.0)
            rows.append({
                "latent_variable": latent,
                "neural_feature": neural,
                "n_subjects": int(len(subject_rs)),
                "mean_r_fisher": float(np.tanh(np.mean(z))),
                "t_value": float(t),
                "p_value": float(p),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_fdr_bh"] = fdr_bh(out["p_value"].values)
        out = out.sort_values(["q_value_fdr_bh", "p_value"]).reset_index(drop=True)
    return out


def write_report(out_dir: Path, offset_scores: pd.DataFrame, best_offset: int, merged: pd.DataFrame, trial_enc: pd.DataFrame, subj_enc: pd.DataFrame) -> None:
    lines = []
    lines.append("# Corrected EEG-latent alignment and encoding report")
    lines.append("")
    lines.append("## Alignment")
    lines.append("")
    lines.append(f"- Selected latent trial-index offset: `{best_offset}`")
    lines.append(f"- Corrected merged rows: {len(merged)}")
    lines.append(f"- Rows with p_press: {int(merged['p_press'].notna().sum()) if 'p_press' in merged.columns else 'not available'}")
    lines.append("")
    lines.append("Offset candidates:")
    lines.append("")
    lines.append("| Offset | matched rows | cue match | NoGo match | press match | mean match |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for _, row in offset_scores.sort_values("offset").iterrows():
        lines.append(
            f"| {int(row['offset'])} | {int(row['matched_rows'])} | "
            f"{row.get('cue_match', np.nan):.4f} | {row.get('nogo_match', np.nan):.4f} | "
            f"{row.get('press_match', np.nan):.4f} | {row.get('mean_behavioral_match', np.nan):.4f} |"
        )

    lines.append("")
    lines.append("## Trial-level exploratory encoding")
    lines.append("")
    lines.append(f"- Tested pairs: {len(trial_enc)}")
    lines.append(f"- FDR q < 0.05 pairs: {int((trial_enc['q_value_fdr_bh'] < 0.05).sum()) if not trial_enc.empty else 0}")
    if not trial_enc.empty:
        lines.append("")
        lines.append("| Latent | Neural feature | r | p | q |")
        lines.append("|---|---|---:|---:|---:|")
        for _, row in trial_enc.head(15).iterrows():
            lines.append(
                f"| {row['latent_variable']} | {row['neural_feature']} | "
                f"{row['within_subject_r']:.4f} | {row['p_value']:.3g} | {row['q_value_fdr_bh']:.3g} |"
            )

    lines.append("")
    lines.append("## Subject-level exploratory encoding")
    lines.append("")
    lines.append(f"- Tested pairs: {len(subj_enc)}")
    lines.append(f"- FDR q < 0.05 pairs: {int((subj_enc['q_value_fdr_bh'] < 0.05).sum()) if not subj_enc.empty else 0}")
    if not subj_enc.empty:
        lines.append("")
        lines.append("| Latent | Neural feature | mean r | p | q |")
        lines.append("|---|---|---:|---:|---:|")
        for _, row in subj_enc.head(15).iterrows():
            lines.append(
                f"| {row['latent_variable']} | {row['neural_feature']} | "
                f"{row['mean_r_fisher']:.4f} | {row['p_value']:.3g} | {row['q_value_fdr_bh']:.3g} |"
            )

    lines.append("")
    lines.append("## Interpretation guardrail")
    lines.append("")
    lines.append("These are exploratory encoding screens based on predefined ERP-feature windows. They support hypothesis generation,")
    lines.append("but a manuscript-level neuroscience claim should use targeted contrasts, subject-level inference, and correction")
    lines.append("within prespecified feature families.")
    lines.append("")
    (out_dir / "corrected_eeg_latent_encoding_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=str)
    parser.add_argument("--latents", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--offsets", default="-2,-1,0,1,2", type=str)
    args = parser.parse_args()

    features_path = Path(args.features).expanduser().resolve()
    latents_path = Path(args.latents).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    features = load_features(features_path)
    latents = load_latents(latents_path)
    offsets = [int(x.strip()) for x in args.offsets.split(",") if x.strip()]

    best_offset, offset_scores = choose_offset(features, latents, offsets)
    merged = corrected_merge(features, latents, best_offset)

    trial_enc = trial_level_encoding(merged)
    subj_enc = subject_level_encoding(merged)

    merged.to_csv(out_dir / "eeg_trial_neural_features_latents_corrected.csv", index=False)
    offset_scores.to_csv(out_dir / "latent_alignment_offset_scores.csv", index=False)
    trial_enc.to_csv(out_dir / "eeg_computational_encoding_trial_level_corrected.csv", index=False)
    subj_enc.to_csv(out_dir / "eeg_computational_encoding_subject_level_corrected.csv", index=False)
    write_report(out_dir, offset_scores, best_offset, merged, trial_enc, subj_enc)

    print(f"Selected latent trial-index offset: {best_offset}")
    print(f"Wrote corrected outputs to: {out_dir}")
    if not subj_enc.empty:
        print(subj_enc.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
