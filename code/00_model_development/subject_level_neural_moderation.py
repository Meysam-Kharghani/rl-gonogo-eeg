#!/usr/bin/env python3
# FORMALIZED HISTORICAL SOURCE SNAPSHOT
# Evidence status: HISTORICAL_OR_SUPERSEDED_DEVELOPMENT_EVIDENCE
# This file preserves a historical project stage with public-facing terminology standardized for release.
# The SHA-256 of the original executed/development source is recorded in the corresponding provenance registry.

"""
Subject-level neural-signature moderation analysis for the EEG-computational study.

Purpose
-------
This script asks whether subject-level neural signatures of inhibitory-control
failure and successful inhibition are related to:
1. risk group,
2. behavioral error rates,
3. model-derived RL/DDM latent summaries.

This is intended as an individual-differences / moderation layer after:
- ERP feature extraction,
- time-frequency extraction,
- waveform analysis,
- topography analysis.

Main signatures
---------------
A. Error-monitoring signature: NoGo false alarm minus Go hit
   - ERN Fz, 50–150 ms
   - Pe Fz/Cz/Pz, 200–400 ms
   - response theta Fz/Cz, 0–400 ms
   - response beta Cz, 200–500 ms

B. Inhibition-success signature: NoGo correct rejection minus NoGo false alarm
   - target P300 / late positivity features
   - optional topography-derived summary if topography outputs are supplied

Outputs
-------
- subject_neural_signature_table.csv
- subject_neural_signature_group_tests.csv
- subject_neural_signature_behavior_correlations.csv
- subject_neural_signature_rl_correlations.csv
- EEG_SUBJECT_NEURAL_SIGNATURE_MODERATION_REPORT.md

Example
-------
python code/final/run_eeg_subject_neural_signature_moderation_v1.py \
  --erp_features results/eeg_neural_dynamics_v1_2_corrected/eeg_trial_neural_features_latents_corrected.csv \
  --tf_features results/eeg_time_frequency_analysis_v1/tf_trial_bandpower_features.csv \
  --topography_dir results/eeg_topography_analysis_v1 \
  --out results/eeg_subject_neural_signature_moderation_v1
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


ERP_ERROR_FEATURES = {
    "ern_fz_mean": "resp_ern_fz_50_150_mean_uv",
    "ern_fz_min": "resp_ern_fz_50_150_min_uv",
    "pe_fz_mean": "resp_pe_fz_200_400_mean_uv",
    "pe_fz_max": "resp_pe_fz_200_400_max_uv",
    "pe_cz_mean": "resp_pe_cz_200_400_mean_uv",
    "pe_cz_max": "resp_pe_cz_200_400_max_uv",
    "pe_pz_mean": "resp_pe_pz_200_400_mean_uv",
    "pe_pz_max": "resp_pe_pz_200_400_max_uv",
    "rp_cz_mean": "resp_rp_cz_m300_0_mean_uv",
}

ERP_INHIBITION_FEATURES = {
    "target_p300_cz_mean": "target_p300_cz_250_500_mean_uv",
    "target_p300_cz_max": "target_p300_cz_250_500_max_uv",
    "target_p300_pz_mean": "target_p300_pz_250_500_mean_uv",
    "target_p300_pz_max": "target_p300_pz_250_500_max_uv",
    "target_n2_fz_mean": "target_n2_fz_200_350_mean_uv",
    "target_n2_cz_mean": "target_n2_cz_200_350_mean_uv",
    "cue_cnv_late_cz_mean": "cue_cnv_late_cz_200_500_mean_uv",
    "cue_p3_cz_mean": "cue_p3_cz_250_450_mean_uv",
}

TF_ERROR_FEATURES = {
    "resp_theta_fz_0_400": "resp_theta_fz_0_400",
    "resp_theta_fz_0_200": "resp_theta_fz_0_200",
    "resp_theta_cz_0_400": "resp_theta_cz_0_400",
    "resp_beta_cz_200_500": "resp_beta_cz_200_500",
    "resp_beta_cz_m400_0": "resp_beta_cz_m400_0",
    "resp_mu_cz_m400_0": "resp_mu_cz_m400_0",
}

TF_INHIBITION_FEATURES = {
    "target_theta_fz_200_500": "target_theta_fz_200_500",
    "target_theta_cz_200_500": "target_theta_cz_200_500",
    "target_alpha_pz_250_500": "target_alpha_pz_250_500",
    "target_beta_cz_250_500": "target_beta_cz_250_500",
}

RL_COLUMNS = [
    "p_press",
    "control_current",
    "actor_qdiff_current",
    "belief_nogo_current",
    "associability_current",
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


def load_erp_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "subject_id_norm" not in df.columns:
        df["subject_id_norm"] = df["subject_id"].map(normalize_subject_id)
    if "trial_index_numeric" not in df.columns:
        df["trial_index_numeric"] = pd.to_numeric(df["trial_index"], errors="coerce")

    if "group" in df.columns:
        df["group"] = df["group"].map(norm_text)
    else:
        df["group"] = df["subject_id_norm"].map(lambda s: "high" if str(s).startswith("H") else "low")

    if "trial_class" in df.columns:
        df["trial_class_norm"] = df["trial_class"].map(norm_text)
    else:
        df["trial_class_norm"] = ""

    if "response_type" in df.columns:
        df["response_type_norm"] = df["response_type"].map(norm_text)
    else:
        df["response_type_norm"] = ""

    if "response_present" in df.columns:
        df["response_present_numeric"] = pd.to_numeric(df["response_present"], errors="coerce").fillna(0).astype(int)
    else:
        df["response_present_numeric"] = np.nan

    if "rt_response_from_target_ms" in df.columns:
        df["rt_ms"] = pd.to_numeric(df["rt_response_from_target_ms"], errors="coerce")
    else:
        df["rt_ms"] = np.nan

    df["outcome"] = df["response_type_norm"].map({
        "hit": "go_hit",
        "miss": "go_miss",
        "cr": "nogo_cr",
        "correct rejection": "nogo_cr",
        "correct_rejection": "nogo_cr",
        "fa": "nogo_fa",
        "false alarm": "nogo_fa",
        "false_alarm": "nogo_fa",
    })

    press = df["response_present_numeric"].fillna(0).astype(int)
    is_go = df["trial_class_norm"] == "go"
    is_nogo = df["trial_class_norm"] == "nogo"

    df.loc[df["outcome"].isna() & is_go & (press == 1), "outcome"] = "go_hit"
    df.loc[df["outcome"].isna() & is_go & (press == 0), "outcome"] = "go_miss"
    df.loc[df["outcome"].isna() & is_nogo & (press == 0), "outcome"] = "nogo_cr"
    df.loc[df["outcome"].isna() & is_nogo & (press == 1), "outcome"] = "nogo_fa"

    return df


def load_tf_features(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None

    df = pd.read_csv(path)
    if "subject_id_norm" not in df.columns:
        df["subject_id_norm"] = df["subject_id"].map(normalize_subject_id)
    if "trial_index_numeric" not in df.columns:
        df["trial_index_numeric"] = pd.to_numeric(df["trial_index"], errors="coerce")

    keep = ["subject_id_norm", "trial_index_numeric"]
    for col in list(TF_ERROR_FEATURES.values()) + list(TF_INHIBITION_FEATURES.values()):
        if col in df.columns and col not in keep:
            keep.append(col)

    return df[keep].copy()


def merge_features(erp: pd.DataFrame, tf: Optional[pd.DataFrame]) -> pd.DataFrame:
    if tf is None:
        return erp
    return erp.merge(tf, on=["subject_id_norm", "trial_index_numeric"], how="left", suffixes=("", "_tf"))


def safe_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan
    return float(vals.mean())


def contrast_for_subject(g: pd.DataFrame, feature_col: str, outcome_a: str, outcome_b: str, min_trials: int) -> Tuple[float, int, int]:
    if feature_col not in g.columns:
        return np.nan, 0, 0

    a = pd.to_numeric(g.loc[g["outcome"] == outcome_a, feature_col], errors="coerce").dropna()
    b = pd.to_numeric(g.loc[g["outcome"] == outcome_b, feature_col], errors="coerce").dropna()

    if len(a) < min_trials or len(b) < min_trials:
        return np.nan, int(len(a)), int(len(b))

    return float(a.mean() - b.mean()), int(len(a)), int(len(b))


def compute_subject_table(df: pd.DataFrame, min_trials: int, topography_dir: Optional[Path]) -> pd.DataFrame:
    rows = []

    for subject, g in df.groupby("subject_id_norm"):
        row = {
            "subject_id": subject,
            "group": g["group"].iloc[0] if "group" in g.columns else ("high" if str(subject).startswith("H") else "low"),
            "n_trials": int(len(g)),
            "n_go_hit": int((g["outcome"] == "go_hit").sum()),
            "n_go_miss": int((g["outcome"] == "go_miss").sum()),
            "n_nogo_cr": int((g["outcome"] == "nogo_cr").sum()),
            "n_nogo_fa": int((g["outcome"] == "nogo_fa").sum()),
        }

        n_go = row["n_go_hit"] + row["n_go_miss"]
        n_nogo = row["n_nogo_cr"] + row["n_nogo_fa"]

        row["press_rate"] = float(g["response_present_numeric"].mean()) if "response_present_numeric" in g.columns else np.nan
        row["go_miss_rate"] = row["n_go_miss"] / n_go if n_go > 0 else np.nan
        row["nogo_fa_rate"] = row["n_nogo_fa"] / n_nogo if n_nogo > 0 else np.nan
        row["mean_rt_ms"] = safe_mean(g.loc[g["response_present_numeric"] == 1, "rt_ms"]) if "rt_ms" in g.columns else np.nan
        row["mean_hit_rt_ms"] = safe_mean(g.loc[g["outcome"] == "go_hit", "rt_ms"]) if "rt_ms" in g.columns else np.nan
        row["mean_fa_rt_ms"] = safe_mean(g.loc[g["outcome"] == "nogo_fa", "rt_ms"]) if "rt_ms" in g.columns else np.nan

        # RL latent summaries.
        for col in RL_COLUMNS:
            if col in g.columns:
                row[f"mean_{col}"] = safe_mean(g[col])
                if "nogo" in g["trial_class_norm"].unique():
                    row[f"mean_{col}_nogo"] = safe_mean(g.loc[g["trial_class_norm"] == "nogo", col])
                if "go" in g["trial_class_norm"].unique():
                    row[f"mean_{col}_go"] = safe_mean(g.loc[g["trial_class_norm"] == "go", col])

        # Error-monitoring signatures: FA minus Go Hit.
        error_signature_cols = []
        for short, col in ERP_ERROR_FEATURES.items():
            val, na, nb = contrast_for_subject(g, col, "nogo_fa", "go_hit", min_trials)
            outcol = f"sig_error_fa_minus_hit_{short}"
            row[outcol] = val
            row[f"{outcol}_n_fa"] = na
            row[f"{outcol}_n_hit"] = nb
            if np.isfinite(val):
                error_signature_cols.append(outcol)

        for short, col in TF_ERROR_FEATURES.items():
            val, na, nb = contrast_for_subject(g, col, "nogo_fa", "go_hit", min_trials)
            outcol = f"sig_error_fa_minus_hit_{short}"
            row[outcol] = val
            row[f"{outcol}_n_fa"] = na
            row[f"{outcol}_n_hit"] = nb
            if np.isfinite(val):
                error_signature_cols.append(outcol)

        # Successful inhibition: CR minus FA.
        inhibition_signature_cols = []
        for short, col in ERP_INHIBITION_FEATURES.items():
            val, na, nb = contrast_for_subject(g, col, "nogo_cr", "nogo_fa", min_trials)
            outcol = f"sig_inhibition_cr_minus_fa_{short}"
            row[outcol] = val
            row[f"{outcol}_n_cr"] = na
            row[f"{outcol}_n_fa"] = nb
            if np.isfinite(val):
                inhibition_signature_cols.append(outcol)

        for short, col in TF_INHIBITION_FEATURES.items():
            val, na, nb = contrast_for_subject(g, col, "nogo_cr", "nogo_fa", min_trials)
            outcol = f"sig_inhibition_cr_minus_fa_{short}"
            row[outcol] = val
            row[f"{outcol}_n_cr"] = na
            row[f"{outcol}_n_fa"] = nb
            if np.isfinite(val):
                inhibition_signature_cols.append(outcol)

        # Composite signatures with direction harmonization:
        # For error monitoring, stronger signal = more negative ERN, more positive Pe,
        # more positive theta, and more negative beta if present.
        composite_terms = []
        direction_map = {}
        for key in ["ern_fz_mean", "ern_fz_min"]:
            c = f"sig_error_fa_minus_hit_{key}"
            if c in row and np.isfinite(row[c]):
                direction_map[c] = -1.0
        for key in ["pe_fz_mean", "pe_fz_max", "pe_cz_mean", "pe_cz_max", "pe_pz_mean", "pe_pz_max"]:
            c = f"sig_error_fa_minus_hit_{key}"
            if c in row and np.isfinite(row[c]):
                direction_map[c] = +1.0
        for key in ["resp_theta_fz_0_400", "resp_theta_fz_0_200", "resp_theta_cz_0_400"]:
            c = f"sig_error_fa_minus_hit_{key}"
            if c in row and np.isfinite(row[c]):
                direction_map[c] = +1.0
        for key in ["resp_beta_cz_200_500"]:
            c = f"sig_error_fa_minus_hit_{key}"
            if c in row and np.isfinite(row[c]):
                direction_map[c] = -1.0

        # Store raw directed terms for later z-scoring across subjects.
        for c, sign in direction_map.items():
            row[f"directed_{c}"] = sign * row[c]

        rows.append(row)

    table = pd.DataFrame(rows)

    # Build composite z-scores across subjects using directed columns.
    directed_error_cols = [c for c in table.columns if c.startswith("directed_sig_error_fa_minus_hit_")]
    if directed_error_cols:
        zparts = []
        for c in directed_error_cols:
            x = pd.to_numeric(table[c], errors="coerce")
            sd = x.std(skipna=True)
            if pd.notna(sd) and sd > 0:
                zparts.append((x - x.mean(skipna=True)) / sd)
        if zparts:
            table["composite_error_monitoring_signature_z"] = pd.concat(zparts, axis=1).mean(axis=1, skipna=True)

    # Inhibition composite: positive late/P300-like CR-FA and positive theta/alpha terms
    inhibition_cols = [
        c for c in table.columns
        if c.startswith("sig_inhibition_cr_minus_fa_")
        and not c.endswith(("_n_cr", "_n_fa"))
    ]
    if inhibition_cols:
        zparts = []
        for c in inhibition_cols:
            x = pd.to_numeric(table[c], errors="coerce")
            sd = x.std(skipna=True)
            if pd.notna(sd) and sd > 0:
                zparts.append((x - x.mean(skipna=True)) / sd)
        if zparts:
            table["composite_inhibition_success_signature_z"] = pd.concat(zparts, axis=1).mean(axis=1, skipna=True)

    # Optional topography summary integration.
    if topography_dir is not None and topography_dir.exists():
        sig_path = topography_dir / "eeg_topography_significant_channels.csv"
        if sig_path.exists():
            topo = pd.read_csv(sig_path)
            table["topography_features_available"] = 1
            table["topography_n_features_q_lt_05"] = int((pd.to_numeric(topo.get("n_channels_q_lt_05", pd.Series(dtype=float)), errors="coerce") > 0).sum())
        else:
            table["topography_features_available"] = 0

    return table


def group_tests(subject_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidate_cols = [
        c for c in subject_table.columns
        if (
            c.startswith("sig_error_")
            or c.startswith("sig_inhibition_")
            or c.startswith("composite_")
            or c in ["nogo_fa_rate", "go_miss_rate", "press_rate", "mean_rt_ms"]
        )
        and not c.endswith(("_n_fa", "_n_hit", "_n_cr"))
    ]

    for col in candidate_cols:
        high = pd.to_numeric(subject_table.loc[subject_table["group"] == "high", col], errors="coerce").dropna()
        low = pd.to_numeric(subject_table.loc[subject_table["group"] == "low", col], errors="coerce").dropna()
        if len(high) < 5 or len(low) < 5:
            continue

        t, p = stats.ttest_ind(high.values, low.values, equal_var=False)
        rows.append({
            "variable": col,
            "test": "welch_high_minus_low",
            "n_high": int(len(high)),
            "n_low": int(len(low)),
            "mean_high": float(high.mean()),
            "mean_low": float(low.mean()),
            "mean_difference_high_minus_low": float(high.mean() - low.mean()),
            "t_value": float(t),
            "p_value": float(p),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_fdr_bh"] = fdr_bh(out["p_value"].values)
        out = out.sort_values(["q_value_fdr_bh", "p_value"]).reset_index(drop=True)
    return out


def correlation_tests(subject_table: pd.DataFrame, target_cols: List[str], predictor_cols: List[str], label: str) -> pd.DataFrame:
    rows = []

    for target in target_cols:
        if target not in subject_table.columns:
            continue
        y = pd.to_numeric(subject_table[target], errors="coerce")

        for pred in predictor_cols:
            if pred not in subject_table.columns or pred == target:
                continue
            x = pd.to_numeric(subject_table[pred], errors="coerce")
            mask = np.isfinite(x) & np.isfinite(y)
            n = int(mask.sum())
            if n < 10:
                continue
            if x[mask].std() <= 1e-12 or y[mask].std() <= 1e-12:
                continue

            pear_r, pear_p = stats.pearsonr(x[mask], y[mask])
            spear_r, spear_p = stats.spearmanr(x[mask], y[mask])
            rows.append({
                "analysis_family": label,
                "target": target,
                "predictor": pred,
                "n_subjects": n,
                "pearson_r": float(pear_r),
                "pearson_p": float(pear_p),
                "spearman_r": float(spear_r),
                "spearman_p": float(spear_p),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["pearson_q_fdr_bh"] = fdr_bh(out["pearson_p"].values)
        out["spearman_q_fdr_bh"] = fdr_bh(out["spearman_p"].values)
        out = out.sort_values(["pearson_q_fdr_bh", "pearson_p"]).reset_index(drop=True)
    return out


def write_report(out_dir: Path, subject_table: pd.DataFrame, group_df: pd.DataFrame, behavior_corr: pd.DataFrame, rl_corr: pd.DataFrame, min_trials: int) -> None:
    lines = []
    lines.append("# EEG subject-level neural-signature moderation report")
    lines.append("")
    lines.append("This report tests whether neural signatures of inhibitory-control failure and successful inhibition relate to risk group, behavior, and RL/DDM latent summaries.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Subjects: {len(subject_table)}")
    lines.append(f"- Minimum trials per condition for subject contrasts: {min_trials}")
    if "group" in subject_table.columns:
        lines.append(f"- High-risk subjects: {int((subject_table['group'] == 'high').sum())}")
        lines.append(f"- Low-risk subjects: {int((subject_table['group'] == 'low').sum())}")
    if "composite_error_monitoring_signature_z" in subject_table.columns:
        n = int(pd.to_numeric(subject_table["composite_error_monitoring_signature_z"], errors="coerce").notna().sum())
        lines.append(f"- Subjects with composite error-monitoring signature: {n}")
    if "composite_inhibition_success_signature_z" in subject_table.columns:
        n = int(pd.to_numeric(subject_table["composite_inhibition_success_signature_z"], errors="coerce").notna().sum())
        lines.append(f"- Subjects with composite inhibition-success signature: {n}")

    lines.append("")
    lines.append("## Group differences")
    lines.append("")
    lines.append(f"- Tested group rows: {len(group_df)}")
    lines.append(f"- FDR q < 0.05 rows: {int((group_df['q_value_fdr_bh'] < 0.05).sum()) if not group_df.empty else 0}")
    if not group_df.empty:
        lines.append("")
        lines.append("| Variable | n high | n low | high mean | low mean | high-low | p | q |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in group_df.head(30).iterrows():
            lines.append(
                f"| {row['variable']} | {int(row['n_high'])} | {int(row['n_low'])} | "
                f"{row['mean_high']:.4f} | {row['mean_low']:.4f} | {row['mean_difference_high_minus_low']:.4f} | "
                f"{row['p_value']:.3g} | {row['q_value_fdr_bh']:.3g} |"
            )

    lines.append("")
    lines.append("## Behavior correlations")
    lines.append("")
    lines.append(f"- Tested behavior-correlation rows: {len(behavior_corr)}")
    lines.append(f"- Pearson FDR q < 0.05 rows: {int((behavior_corr['pearson_q_fdr_bh'] < 0.05).sum()) if not behavior_corr.empty else 0}")
    if not behavior_corr.empty:
        lines.append("")
        lines.append("| Neural target | Behavioral predictor | n | Pearson r | p | q | Spearman rho |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for _, row in behavior_corr.head(30).iterrows():
            lines.append(
                f"| {row['target']} | {row['predictor']} | {int(row['n_subjects'])} | "
                f"{row['pearson_r']:.4f} | {row['pearson_p']:.3g} | {row['pearson_q_fdr_bh']:.3g} | "
                f"{row['spearman_r']:.4f} |"
            )

    lines.append("")
    lines.append("## RL/DDM latent correlations")
    lines.append("")
    lines.append(f"- Tested RL-correlation rows: {len(rl_corr)}")
    lines.append(f"- Pearson FDR q < 0.05 rows: {int((rl_corr['pearson_q_fdr_bh'] < 0.05).sum()) if not rl_corr.empty else 0}")
    if not rl_corr.empty:
        lines.append("")
        lines.append("| Neural target | RL predictor | n | Pearson r | p | q | Spearman rho |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for _, row in rl_corr.head(30).iterrows():
            lines.append(
                f"| {row['target']} | {row['predictor']} | {int(row['n_subjects'])} | "
                f"{row['pearson_r']:.4f} | {row['pearson_p']:.3g} | {row['pearson_q_fdr_bh']:.3g} | "
                f"{row['spearman_r']:.4f} |"
            )

    lines.append("")
    lines.append("## Interpretation guardrail")
    lines.append("")
    lines.append("- This is a subject-level individual-differences analysis with N around 40, so effects should be interpreted cautiously.")
    lines.append("- Strong trial-level neural signatures may or may not explain between-subject risk-group differences.")
    lines.append("- If group/correlation effects do not survive FDR, describe the neural findings as robust error-monitoring signatures rather than risk-group mechanisms.")
    lines.append("- If composite neural signatures relate to FA or miss rates, they can support a cautious individual-differences claim.")
    lines.append("")
    (out_dir / "EEG_SUBJECT_NEURAL_SIGNATURE_MODERATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--erp_features", type=str, required=True)
    parser.add_argument("--tf_features", type=str, default="")
    parser.add_argument("--topography_dir", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--min_trials", type=int, default=10)
    args = parser.parse_args()

    erp_path = Path(args.erp_features).expanduser().resolve()
    tf_path = Path(args.tf_features).expanduser().resolve() if args.tf_features else None
    topo_dir = Path(args.topography_dir).expanduser().resolve() if args.topography_dir else None
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    erp = load_erp_features(erp_path)
    tf = load_tf_features(tf_path)
    df = merge_features(erp, tf)

    subject_table = compute_subject_table(df, min_trials=int(args.min_trials), topography_dir=topo_dir)

    neural_targets = [
        c for c in subject_table.columns
        if (
            c.startswith("sig_error_")
            or c.startswith("sig_inhibition_")
            or c.startswith("composite_")
        )
        and not c.endswith(("_n_fa", "_n_hit", "_n_cr"))
        and not c.startswith("directed_")
    ]

    behavior_predictors = [
        "nogo_fa_rate", "go_miss_rate", "press_rate", "mean_rt_ms", "mean_hit_rt_ms", "mean_fa_rt_ms"
    ]

    rl_predictors = []
    for col in subject_table.columns:
        if col.startswith("mean_") and any(lat in col for lat in RL_COLUMNS):
            rl_predictors.append(col)

    group_df = group_tests(subject_table)
    behavior_corr = correlation_tests(subject_table, neural_targets, behavior_predictors, label="behavior")
    rl_corr = correlation_tests(subject_table, neural_targets, rl_predictors, label="rl_latents")

    subject_table.to_csv(out_dir / "subject_neural_signature_table.csv", index=False)
    group_df.to_csv(out_dir / "subject_neural_signature_group_tests.csv", index=False)
    behavior_corr.to_csv(out_dir / "subject_neural_signature_behavior_correlations.csv", index=False)
    rl_corr.to_csv(out_dir / "subject_neural_signature_rl_correlations.csv", index=False)

    write_report(out_dir, subject_table, group_df, behavior_corr, rl_corr, int(args.min_trials))

    print(f"Wrote outputs to: {out_dir}")
    if not group_df.empty:
        print("Top group tests:")
        print(group_df.head(12).to_string(index=False))
    if not behavior_corr.empty:
        print("Top behavior correlations:")
        print(behavior_corr.head(12).to_string(index=False))
    if not rl_corr.empty:
        print("Top RL correlations:")
        print(rl_corr.head(12).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
