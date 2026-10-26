#!/usr/bin/env python3
# FORMALIZED HISTORICAL SOURCE SNAPSHOT
# Evidence status: HISTORICAL_OR_SUPERSEDED_DEVELOPMENT_EVIDENCE
# This file preserves a historical project stage with public-facing terminology standardized for release.
# The SHA-256 of the original executed/development source is recorded in the corresponding provenance registry.

"""
Robust validation for entropy candidates in the EEG-computational study.

This script validates the strongest entropy-related candidates from
run_eeg_entropy_feature_analysis_v1_1.py.

It intentionally does not search broadly. It tests a small preregistered-style
candidate set suggested by the historical implementation entropy screen:

A. Outcome contrasts
   - FA minus Go Hit response TF fractions/spectral entropy
   - CR minus FA target/cue TF fractions/spectral entropy
   across min-trial thresholds with sign-flip permutation and bootstrap CIs.

B. Trial-level targeted encoding
   - action_surprise / policy_entropy / policy_confidence
   against response spectral entropy, theta fraction, beta fraction, ERN, and Pe.
   Validation uses subject-wise correlations plus within-subject label
   permutation of the predictor.

C. Subject-level candidate correlations
   - CR-minus-FA cue_global_spectral_entropy vs policy-uncertainty summaries.
   Validation uses bootstrap and permutation over subjects.

Interpretation guardrails
-------------------------
- action_surprise uses the observed response; it is not predecision.
- policy_entropy/confidence are model-derived uncertainty/certainty summaries.
- spectral entropy/fractions are derived from existing TF bandpower features.
- This is supportive/exploratory unless robust across permutations,
  bootstraps, and min-trial thresholds.

Inputs
------
- entropy_trial_features.csv from results/eeg_entropy_feature_analysis_v1_1
- entropy_subject_summary.csv from results/eeg_entropy_feature_analysis_v1_1

Outputs
-------
- robust_entropy_outcome_contrast_summary.csv
- robust_entropy_encoding_observed_subject_rs.csv
- robust_entropy_encoding_null.csv
- robust_entropy_encoding_summary.csv
- robust_entropy_subject_correlation_summary.csv
- ROBUST_ENTROPY_CANDIDATE_REPORT.md

Example
-------
python code/final/run_eeg_entropy_robust_candidate_validation_v1.py \
  --entropy_trial_features results/eeg_entropy_feature_analysis_v1_1/entropy_trial_features.csv \
  --entropy_subject_summary results/eeg_entropy_feature_analysis_v1_1/entropy_subject_summary.csv \
  --out results/eeg_entropy_robust_candidate_validation_v1 \
  --n_perm 2000 \
  --n_boot 3000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


OUTCOME_CANDIDATES = [
    {
        "contrast": "nogo_fa_minus_go_hit",
        "a": "nogo_fa",
        "b": "go_hit",
        "features": [
            "resp_global_theta_fraction",
            "resp_cz_allwindows_theta_fraction",
            "resp_global_beta_fraction",
            "resp_cz_allwindows_beta_fraction",
            "resp_global_mu_fraction",
            "resp_global_spectral_entropy",
            "resp_cz_allwindows_spectral_entropy",
        ],
    },
    {
        "contrast": "nogo_cr_minus_nogo_fa",
        "a": "nogo_cr",
        "b": "nogo_fa",
        "features": [
            "target_global_theta_fraction",
            "target_global_alpha_fraction",
            "target_global_beta_fraction",
            "target_global_spectral_entropy",
            "target_cz_allwindows_spectral_entropy",
            "cue_global_spectral_entropy",
            "cue_global_theta_fraction",
            "cue_global_beta_fraction",
        ],
    },
    {
        "contrast": "go_miss_minus_go_hit",
        "a": "go_miss",
        "b": "go_hit",
        "features": [
            "target_global_theta_fraction",
            "target_global_beta_fraction",
            "target_global_spectral_entropy",
            "cue_global_spectral_entropy",
            "cue_cz_200_500_spectral_entropy",
        ],
    },
]

ENCODING_CANDIDATES = [
    ("action_surprise", "resp_global_spectral_entropy"),
    ("action_surprise", "resp_cz_allwindows_spectral_entropy"),
    ("action_surprise", "resp_global_theta_fraction"),
    ("action_surprise", "resp_cz_allwindows_theta_fraction"),
    ("action_surprise", "resp_global_beta_fraction"),
    ("action_surprise", "resp_pe_cz_200_400_mean_uv"),
    ("action_surprise", "resp_ern_fz_50_150_mean_uv"),
    ("policy_entropy", "resp_global_spectral_entropy"),
    ("policy_entropy", "resp_cz_allwindows_spectral_entropy"),
    ("policy_entropy", "resp_global_theta_fraction"),
    ("policy_entropy", "resp_global_beta_fraction"),
    ("policy_entropy", "resp_ern_fz_50_150_mean_uv"),
    ("policy_entropy", "resp_pe_cz_200_400_mean_uv"),
    ("policy_confidence", "resp_global_spectral_entropy"),
    ("policy_confidence", "resp_cz_allwindows_spectral_entropy"),
    ("policy_confidence", "resp_global_theta_fraction"),
    ("policy_confidence", "resp_global_beta_fraction"),
    ("policy_confidence", "resp_ern_fz_50_150_mean_uv"),
    ("policy_confidence", "resp_pe_cz_200_400_mean_uv"),
]

SUBJECT_CORRELATION_CANDIDATES = [
    ("contrast_cr_minus_fa_cue_global_spectral_entropy", "mean_policy_entropy_nogo"),
    ("contrast_cr_minus_fa_cue_global_spectral_entropy", "mean_policy_confidence_nogo"),
    ("contrast_cr_minus_fa_cue_global_spectral_entropy", "press_rate"),
    ("contrast_cr_minus_fa_cue_global_spectral_entropy", "go_miss_rate"),
    ("contrast_fa_minus_hit_resp_global_theta_fraction", "mean_policy_entropy_nogo"),
    ("contrast_fa_minus_hit_resp_global_beta_fraction", "mean_policy_entropy_nogo"),
]


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
    tmp = np.empty(n)
    tmp[order] = q
    out[ok] = tmp
    return out


def fisher_z(r: float) -> float:
    if not np.isfinite(r):
        return np.nan
    return float(np.arctanh(np.clip(r, -0.999999, 0.999999)))


def inv_fisher_z(z: float) -> float:
    if not np.isfinite(z):
        return np.nan
    return float(np.tanh(z))


def safe_corr(x: pd.Series, y: pd.Series, min_n: int = 20) -> Tuple[float, int]:
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    mask = np.isfinite(xx) & np.isfinite(yy)
    n = int(mask.sum())
    if n < min_n:
        return np.nan, n
    if xx[mask].std() <= 1e-12 or yy[mask].std() <= 1e-12:
        return np.nan, n
    r, _ = stats.pearsonr(xx[mask], yy[mask])
    return float(r), n


def bootstrap_mean_ci(vals: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, float, float, int]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan, 0
    if n_boot <= 0:
        return float(np.mean(vals)), np.nan, np.nan, 0
    boots = []
    n = len(vals)
    for _ in range(n_boot):
        sample = vals[rng.integers(0, n, size=n)]
        boots.append(float(np.mean(sample)))
    boots = np.asarray(boots, dtype=float)
    return float(np.mean(vals)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), int(len(boots))


def signflip_p(vals: np.ndarray, observed_mean: float, n_perm: int, rng: np.random.Generator) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0 or n_perm <= 0:
        return np.nan
    null = []
    n = len(vals)
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        null.append(float(np.mean(vals * signs)))
    null = np.asarray(null, dtype=float)
    return float((np.sum(np.abs(null) >= abs(observed_mean)) + 1) / (len(null) + 1))


def contrast_values_by_subject(df: pd.DataFrame, feature: str, a: str, b: str, min_trials: int) -> pd.DataFrame:
    rows = []
    if feature not in df.columns:
        return pd.DataFrame(rows)

    for subject, g in df.groupby("subject_id_norm"):
        va = pd.to_numeric(g.loc[g["outcome"] == a, feature], errors="coerce").dropna()
        vb = pd.to_numeric(g.loc[g["outcome"] == b, feature], errors="coerce").dropna()
        if len(va) < min_trials or len(vb) < min_trials:
            continue
        rows.append({
            "subject_id": subject,
            "group": g["group"].iloc[0] if "group" in g.columns else "",
            "feature": feature,
            "outcome_a": a,
            "outcome_b": b,
            "n_a": int(len(va)),
            "n_b": int(len(vb)),
            "difference": float(va.mean() - vb.mean()),
        })
    return pd.DataFrame(rows)


def run_outcome_contrasts(df: pd.DataFrame, thresholds: List[int], n_perm: int, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []

    for spec in OUTCOME_CANDIDATES:
        contrast = spec["contrast"]
        a = spec["a"]
        b = spec["b"]
        for feature in spec["features"]:
            if feature not in df.columns:
                continue
            for min_trials in thresholds:
                d = contrast_values_by_subject(df, feature, a, b, min_trials)
                vals = pd.to_numeric(d["difference"], errors="coerce").dropna().values if not d.empty else np.array([])
                if len(vals) < 8:
                    continue
                t, p = stats.ttest_1samp(vals, 0.0)
                mean, lo, hi, nb = bootstrap_mean_ci(vals, n_boot, rng)
                perm_p = signflip_p(vals, mean, n_perm, rng)
                rows.append({
                    "analysis": "outcome_contrast",
                    "contrast": contrast,
                    "feature": feature,
                    "min_trials": int(min_trials),
                    "n_subjects": int(len(vals)),
                    "mean_difference": mean,
                    "boot_ci_low": lo,
                    "boot_ci_high": hi,
                    "t_value": float(t),
                    "p_value": float(p),
                    "signflip_p": perm_p,
                    "positive_subjects": int(np.sum(vals > 0)),
                    "negative_subjects": int(np.sum(vals < 0)),
                    "mean_n_a": float(pd.to_numeric(d["n_a"], errors="coerce").mean()) if not d.empty else np.nan,
                    "mean_n_b": float(pd.to_numeric(d["n_b"], errors="coerce").mean()) if not d.empty else np.nan,
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_fdr_bh"] = fdr_bh(out["p_value"].values)
        out["signflip_q_fdr_bh"] = fdr_bh(out["signflip_p"].values)
        out = out.sort_values(["signflip_q_fdr_bh", "q_value_fdr_bh", "p_value"]).reset_index(drop=True)
    return out


def subject_correlations_for_pair(df: pd.DataFrame, predictor: str, target: str, min_n: int) -> pd.DataFrame:
    rows = []
    if predictor not in df.columns or target not in df.columns:
        return pd.DataFrame(rows)

    for subject, g in df.groupby("subject_id_norm"):
        r, n = safe_corr(g[predictor], g[target], min_n=min_n)
        if np.isfinite(r):
            rows.append({
                "subject_id": subject,
                "group": g["group"].iloc[0] if "group" in g.columns else "",
                "predictor": predictor,
                "target": target,
                "n_trials": int(n),
                "r": r,
                "fisher_z": fisher_z(r),
            })
    return pd.DataFrame(rows)


def permutation_encoding_mean_z(df: pd.DataFrame, predictor: str, target: str, min_n: int, n_perm: int, rng: np.random.Generator) -> Tuple[float, np.ndarray]:
    obs = subject_correlations_for_pair(df, predictor, target, min_n)
    if obs.empty:
        return np.nan, np.array([])
    obs_z = pd.to_numeric(obs["fisher_z"], errors="coerce").dropna().values
    if len(obs_z) < 8:
        return np.nan, np.array([])
    observed_mean_z = float(np.mean(obs_z))

    if n_perm <= 0:
        return observed_mean_z, np.array([])

    null = []
    # Shuffle the predictor within each subject to preserve each subject's
    # marginal predictor distribution and each EEG time series.
    grouped_indices = {s: g.index.values for s, g in df.groupby("subject_id_norm")}
    base_pred = pd.to_numeric(df[predictor], errors="coerce").values.copy()

    for _ in range(n_perm):
        pred_perm = base_pred.copy()
        for _, idx in grouped_indices.items():
            vals = pred_perm[idx].copy()
            rng.shuffle(vals)
            pred_perm[idx] = vals

        zs = []
        for subject, g in df.groupby("subject_id_norm"):
            x = pd.Series(pred_perm[g.index], index=g.index)
            y = g[target]
            r, n = safe_corr(x, y, min_n=min_n)
            if np.isfinite(r):
                zs.append(fisher_z(r))
        if len(zs) >= 8:
            null.append(float(np.mean(zs)))

    return observed_mean_z, np.asarray(null, dtype=float)


def run_encoding(df: pd.DataFrame, n_perm: int, n_boot: int, rng: np.random.Generator, min_n: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs_rows = []
    null_rows = []
    summary_rows = []

    for predictor, target in ENCODING_CANDIDATES:
        if predictor not in df.columns or target not in df.columns:
            continue

        obs = subject_correlations_for_pair(df, predictor, target, min_n=min_n)
        if obs.empty:
            continue
        obs_rows.append(obs)

        z = pd.to_numeric(obs["fisher_z"], errors="coerce").dropna().values
        if len(z) < 8:
            continue

        t, p = stats.ttest_1samp(z, 0.0)
        mean_z, lo_z, hi_z, nb = bootstrap_mean_ci(z, n_boot, rng)
        observed_mean_z, null = permutation_encoding_mean_z(df, predictor, target, min_n, n_perm, rng)

        if len(null):
            perm_p = float((np.sum(np.abs(null) >= abs(observed_mean_z)) + 1) / (len(null) + 1))
            for i, val in enumerate(null, start=1):
                null_rows.append({
                    "predictor": predictor,
                    "target": target,
                    "perm_idx": i,
                    "null_mean_fisher_z": float(val),
                    "null_mean_r": inv_fisher_z(float(val)),
                })
        else:
            perm_p = np.nan

        summary_rows.append({
            "predictor": predictor,
            "target": target,
            "n_subjects": int(len(z)),
            "mean_r": inv_fisher_z(mean_z),
            "boot_ci_low_r": inv_fisher_z(lo_z),
            "boot_ci_high_r": inv_fisher_z(hi_z),
            "mean_fisher_z": mean_z,
            "boot_ci_low_z": lo_z,
            "boot_ci_high_z": hi_z,
            "t_value": float(t),
            "p_value": float(p),
            "within_subject_permutation_p": perm_p,
            "n_permutations": int(len(null)),
        })

    observed = pd.concat(obs_rows, ignore_index=True) if obs_rows else pd.DataFrame()
    null_df = pd.DataFrame(null_rows)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["q_value_fdr_bh"] = fdr_bh(summary["p_value"].values)
        summary["permutation_q_fdr_bh"] = fdr_bh(summary["within_subject_permutation_p"].values)
        summary = summary.sort_values(["permutation_q_fdr_bh", "q_value_fdr_bh", "p_value"]).reset_index(drop=True)
    return observed, null_df, summary


def run_subject_correlations(subj: pd.DataFrame, n_perm: int, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for target, predictor in SUBJECT_CORRELATION_CANDIDATES:
        if target not in subj.columns or predictor not in subj.columns:
            continue
        x = pd.to_numeric(subj[predictor], errors="coerce")
        y = pd.to_numeric(subj[target], errors="coerce")
        mask = np.isfinite(x) & np.isfinite(y)
        n = int(mask.sum())
        if n < 10 or x[mask].std() <= 1e-12 or y[mask].std() <= 1e-12:
            continue

        r, p = stats.pearsonr(x[mask], y[mask])
        rho, sp = stats.spearmanr(x[mask], y[mask])

        boot = []
        idx_valid = np.where(mask.values)[0]
        for _ in range(n_boot):
            idx = rng.choice(idx_valid, size=len(idx_valid), replace=True)
            xb = x.iloc[idx]
            yb = y.iloc[idx]
            if xb.std() > 1e-12 and yb.std() > 1e-12:
                boot.append(stats.pearsonr(xb, yb)[0])
        boot = np.asarray(boot, dtype=float)
        boot = boot[np.isfinite(boot)]

        null = []
        xv = x.iloc[idx_valid].values.copy()
        yv = y.iloc[idx_valid].values.copy()
        for _ in range(n_perm):
            xp = xv.copy()
            rng.shuffle(xp)
            if np.std(xp) > 1e-12 and np.std(yv) > 1e-12:
                null.append(stats.pearsonr(xp, yv)[0])
        null = np.asarray(null, dtype=float)
        null = null[np.isfinite(null)]
        perm_p = float((np.sum(np.abs(null) >= abs(r)) + 1) / (len(null) + 1)) if len(null) else np.nan

        rows.append({
            "target": target,
            "predictor": predictor,
            "n_subjects": n,
            "pearson_r": float(r),
            "pearson_p": float(p),
            "pearson_boot_ci_low": float(np.quantile(boot, 0.025)) if len(boot) else np.nan,
            "pearson_boot_ci_high": float(np.quantile(boot, 0.975)) if len(boot) else np.nan,
            "pearson_permutation_p": perm_p,
            "spearman_rho": float(rho),
            "spearman_p": float(sp),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["pearson_q_fdr_bh"] = fdr_bh(out["pearson_p"].values)
        out["pearson_permutation_q_fdr_bh"] = fdr_bh(out["pearson_permutation_p"].values)
        out = out.sort_values(["pearson_permutation_q_fdr_bh", "pearson_q_fdr_bh", "pearson_p"]).reset_index(drop=True)
    return out


def fmt(x, digits=4):
    try:
        x = float(x)
        if not np.isfinite(x):
            return "NA"
        return f"{x:.{digits}f}"
    except Exception:
        return "NA"


def write_report(out_dir: Path, outcome_df: pd.DataFrame, encoding_df: pd.DataFrame, subj_corr_df: pd.DataFrame, n_perm: int, n_boot: int) -> None:
    lines = []
    lines.append("# Robust entropy-candidate validation report")
    lines.append("")
    lines.append("This report validates targeted entropy candidates from the historical implementation entropy screen.")
    lines.append("")
    lines.append("## Settings")
    lines.append("")
    lines.append(f"- Permutations: {n_perm}")
    lines.append(f"- Bootstraps: {n_boot}")

    lines.append("")
    lines.append("## A. Outcome-contrast robustness")
    lines.append("")
    lines.append(f"- Tested rows: {len(outcome_df)}")
    lines.append(f"- Sign-flip permutation FDR q < 0.05 rows: {int((outcome_df['signflip_q_fdr_bh'] < 0.05).sum()) if not outcome_df.empty else 0}")
    if not outcome_df.empty:
        lines.append("")
        lines.append("| Contrast | Feature | min trials | n | Mean diff | boot CI | sign-flip p | sign-flip q |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for _, row in outcome_df.head(45).iterrows():
            lines.append(
                f"| {row['contrast']} | {row['feature']} | {int(row['min_trials'])} | {int(row['n_subjects'])} | "
                f"{row['mean_difference']:.4f} | [{row['boot_ci_low']:.4f}, {row['boot_ci_high']:.4f}] | "
                f"{row['signflip_p']:.4g} | {row['signflip_q_fdr_bh']:.4g} |"
            )

    lines.append("")
    lines.append("## B. Trial-level encoding robustness")
    lines.append("")
    lines.append(f"- Tested candidate pairs: {len(encoding_df)}")
    lines.append(f"- Within-subject permutation FDR q < 0.05 rows: {int((encoding_df['permutation_q_fdr_bh'] < 0.05).sum()) if not encoding_df.empty else 0}")
    if not encoding_df.empty:
        lines.append("")
        lines.append("| Predictor | Target | n subjects | mean r | boot CI | perm p | perm q |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for _, row in encoding_df.head(45).iterrows():
            lines.append(
                f"| {row['predictor']} | {row['target']} | {int(row['n_subjects'])} | "
                f"{row['mean_r']:.4f} | [{row['boot_ci_low_r']:.4f}, {row['boot_ci_high_r']:.4f}] | "
                f"{row['within_subject_permutation_p']:.4g} | {row['permutation_q_fdr_bh']:.4g} |"
            )

    lines.append("")
    lines.append("## C. Subject-level candidate correlations")
    lines.append("")
    lines.append(f"- Tested rows: {len(subj_corr_df)}")
    lines.append(f"- Permutation FDR q < 0.05 rows: {int((subj_corr_df['pearson_permutation_q_fdr_bh'] < 0.05).sum()) if not subj_corr_df.empty else 0}")
    if not subj_corr_df.empty:
        lines.append("")
        lines.append("| Target | Predictor | n | r | boot CI | perm p | perm q |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for _, row in subj_corr_df.head(20).iterrows():
            lines.append(
                f"| {row['target']} | {row['predictor']} | {int(row['n_subjects'])} | "
                f"{row['pearson_r']:.4f} | [{row['pearson_boot_ci_low']:.4f}, {row['pearson_boot_ci_high']:.4f}] | "
                f"{row['pearson_permutation_p']:.4g} | {row['pearson_permutation_q_fdr_bh']:.4g} |"
            )

    lines.append("")
    lines.append("## Interpretation guardrail")
    lines.append("")
    lines.append("- Outcome-contrast rows that survive every min-trial threshold are the safest.")
    lines.append("- action_surprise is post-outcome/model-surprise, not predecision prediction.")
    lines.append("- If entropy predictors do not improve design+RL prediction, frame them as characterization/validation rather than predictive augmentation.")
    lines.append("- Strong entropy findings should be placed after the main ERN/Pe/theta/beta story.")
    lines.append("")

    (out_dir / "ROBUST_ENTROPY_CANDIDATE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entropy_trial_features", type=str, required=True)
    parser.add_argument("--entropy_subject_summary", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n_perm", type=int, default=2000)
    parser.add_argument("--n_boot", type=int, default=3000)
    parser.add_argument("--encoding_min_n", type=int, default=50)
    parser.add_argument("--min_trial_thresholds", type=str, default="5,10,15,20")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    trial_path = Path(args.entropy_trial_features).expanduser().resolve()
    subj_path = Path(args.entropy_subject_summary).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(trial_path)
    subj = pd.read_csv(subj_path)

    thresholds = [int(x.strip()) for x in args.min_trial_thresholds.split(",") if x.strip()]

    outcome_df = run_outcome_contrasts(
        df=df,
        thresholds=thresholds,
        n_perm=int(args.n_perm),
        n_boot=int(args.n_boot),
        rng=rng,
    )
    enc_obs, enc_null, enc_summary = run_encoding(
        df=df,
        n_perm=int(args.n_perm),
        n_boot=int(args.n_boot),
        rng=rng,
        min_n=int(args.encoding_min_n),
    )
    subj_corr = run_subject_correlations(
        subj=subj,
        n_perm=int(args.n_perm),
        n_boot=int(args.n_boot),
        rng=rng,
    )

    outcome_df.to_csv(out_dir / "robust_entropy_outcome_contrast_summary.csv", index=False)
    enc_obs.to_csv(out_dir / "robust_entropy_encoding_observed_subject_rs.csv", index=False)
    enc_null.to_csv(out_dir / "robust_entropy_encoding_null.csv", index=False)
    enc_summary.to_csv(out_dir / "robust_entropy_encoding_summary.csv", index=False)
    subj_corr.to_csv(out_dir / "robust_entropy_subject_correlation_summary.csv", index=False)

    write_report(out_dir, outcome_df, enc_summary, subj_corr, int(args.n_perm), int(args.n_boot))

    print(f"Wrote robust entropy validation to: {out_dir}")
    print((out_dir / "ROBUST_ENTROPY_CANDIDATE_REPORT.md").read_text(encoding="utf-8")[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
