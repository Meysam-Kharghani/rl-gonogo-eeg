#!/usr/bin/env python3
# FORMALIZED HISTORICAL SOURCE SNAPSHOT
# Evidence status: HISTORICAL_OR_SUPERSEDED_DEVELOPMENT_EVIDENCE
# This file preserves a historical project stage with public-facing terminology standardized for release.
# The SHA-256 of the original executed/development source is recorded in the corresponding provenance registry.

"""
Targeted EEG neural-validation analyses for the EEG-computational study.

This script uses the corrected trial-wise EEG + RL/DDM latent table and runs
prespecified subject-level tests linking computational states to EEG feature
families. It is intended as the next step after the exploratory encoding screen.

Inputs:
- eeg_trial_neural_features_latents_corrected.csv

Outputs:
- targeted_neural_validation_subject_effects.csv
- targeted_neural_validation_group_summary.csv
- targeted_error_monitoring_contrasts.csv
- targeted_post_error_adjustment.csv
- TARGETED_NEURAL_VALIDATION_REPORT.md

Example:
python code/final/run_eeg_targeted_neural_validation_v1.py \
  --features results/eeg_neural_dynamics_v1_2_corrected/eeg_trial_neural_features_latents_corrected.csv \
  --out results/eeg_targeted_neural_validation_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


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


def norm_text(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def safe_log_rt_ms(x: pd.Series) -> pd.Series:
    rt = pd.to_numeric(x, errors="coerce")
    return np.log(rt.where(rt > 0))


def residualize(y: pd.Series, covariates: Optional[pd.DataFrame]) -> pd.Series:
    yy = pd.to_numeric(y, errors="coerce").astype(float)
    if covariates is None or covariates.empty:
        return yy - yy.mean(skipna=True)

    X = covariates.copy()
    # Convert categorical columns to dummies and numeric columns to floats.
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    num_cols = [c for c in X.columns if c not in cat_cols]
    parts = []
    if num_cols:
        num = X[num_cols].apply(pd.to_numeric, errors="coerce")
        parts.append(num)
    if cat_cols:
        parts.append(pd.get_dummies(X[cat_cols].astype(str), drop_first=True, dummy_na=False, dtype=float))

    if not parts:
        return yy - yy.mean(skipna=True)

    XX = pd.concat(parts, axis=1)
    XX = XX.replace([np.inf, -np.inf], np.nan)
    mask = yy.notna() & XX.notna().all(axis=1)

    out = pd.Series(np.nan, index=yy.index, dtype=float)
    if int(mask.sum()) < max(5, XX.shape[1] + 3):
        return out

    Xmat = np.column_stack([np.ones(int(mask.sum())), XX.loc[mask].astype(float).values])
    yvec = yy.loc[mask].astype(float).values

    try:
        beta, *_ = np.linalg.lstsq(Xmat, yvec, rcond=None)
        resid = yvec - Xmat @ beta
        out.loc[mask] = resid
    except Exception:
        return out
    return out


def subject_residual_correlation(
    df: pd.DataFrame,
    predictor: str,
    neural_feature: str,
    subset: str,
    covariate_cols: List[str],
    min_n: int = 25,
) -> List[dict]:
    rows = []

    if subset == "all":
        d = df.copy()
    elif subset == "response":
        d = df[pd.to_numeric(df["response_present"], errors="coerce") == 1].copy()
    elif subset == "fa":
        d = df[df["response_type"].map(norm_text) == "fa"].copy()
    else:
        raise ValueError(f"Unknown subset: {subset}")

    if d.empty or predictor not in d.columns or neural_feature not in d.columns:
        return rows

    for subject, g in d.groupby("subject_id"):
        g = g.copy()
        covs = [c for c in covariate_cols if c in g.columns]
        cov_df = g[covs] if covs else None

        x = residualize(g[predictor], cov_df)
        y = residualize(g[neural_feature], cov_df)
        mask = np.isfinite(x) & np.isfinite(y)

        if int(mask.sum()) < min_n:
            continue
        if float(np.nanstd(x[mask])) <= 1e-12 or float(np.nanstd(y[mask])) <= 1e-12:
            continue

        r, p = stats.pearsonr(x[mask], y[mask])
        rows.append({
            "subject_id": subject,
            "group": g["group"].iloc[0] if "group" in g.columns else "",
            "predictor": predictor,
            "neural_feature": neural_feature,
            "subset": subset,
            "covariates": ",".join(covs),
            "n_trials": int(mask.sum()),
            "r": float(r),
            "p_within_subject": float(p),
            "z_fisher": float(np.arctanh(np.clip(r, -0.999999, 0.999999))),
        })
    return rows


def summarize_subject_effects(subject_effects: pd.DataFrame) -> pd.DataFrame:
    if subject_effects.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["family", "predictor", "neural_feature", "subset", "covariates"]
    for keys, g in subject_effects.groupby(group_cols):
        z = pd.to_numeric(g["z_fisher"], errors="coerce").dropna().values
        if len(z) < 10:
            continue
        t, p = stats.ttest_1samp(z, 0.0)
        mean_z = float(np.mean(z))
        se_z = float(stats.sem(z))
        ci_low_z, ci_high_z = stats.t.interval(0.95, df=len(z)-1, loc=mean_z, scale=se_z) if len(z) > 1 else (np.nan, np.nan)
        rows.append({
            "family": keys[0],
            "predictor": keys[1],
            "neural_feature": keys[2],
            "subset": keys[3],
            "covariates": keys[4],
            "n_subjects": int(len(z)),
            "mean_r": float(np.tanh(mean_z)),
            "ci95_low_r": float(np.tanh(ci_low_z)) if np.isfinite(ci_low_z) else np.nan,
            "ci95_high_r": float(np.tanh(ci_high_z)) if np.isfinite(ci_high_z) else np.nan,
            "t_value": float(t),
            "p_value": float(p),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_fdr_bh"] = fdr_bh(out["p_value"].values)
        out = out.sort_values(["q_value_fdr_bh", "p_value"]).reset_index(drop=True)
    return out


def add_next_trial_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["subject_id", "trial_index_numeric"]).copy()
    out["next_response_present"] = out.groupby("subject_id")["response_present"].shift(-1)
    if "error" in out.columns:
        out["next_error"] = out.groupby("subject_id")["error"].shift(-1)
    else:
        out["next_error"] = np.nan
    out["next_log_rt"] = out.groupby("subject_id")["rt_response_from_target_ms"].shift(-1)
    out["next_log_rt"] = safe_log_rt_ms(out["next_log_rt"])
    if "control_current" in out.columns:
        out["next_control_current"] = out.groupby("subject_id")["control_current"].shift(-1)
        out["delta_control_next_minus_current"] = out["next_control_current"] - out["control_current"]
    if "p_press" in out.columns:
        out["next_p_press"] = out.groupby("subject_id")["p_press"].shift(-1)
        out["delta_p_press_next_minus_current"] = out["next_p_press"] - out["p_press"]
    return out


def build_test_plan() -> List[dict]:
    return [
        {
            "family": "cue_preparatory_control",
            "subset": "all",
            "predictors": ["control_current", "p_press", "actor_qdiff_current"],
            "features": [
                "cue_cnv_late_cz_200_500_mean_uv",
                "cue_cnv_late_cz_200_500_min_uv",
                "cue_cnv_late_cz_200_500_max_uv",
                "cue_cnv_late_cz_200_500_slope_uv_per_s",
                "cue_p3_cz_250_450_mean_uv",
                "cue_p3_cz_250_450_max_uv",
                "cue_n2_cz_200_350_mean_uv",
                "cue_n2_cz_200_350_max_uv",
                "cue_n2_cz_200_350_slope_uv_per_s",
                "cue_n2_fz_200_350_mean_uv",
                "cue_n2_fz_200_350_max_uv",
            ],
            "covariates": ["cue_type", "trial_class", "response_present"],
            "min_n": 80,
        },
        {
            "family": "target_processing_policy",
            "subset": "all",
            "predictors": ["p_press", "actor_qdiff_current", "control_current"],
            "features": [
                "target_n2_fz_200_350_mean_uv",
                "target_n2_fz_200_350_min_uv",
                "target_n2_fz_200_350_slope_uv_per_s",
                "target_n2_cz_200_350_mean_uv",
                "target_n2_cz_200_350_min_uv",
                "target_n2_cz_200_350_slope_uv_per_s",
                "target_p300_cz_250_500_mean_uv",
                "target_p300_cz_250_500_max_uv",
                "target_p300_cz_250_500_slope_uv_per_s",
            ],
            "covariates": ["cue_type", "trial_class", "response_present"],
            "min_n": 80,
        },
        {
            "family": "motor_preparation_policy",
            "subset": "response",
            "predictors": ["p_press", "actor_qdiff_current"],
            "features": [
                "resp_rp_cz_m300_0_mean_uv",
                "resp_rp_cz_m300_0_min_uv",
                "resp_rp_cz_m300_0_max_uv",
                "resp_rp_cz_m300_0_slope_uv_per_s",
            ],
            "covariates": ["cue_type", "trial_class", "log_rt"],
            "min_n": 40,
        },
        {
            "family": "response_monitoring_policy",
            "subset": "response",
            "predictors": ["p_press", "actor_qdiff_current"],
            "features": [
                "resp_ern_fz_50_150_mean_uv",
                "resp_ern_fz_50_150_min_uv",
                "resp_ern_fz_50_150_max_uv",
                "resp_pe_fz_200_400_mean_uv",
                "resp_pe_fz_200_400_max_uv",
                "resp_pe_cz_200_400_mean_uv",
                "resp_pe_cz_200_400_max_uv",
            ],
            "covariates": ["cue_type", "trial_class", "log_rt"],
            "min_n": 40,
        },
    ]


def run_targeted_correlations(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["log_rt"] = safe_log_rt_ms(df["rt_response_from_target_ms"])

    all_rows = []
    for spec in build_test_plan():
        for pred in spec["predictors"]:
            if pred not in df.columns:
                continue
            for feat in spec["features"]:
                if feat not in df.columns:
                    continue
                rows = subject_residual_correlation(
                    df=df,
                    predictor=pred,
                    neural_feature=feat,
                    subset=spec["subset"],
                    covariate_cols=spec["covariates"],
                    min_n=int(spec["min_n"]),
                )
                for r in rows:
                    r["family"] = spec["family"]
                all_rows.extend(rows)

    subj = pd.DataFrame(all_rows)
    summary = summarize_subject_effects(subj)
    return subj, summary


def fa_vs_hit_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "resp_ern_fz_50_150_mean_uv",
        "resp_ern_fz_50_150_min_uv",
        "resp_ern_fz_50_150_max_uv",
        "resp_pe_fz_200_400_mean_uv",
        "resp_pe_fz_200_400_max_uv",
        "resp_pe_cz_200_400_mean_uv",
        "resp_pe_cz_200_400_max_uv",
    ]
    rows = []
    d = df[df["response_type"].map(norm_text).isin(["fa", "hit"])].copy()
    for feat in features:
        if feat not in d.columns:
            continue
        diffs = []
        for subject, g in d.groupby("subject_id"):
            fa = pd.to_numeric(g.loc[g["response_type"].map(norm_text) == "fa", feat], errors="coerce").dropna()
            hit = pd.to_numeric(g.loc[g["response_type"].map(norm_text) == "hit", feat], errors="coerce").dropna()
            if len(fa) >= 5 and len(hit) >= 20:
                diffs.append({
                    "subject_id": subject,
                    "feature": feat,
                    "fa_mean": float(fa.mean()),
                    "hit_mean": float(hit.mean()),
                    "fa_minus_hit": float(fa.mean() - hit.mean()),
                    "n_fa": int(len(fa)),
                    "n_hit": int(len(hit)),
                })
        if len(diffs) >= 10:
            vals = np.array([x["fa_minus_hit"] for x in diffs], dtype=float)
            t, p = stats.ttest_1samp(vals, 0.0)
            rows.append({
                "contrast": "false_alarm_minus_hit",
                "feature": feat,
                "n_subjects": int(len(vals)),
                "mean_difference_uv": float(np.mean(vals)),
                "ci95_low_uv": float(stats.t.interval(0.95, df=len(vals)-1, loc=np.mean(vals), scale=stats.sem(vals))[0]),
                "ci95_high_uv": float(stats.t.interval(0.95, df=len(vals)-1, loc=np.mean(vals), scale=stats.sem(vals))[1]),
                "t_value": float(t),
                "p_value": float(p),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_fdr_bh"] = fdr_bh(out["p_value"].values)
        out = out.sort_values(["q_value_fdr_bh", "p_value"]).reset_index(drop=True)
    return out


def post_error_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    d = add_next_trial_columns(df)
    d = d[d["response_type"].map(norm_text) == "fa"].copy()

    predictors = [
        "resp_ern_fz_50_150_mean_uv",
        "resp_ern_fz_50_150_min_uv",
        "resp_pe_fz_200_400_mean_uv",
        "resp_pe_cz_200_400_mean_uv",
    ]
    outcomes = [
        "next_log_rt",
        "next_response_present",
        "next_error",
        "delta_control_next_minus_current",
        "delta_p_press_next_minus_current",
    ]
    rows = []

    for pred in predictors:
        if pred not in d.columns:
            continue
        for outcome in outcomes:
            if outcome not in d.columns:
                continue
            subj_rows = []
            for subject, g in d.groupby("subject_id"):
                x = pd.to_numeric(g[pred], errors="coerce")
                y = pd.to_numeric(g[outcome], errors="coerce")
                mask = np.isfinite(x) & np.isfinite(y)
                if int(mask.sum()) < 8:
                    continue
                if float(np.nanstd(x[mask])) <= 1e-12 or float(np.nanstd(y[mask])) <= 1e-12:
                    continue
                r, p = stats.pearsonr(x[mask], y[mask])
                subj_rows.append(np.arctanh(np.clip(float(r), -0.999999, 0.999999)))
            if len(subj_rows) >= 10:
                z = np.array(subj_rows)
                t, p = stats.ttest_1samp(z, 0.0)
                rows.append({
                    "subset": "false_alarm_trials",
                    "predictor_feature": pred,
                    "next_trial_outcome": outcome,
                    "n_subjects": int(len(z)),
                    "mean_r": float(np.tanh(np.mean(z))),
                    "t_value": float(t),
                    "p_value": float(p),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_fdr_bh"] = fdr_bh(out["p_value"].values)
        out = out.sort_values(["q_value_fdr_bh", "p_value"]).reset_index(drop=True)
    return out


def write_report(out_dir: Path, df: pd.DataFrame, summary: pd.DataFrame, contrasts: pd.DataFrame, posterr: pd.DataFrame) -> None:
    lines = []
    lines.append("# Targeted EEG neural-validation report")
    lines.append("")
    lines.append("This report summarizes prespecified subject-level tests linking RL/DDM computational states to EEG feature families.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Trial rows: {len(df)}")
    lines.append(f"- Subjects: {df['subject_id'].nunique() if 'subject_id' in df.columns else 'unknown'}")
    lines.append("")
    lines.append("## Targeted computational-state encoding")
    lines.append("")
    lines.append(f"- Tested targeted pairs: {len(summary)}")
    lines.append(f"- FDR q < 0.05 pairs: {int((summary['q_value_fdr_bh'] < 0.05).sum()) if not summary.empty else 0}")
    if not summary.empty:
        lines.append("")
        lines.append("| Family | Predictor | Neural feature | subset | mean r | 95% CI | p | q |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|")
        for _, row in summary.head(20).iterrows():
            lines.append(
                f"| {row['family']} | {row['predictor']} | {row['neural_feature']} | {row['subset']} | "
                f"{row['mean_r']:.4f} | [{row['ci95_low_r']:.4f}, {row['ci95_high_r']:.4f}] | "
                f"{row['p_value']:.3g} | {row['q_value_fdr_bh']:.3g} |"
            )
    lines.append("")
    lines.append("## False-alarm versus hit response-locked contrasts")
    lines.append("")
    lines.append(f"- Tested contrasts: {len(contrasts)}")
    lines.append(f"- FDR q < 0.05 contrasts: {int((contrasts['q_value_fdr_bh'] < 0.05).sum()) if not contrasts.empty else 0}")
    if not contrasts.empty:
        lines.append("")
        lines.append("| Feature | FA - Hit mean difference | 95% CI | p | q |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, row in contrasts.iterrows():
            lines.append(
                f"| {row['feature']} | {row['mean_difference_uv']:.4f} | "
                f"[{row['ci95_low_uv']:.4f}, {row['ci95_high_uv']:.4f}] | "
                f"{row['p_value']:.3g} | {row['q_value_fdr_bh']:.3g} |"
            )
    lines.append("")
    lines.append("## Post-error adjustment from false-alarm EEG")
    lines.append("")
    lines.append(f"- Tested pairs: {len(posterr)}")
    lines.append(f"- FDR q < 0.05 pairs: {int((posterr['q_value_fdr_bh'] < 0.05).sum()) if not posterr.empty else 0}")
    if not posterr.empty:
        lines.append("")
        lines.append("| Predictor feature | Next-trial outcome | mean r | p | q |")
        lines.append("|---|---|---:|---:|---:|")
        for _, row in posterr.head(20).iterrows():
            lines.append(
                f"| {row['predictor_feature']} | {row['next_trial_outcome']} | "
                f"{row['mean_r']:.4f} | {row['p_value']:.3g} | {row['q_value_fdr_bh']:.3g} |"
            )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Use this report to decide which EEG-computational links are strong enough for a manuscript-level neural-validation section.")
    lines.append("Effects surviving subject-level inference and FDR correction within targeted feature families are the most defensible.")
    lines.append("")
    (out_dir / "TARGETED_NEURAL_VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    features_path = Path(args.features).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_path)
    required = ["subject_id", "response_type", "response_present", "trial_index_numeric"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input feature table missing required columns: {missing}")

    subj_effects, summary = run_targeted_correlations(df)
    contrasts = fa_vs_hit_contrasts(df)
    posterr = post_error_adjustment(df)

    subj_effects.to_csv(out_dir / "targeted_neural_validation_subject_effects.csv", index=False)
    summary.to_csv(out_dir / "targeted_neural_validation_group_summary.csv", index=False)
    contrasts.to_csv(out_dir / "targeted_error_monitoring_contrasts.csv", index=False)
    posterr.to_csv(out_dir / "targeted_post_error_adjustment.csv", index=False)

    write_report(out_dir, df, summary, contrasts, posterr)

    print(f"Wrote outputs to: {out_dir}")
    if not summary.empty:
        print(summary.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
