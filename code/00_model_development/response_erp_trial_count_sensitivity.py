#!/usr/bin/env python3
# FORMALIZED HISTORICAL SOURCE SNAPSHOT
# Evidence status: HISTORICAL_OR_SUPERSEDED_DEVELOPMENT_EVIDENCE
# This file preserves a historical project stage with public-facing terminology standardized for release.
# The SHA-256 of the original executed/development source is recorded in the corresponding provenance registry.

"""
Minimum-trial sensitivity checks for ERP waveform contrasts.

This script uses outputs from run_eeg_erp_waveform_analysis_v1.py and tests
whether key ERP contrasts remain stable when subjects with very few trials in
a condition are excluded.

Example:
python code/final/run_eeg_erp_min_trial_sensitivity_v1.py \
  --erp_dir results/eeg_erp_waveform_analysis_v1 \
  --out results/eeg_erp_min_trial_sensitivity_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


CONTRASTS = [
    {"name": "nogo_fa_minus_go_hit", "a": "nogo_fa", "b": "go_hit", "thresholds": [1, 5, 10, 15, 20]},
    {"name": "nogo_cr_minus_nogo_fa", "a": "nogo_cr", "b": "nogo_fa", "thresholds": [1, 5, 10, 15, 20]},
    {"name": "go_hit_minus_go_miss", "a": "go_hit", "b": "go_miss", "thresholds": [1, 5, 10, 15, 20]},
]

PRIMARY_FEATURES = [
    "resp_ern_fz_50_150",
    "resp_ern_fz_min_50_150",
    "resp_pe_fz_200_400",
    "resp_pe_cz_200_400",
    "resp_pe_pz_200_400",
    "resp_rp_cz_m300_0",
    "target_p300_pz_250_500",
    "target_p300_cz_250_500",
    "target_n2_fz_200_350",
    "target_n2_cz_200_350",
    "cue_cnv_200_500",
    "cue_p3_250_450",
    "cue_p3_pz_250_450",
    "cue_n2_fz_200_350",
]


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


def get_count_col(feature: str) -> str:
    return f"{feature}_n_trials"


def run_sensitivity(window_features: pd.DataFrame, subject_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Add condition count columns to the window-feature table.
    counts = subject_summary[[
        "subject_id", "n_go_hit", "n_go_miss", "n_nogo_cr", "n_nogo_fa"
    ]].copy()
    df = window_features.merge(counts, on="subject_id", how="left", suffixes=("", "_summary"))

    for contrast in CONTRASTS:
        a = contrast["a"]
        b = contrast["b"]
        count_a = f"n_{a}"
        count_b = f"n_{b}"

        for feature in PRIMARY_FEATURES:
            if feature not in df.columns:
                continue

            wide = df.pivot_table(
                index=["subject_id", "group", count_a, count_b],
                columns="outcome",
                values=feature,
                aggfunc="mean",
            ).reset_index()

            if a not in wide.columns or b not in wide.columns:
                continue

            wide["difference_uv"] = wide[a] - wide[b]

            for min_trials in contrast["thresholds"]:
                keep = wide[
                    (pd.to_numeric(wide[count_a], errors="coerce") >= min_trials) &
                    (pd.to_numeric(wide[count_b], errors="coerce") >= min_trials) &
                    wide["difference_uv"].notna()
                ].copy()

                if len(keep) < 8:
                    continue

                vals = keep["difference_uv"].astype(float).values
                t, p = stats.ttest_1samp(vals, 0.0)
                ci = stats.t.interval(0.95, df=len(vals)-1, loc=float(np.mean(vals)), scale=float(stats.sem(vals))) if len(vals) > 1 else (np.nan, np.nan)

                rows.append({
                    "contrast": contrast["name"],
                    "feature": feature,
                    "min_trials_per_condition": int(min_trials),
                    "n_subjects": int(len(vals)),
                    "mean_difference_uv": float(np.mean(vals)),
                    "ci95_low_uv": float(ci[0]),
                    "ci95_high_uv": float(ci[1]),
                    "t_value": float(t),
                    "p_value": float(p),
                })

                # group-specific estimates
                for group, gd in keep.groupby("group"):
                    if len(gd) < 8:
                        continue
                    gvals = gd["difference_uv"].astype(float).values
                    gt, gp = stats.ttest_1samp(gvals, 0.0)
                    gci = stats.t.interval(0.95, df=len(gvals)-1, loc=float(np.mean(gvals)), scale=float(stats.sem(gvals))) if len(gvals) > 1 else (np.nan, np.nan)
                    rows.append({
                        "contrast": f"{contrast['name']}_{group}",
                        "feature": feature,
                        "min_trials_per_condition": int(min_trials),
                        "n_subjects": int(len(gvals)),
                        "mean_difference_uv": float(np.mean(gvals)),
                        "ci95_low_uv": float(gci[0]),
                        "ci95_high_uv": float(gci[1]),
                        "t_value": float(gt),
                        "p_value": float(gp),
                    })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_fdr_bh"] = fdr_bh(out["p_value"].values)
        out = out.sort_values(["contrast", "feature", "min_trials_per_condition"]).reset_index(drop=True)
    return out


def summarize_stability(stats_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if stats_df.empty:
        return pd.DataFrame()

    for (contrast, feature), g in stats_df.groupby(["contrast", "feature"]):
        main = g[~g["contrast"].str.endswith("_high") & ~g["contrast"].str.endswith("_low")].copy()
        if main.empty:
            continue
        rows.append({
            "contrast": contrast,
            "feature": feature,
            "n_thresholds_tested": int(main["min_trials_per_condition"].nunique()),
            "max_min_trial_threshold_tested": int(main["min_trials_per_condition"].max()),
            "all_effects_same_direction": bool((main["mean_difference_uv"] > 0).all() or (main["mean_difference_uv"] < 0).all()),
            "n_thresholds_q_lt_05": int((main["q_value_fdr_bh"] < 0.05).sum()),
            "smallest_p": float(main["p_value"].min()),
            "smallest_q": float(main["q_value_fdr_bh"].min()),
            "effect_at_highest_threshold_uv": float(main.sort_values("min_trials_per_condition").iloc[-1]["mean_difference_uv"]),
            "n_subjects_at_highest_threshold": int(main.sort_values("min_trials_per_condition").iloc[-1]["n_subjects"]),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["smallest_q", "smallest_p"]).reset_index(drop=True)
    return out


def write_report(out_dir: Path, stats_df: pd.DataFrame, stability: pd.DataFrame) -> None:
    lines = []
    lines.append("# ERP minimum-trial sensitivity report")
    lines.append("")
    lines.append("This report checks whether ERP contrast effects survive after excluding subjects with too few trials in either condition.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Sensitivity rows: {len(stats_df)}")
    lines.append(f"- Stability rows: {len(stability)}")
    if not stability.empty:
        lines.append("")
        lines.append("Top stable effects:")
        lines.append("")
        lines.append("| Contrast | Feature | thresholds q<.05 | highest threshold | effect at highest threshold | n at highest threshold | min q |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for _, row in stability.head(20).iterrows():
            lines.append(
                f"| {row['contrast']} | {row['feature']} | {int(row['n_thresholds_q_lt_05'])} | "
                f"{int(row['max_min_trial_threshold_tested'])} | {row['effect_at_highest_threshold_uv']:.3f} | "
                f"{int(row['n_subjects_at_highest_threshold'])} | {row['smallest_q']:.3g} |"
            )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Effects that keep the same direction and remain significant at stricter minimum-trial thresholds are safer for manuscript claims.")
    lines.append("Effects based on conditions with very few trials should be described as exploratory unless they pass this sensitivity check.")
    lines.append("")
    (out_dir / "ERP_MIN_TRIAL_SENSITIVITY_REPORT.md").write_text("\\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--erp_dir", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    args = parser.parse_args()

    erp_dir = Path(args.erp_dir).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    window_features = pd.read_csv(erp_dir / "erp_subject_window_features.csv")
    subject_summary = pd.read_csv(erp_dir / "erp_subject_processing_summary.csv")

    stats_df = run_sensitivity(window_features, subject_summary)
    stability = summarize_stability(stats_df)

    stats_df.to_csv(out_dir / "erp_min_trial_sensitivity_stats.csv", index=False)
    stability.to_csv(out_dir / "erp_min_trial_sensitivity_stability_summary.csv", index=False)
    write_report(out_dir, stats_df, stability)

    print(f"Wrote outputs to: {out_dir}")
    if not stability.empty:
        print(stability.head(20).to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
