#!/usr/bin/env python3
# FORMALIZED HISTORICAL SOURCE SNAPSHOT
# Evidence status: HISTORICAL_OR_SUPERSEDED_DEVELOPMENT_EVIDENCE
# This file preserves a historical project stage with public-facing terminology standardized for release.
# The SHA-256 of the original executed/development source is recorded in the corresponding provenance registry.

"""
Permutation validation for EEG feature decoding.

This script repeats the feature-level decoding analysis under label permutations
to estimate empirical null distributions. It is designed as a validation layer
after run_eeg_feature_decoding_v1.py.

Two permutation modes are used:
- within_subject: labels are permuted within each subject, preserving each
  participant's class balance.
- global: labels are permuted across the full analysis table.

Outputs:
- eeg_decoding_permutation_observed.csv
- eeg_decoding_permutation_null.csv
- eeg_decoding_permutation_summary.csv
- EEG_DECODING_PERMUTATION_REPORT.md

Example:
python code/final/run_eeg_decoding_permutation_validation_v1.py \
  --erp_features results/eeg_neural_dynamics_v1_2_corrected/eeg_trial_neural_features_latents_corrected.csv \
  --tf_features results/eeg_time_frequency_analysis_v1/tf_trial_bandpower_features.csv \
  --out results/eeg_decoding_permutation_validation_v1 \
  --n_perm 200
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TASKS = [
    ("press_prediction_cue_target", ["tf_cue_target", "combined_cue_target", "erp_cue_target"]),
    ("nogo_failure_prediction_cue_target", ["tf_cue_target", "combined_cue_target", "erp_cue_target"]),
    ("go_omission_prediction_cue_target", ["combined_cue_target", "erp_cue_target", "tf_cue_target"]),
    ("error_monitoring_signature_response", ["combined_response", "erp_response", "tf_response"]),
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


def load_and_merge(erp_path: Path, tf_path: Optional[Path]) -> pd.DataFrame:
    erp = pd.read_csv(erp_path)
    if "subject_id_norm" not in erp.columns:
        erp["subject_id_norm"] = erp["subject_id"].map(normalize_subject_id)
    if "trial_index_numeric" not in erp.columns:
        erp["trial_index_numeric"] = pd.to_numeric(erp["trial_index"], errors="coerce")

    erp["trial_class_norm"] = erp["trial_class"].map(norm_text)
    erp["response_type_norm"] = erp["response_type"].map(norm_text)
    erp["outcome"] = erp["response_type_norm"].map({
        "hit": "go_hit",
        "miss": "go_miss",
        "cr": "nogo_cr",
        "correct rejection": "nogo_cr",
        "correct_rejection": "nogo_cr",
        "fa": "nogo_fa",
        "false alarm": "nogo_fa",
        "false_alarm": "nogo_fa",
    })
    press = pd.to_numeric(erp["response_present"], errors="coerce").fillna(0).astype(int)
    is_go = erp["trial_class_norm"] == "go"
    is_nogo = erp["trial_class_norm"] == "nogo"
    erp.loc[erp["outcome"].isna() & is_go & (press == 1), "outcome"] = "go_hit"
    erp.loc[erp["outcome"].isna() & is_go & (press == 0), "outcome"] = "go_miss"
    erp.loc[erp["outcome"].isna() & is_nogo & (press == 0), "outcome"] = "nogo_cr"
    erp.loc[erp["outcome"].isna() & is_nogo & (press == 1), "outcome"] = "nogo_fa"

    if tf_path is not None and tf_path.exists():
        tf = pd.read_csv(tf_path)
        if "subject_id_norm" not in tf.columns:
            tf["subject_id_norm"] = tf["subject_id"].map(normalize_subject_id)
        if "trial_index_numeric" not in tf.columns:
            tf["trial_index_numeric"] = pd.to_numeric(tf["trial_index"], errors="coerce")

        key_cols = ["subject_id_norm", "trial_index_numeric"]
        meta_tf_cols = {
            "subject_id", "group", "trial_index", "cue_type", "trial_class",
            "response_present", "response_type", "outcome", "rt_response_from_target_ms",
            "latent_trial_index_offset_used"
        }
        tf_feature_cols = [
            c for c in tf.columns
            if c not in meta_tf_cols and c not in key_cols
            and pd.api.types.is_numeric_dtype(tf[c])
        ]
        erp = erp.merge(tf[key_cols + tf_feature_cols], on=key_cols, how="left", suffixes=("", "_tf"))

    return erp


def select_features(df: pd.DataFrame, feature_family: str) -> List[str]:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    meta_exclude = {
        "trial_index", "trial_index_numeric", "response_present", "rt_response_from_target_ms",
        "cue_onset_sec", "target_onset_sec", "response_onset_from_metadata_sec",
        "response_marker_onset_sec", "response_marker_error_ms",
        "condition_cue_match", "condition_trial_class_match",
        "p_press", "control_current", "actor_qdiff_current", "belief_nogo_current",
        "associability_current", "is_nogo", "action_press", "correct", "error", "fa",
        "latent_trial_index_offset_used",
    }

    erp_cue_target = [
        c for c in numeric
        if (c.startswith("cue_") or c.startswith("target_"))
        and c not in meta_exclude
        and (
            c.endswith("_mean_uv") or c.endswith("_min_uv") or c.endswith("_max_uv")
            or c.endswith("_minlat_ms") or c.endswith("_maxlat_ms") or c.endswith("_slope_uv_per_s")
        )
    ]
    erp_response = [
        c for c in numeric
        if c.startswith("resp_")
        and c not in meta_exclude
        and (
            c.endswith("_mean_uv") or c.endswith("_min_uv") or c.endswith("_max_uv")
            or c.endswith("_minlat_ms") or c.endswith("_maxlat_ms") or c.endswith("_slope_uv_per_s")
        )
    ]
    tf_cue_target = [
        c for c in numeric
        if (c.startswith("cue_") or c.startswith("target_"))
        and c not in meta_exclude
        and not c.endswith(("_mean_uv", "_min_uv", "_max_uv", "_minlat_ms", "_maxlat_ms", "_slope_uv_per_s"))
    ]
    tf_response = [
        c for c in numeric
        if c.startswith("resp_")
        and c not in meta_exclude
        and not c.endswith(("_mean_uv", "_min_uv", "_max_uv", "_minlat_ms", "_maxlat_ms", "_slope_uv_per_s"))
    ]

    if feature_family == "erp_cue_target":
        cols = erp_cue_target
    elif feature_family == "tf_cue_target":
        cols = tf_cue_target
    elif feature_family == "combined_cue_target":
        cols = sorted(set(erp_cue_target + tf_cue_target))
    elif feature_family == "erp_response":
        cols = erp_response
    elif feature_family == "tf_response":
        cols = tf_response
    elif feature_family == "combined_response":
        cols = sorted(set(erp_response + tf_response))
    else:
        raise ValueError(f"Unknown feature family: {feature_family}")

    usable = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() >= 20 and x.nunique(dropna=True) > 1:
            usable.append(c)
    return sorted(usable)


def build_task(df: pd.DataFrame, task_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    d = df.copy()
    d["outcome"] = d["outcome"].map(norm_text)

    if task_name == "press_prediction_cue_target":
        d = d[d["response_present"].notna()].copy()
        y = pd.to_numeric(d["response_present"], errors="coerce").fillna(0).astype(int)
    elif task_name == "nogo_failure_prediction_cue_target":
        d = d[d["outcome"].isin(["nogo_cr", "nogo_fa"])].copy()
        y = (d["outcome"] == "nogo_fa").astype(int)
    elif task_name == "go_omission_prediction_cue_target":
        d = d[d["outcome"].isin(["go_hit", "go_miss"])].copy()
        y = (d["outcome"] == "go_miss").astype(int)
    elif task_name == "error_monitoring_signature_response":
        d = d[d["outcome"].isin(["go_hit", "nogo_fa"])].copy()
        y = (d["outcome"] == "nogo_fa").astype(int)
    else:
        raise ValueError(task_name)

    return d.reset_index(drop=True), y.reset_index(drop=True)


def make_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.3,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=2026,
        )),
    ])


def safe_auc(y_true, y_score) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def grouped_cv_score(d: pd.DataFrame, y: pd.Series, feature_cols: List[str], random_state: int = 2026) -> Tuple[float, float]:
    X = d[feature_cols].replace([np.inf, -np.inf], np.nan)
    yy = y.astype(int).values
    groups = d["subject_id_norm"].values

    if len(np.unique(yy)) < 2 or len(np.unique(groups)) < 5:
        return np.nan, np.nan

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    baccs, aucs = [], []
    for train_idx, test_idx in cv.split(X, yy, groups):
        if len(np.unique(yy[train_idx])) < 2 or len(np.unique(yy[test_idx])) < 2:
            continue
        model = make_model()
        model.fit(X.iloc[train_idx], yy[train_idx])
        prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        baccs.append(balanced_accuracy_score(yy[test_idx], pred))
        aucs.append(safe_auc(yy[test_idx], prob))
    return float(np.nanmean(baccs)), float(np.nanmean(aucs))


def within_subject_score(d: pd.DataFrame, y: pd.Series, feature_cols: List[str], min_class_n: int = 8) -> Tuple[float, float, int]:
    tmp = d.copy()
    tmp["label"] = y.astype(int).values
    subj_bacc, subj_auc = [], []

    for subject, g in tmp.groupby("subject_id_norm"):
        labels = g["label"].astype(int).values
        counts = np.bincount(labels, minlength=2)
        if counts.min() < min_class_n:
            continue
        n_splits = int(min(5, counts.min()))
        if n_splits < 2:
            continue

        X = g[feature_cols].replace([np.inf, -np.inf], np.nan)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2026)
        baccs, aucs = [], []
        for train_idx, test_idx in cv.split(X, labels):
            if len(np.unique(labels[train_idx])) < 2 or len(np.unique(labels[test_idx])) < 2:
                continue
            model = make_model()
            model.fit(X.iloc[train_idx], labels[train_idx])
            prob = model.predict_proba(X.iloc[test_idx])[:, 1]
            pred = (prob >= 0.5).astype(int)
            baccs.append(balanced_accuracy_score(labels[test_idx], pred))
            aucs.append(safe_auc(labels[test_idx], prob))
        if baccs:
            subj_bacc.append(float(np.nanmean(baccs)))
            subj_auc.append(float(np.nanmean(aucs)))

    return float(np.nanmean(subj_bacc)), float(np.nanmean(subj_auc)), int(len(subj_bacc))


def permute_labels(y: pd.Series, subjects: pd.Series, rng: np.random.Generator, mode: str) -> pd.Series:
    y_perm = y.copy().reset_index(drop=True)
    subjects = subjects.reset_index(drop=True)

    if mode == "global":
        vals = y_perm.values.copy()
        rng.shuffle(vals)
        return pd.Series(vals)

    if mode == "within_subject":
        vals = y_perm.values.copy()
        for subject in subjects.unique():
            idx = np.where(subjects.values == subject)[0]
            shuffled = vals[idx].copy()
            rng.shuffle(shuffled)
            vals[idx] = shuffled
        return pd.Series(vals)

    raise ValueError(f"Unknown permutation mode: {mode}")


def empirical_p(observed: float, null_vals: np.ndarray) -> float:
    null_vals = np.asarray(null_vals, dtype=float)
    null_vals = null_vals[np.isfinite(null_vals)]
    if not np.isfinite(observed) or len(null_vals) == 0:
        return np.nan
    # One-sided: performance above null.
    return float((np.sum(null_vals >= observed) + 1) / (len(null_vals) + 1))


def run_analysis(df: pd.DataFrame, n_perm: int, seed: int, min_class_n: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    observed_rows = []
    null_rows = []

    for task, families in TASKS:
        d, y = build_task(df, task)
        if len(d) == 0 or len(np.unique(y)) < 2:
            continue

        for family in families:
            feature_cols = select_features(d, family)
            if not feature_cols:
                continue

            obs_group_bacc, obs_group_auc = grouped_cv_score(d, y, feature_cols)
            obs_subj_bacc, obs_subj_auc, n_subjects = within_subject_score(d, y, feature_cols, min_class_n=min_class_n)

            observed_rows.extend([
                {"cv_type": "subject_grouped_cv", "task": task, "feature_family": family, "metric": "balanced_accuracy", "observed_score": obs_group_bacc, "n_features": len(feature_cols), "n_subjects": d["subject_id_norm"].nunique()},
                {"cv_type": "subject_grouped_cv", "task": task, "feature_family": family, "metric": "roc_auc", "observed_score": obs_group_auc, "n_features": len(feature_cols), "n_subjects": d["subject_id_norm"].nunique()},
                {"cv_type": "within_subject_cv", "task": task, "feature_family": family, "metric": "balanced_accuracy", "observed_score": obs_subj_bacc, "n_features": len(feature_cols), "n_subjects": n_subjects},
                {"cv_type": "within_subject_cv", "task": task, "feature_family": family, "metric": "roc_auc", "observed_score": obs_subj_auc, "n_features": len(feature_cols), "n_subjects": n_subjects},
            ])

            for perm_idx in range(1, n_perm + 1):
                for mode in ["within_subject", "global"]:
                    yp = permute_labels(y, d["subject_id_norm"], rng, mode=mode)

                    gb, ga = grouped_cv_score(d, yp, feature_cols, random_state=seed + perm_idx)
                    sb, sa, ns = within_subject_score(d, yp, feature_cols, min_class_n=min_class_n)

                    null_rows.extend([
                        {"permutation_mode": mode, "perm_idx": perm_idx, "cv_type": "subject_grouped_cv", "task": task, "feature_family": family, "metric": "balanced_accuracy", "null_score": gb},
                        {"permutation_mode": mode, "perm_idx": perm_idx, "cv_type": "subject_grouped_cv", "task": task, "feature_family": family, "metric": "roc_auc", "null_score": ga},
                        {"permutation_mode": mode, "perm_idx": perm_idx, "cv_type": "within_subject_cv", "task": task, "feature_family": family, "metric": "balanced_accuracy", "null_score": sb},
                        {"permutation_mode": mode, "perm_idx": perm_idx, "cv_type": "within_subject_cv", "task": task, "feature_family": family, "metric": "roc_auc", "null_score": sa},
                    ])

            print(f"Finished {task} | {family} with {n_perm} permutations.")

    observed = pd.DataFrame(observed_rows)
    null = pd.DataFrame(null_rows)

    summary_rows = []
    for _, obs in observed.iterrows():
        sub = null[
            (null["cv_type"] == obs["cv_type"]) &
            (null["task"] == obs["task"]) &
            (null["feature_family"] == obs["feature_family"]) &
            (null["metric"] == obs["metric"])
        ]
        for mode, g in sub.groupby("permutation_mode"):
            null_vals = pd.to_numeric(g["null_score"], errors="coerce").dropna().values
            summary_rows.append({
                "permutation_mode": mode,
                "cv_type": obs["cv_type"],
                "task": obs["task"],
                "feature_family": obs["feature_family"],
                "metric": obs["metric"],
                "observed_score": float(obs["observed_score"]),
                "null_mean": float(np.mean(null_vals)) if len(null_vals) else np.nan,
                "null_sd": float(np.std(null_vals, ddof=1)) if len(null_vals) > 1 else np.nan,
                "empirical_p_one_sided": empirical_p(float(obs["observed_score"]), null_vals),
                "n_permutations": int(len(null_vals)),
                "n_features": int(obs["n_features"]),
                "n_subjects": int(obs["n_subjects"]),
            })

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["q_value_fdr_bh"] = fdr_bh(summary["empirical_p_one_sided"].values)
        summary = summary.sort_values(["q_value_fdr_bh", "empirical_p_one_sided"]).reset_index(drop=True)

    return observed, null, summary


def write_report(out_dir: Path, summary: pd.DataFrame, n_perm: int) -> None:
    lines = []
    lines.append("# EEG decoding permutation-validation report")
    lines.append("")
    lines.append(f"This report compares observed decoding performance against empirical null distributions from {n_perm} permutations.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Summary rows: {len(summary)}")
    lines.append(f"- Empirical FDR q < 0.05 rows: {int((summary['q_value_fdr_bh'] < 0.05).sum()) if not summary.empty else 0}")
    if not summary.empty:
        lines.append("")
        lines.append("| Permutation mode | CV type | Task | Feature family | Metric | observed | null mean | empirical p | q |")
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|")
        for _, row in summary.head(40).iterrows():
            lines.append(
                f"| {row['permutation_mode']} | {row['cv_type']} | {row['task']} | {row['feature_family']} | "
                f"{row['metric']} | {row['observed_score']:.3f} | {row['null_mean']:.3f} | "
                f"{row['empirical_p_one_sided']:.4f} | {row['q_value_fdr_bh']:.4f} |"
            )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Effects that survive within-subject label permutation are safer as within-person neural decoding claims.")
    lines.append("- Effects that survive subject-grouped CV and permutation are safer as held-out-participant generalization claims.")
    lines.append("- Cue/target tasks can support cautious predictive claims; response-locked FA-vs-Hit decoding supports post-response error-monitoring signatures.")
    lines.append("")
    (out_dir / "EEG_DECODING_PERMUTATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--erp_features", type=str, required=True)
    parser.add_argument("--tf_features", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n_perm", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min_class_n_subject_cv", type=int, default=8)
    args = parser.parse_args()

    erp_path = Path(args.erp_features).expanduser().resolve()
    tf_path = Path(args.tf_features).expanduser().resolve() if args.tf_features else None
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(erp_path, tf_path)
    observed, null, summary = run_analysis(
        df=df,
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        min_class_n=int(args.min_class_n_subject_cv),
    )

    observed.to_csv(out_dir / "eeg_decoding_permutation_observed.csv", index=False)
    null.to_csv(out_dir / "eeg_decoding_permutation_null.csv", index=False)
    summary.to_csv(out_dir / "eeg_decoding_permutation_summary.csv", index=False)
    write_report(out_dir, summary, int(args.n_perm))

    print(f"Wrote outputs to: {out_dir}")
    if not summary.empty:
        print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
