#!/usr/bin/env python3
# FORMALIZED HISTORICAL SOURCE SNAPSHOT
# Evidence status: HISTORICAL_OR_SUPERSEDED_DEVELOPMENT_EVIDENCE
# This file preserves a historical project stage with public-facing terminology standardized for release.
# The SHA-256 of the original executed/development source is recorded in the corresponding provenance registry.

"""
Feature-level EEG decoding analyses for the EEG-computational study.

This script tests whether trial-wise EEG features can predict behavior or
neural event categories. It uses already-extracted ERP and optional
time-frequency features, so it does not reread raw EEG.

Main tasks:
1. press_prediction_cue_target:
   Can cue/target EEG predict whether the participant will press?
2. nogo_failure_prediction_cue_target:
   Can cue/target EEG predict NoGo false alarm vs correct rejection?
3. go_omission_prediction_cue_target:
   Can cue/target EEG predict Go miss vs hit?
4. error_monitoring_signature_response:
   Can response-locked EEG distinguish NoGo false alarms from Go hits?

Outputs:
- eeg_decoding_group_cv_results.csv
- eeg_decoding_subject_cv_results.csv
- eeg_decoding_summary.csv
- EEG_DECODING_REPORT.md

Example:
python code/final/run_eeg_feature_decoding_v1.py \
  --erp_features results/eeg_neural_dynamics_v1_2_corrected/eeg_trial_neural_features_latents_corrected.csv \
  --tf_features results/eeg_time_frequency_analysis_v1/tf_trial_bandpower_features.csv \
  --out results/eeg_feature_decoding_v1
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    # fallback from trial_class and response_present
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
        tf_keep = tf[key_cols + tf_feature_cols].copy()
        df = erp.merge(tf_keep, on=key_cols, how="left", suffixes=("", "_tf"))
    else:
        df = erp.copy()

    return df


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
        return sorted(erp_cue_target)
    if feature_family == "tf_cue_target":
        return sorted(tf_cue_target)
    if feature_family == "combined_cue_target":
        return sorted(set(erp_cue_target + tf_cue_target))
    if feature_family == "erp_response":
        return sorted(erp_response)
    if feature_family == "tf_response":
        return sorted(tf_response)
    if feature_family == "combined_response":
        return sorted(set(erp_response + tf_response))
    if feature_family == "combined_all":
        return sorted(set(erp_cue_target + tf_cue_target + erp_response + tf_response))

    raise ValueError(f"Unknown feature family: {feature_family}")


def build_task(df: pd.DataFrame, task_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    d = df.copy()
    d["outcome"] = d["outcome"].map(norm_text)
    d["trial_class_norm"] = d["trial_class_norm"].map(norm_text)

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
        raise ValueError(f"Unknown task: {task_name}")

    return d.reset_index(drop=True), y.reset_index(drop=True)


def make_model(C: float = 0.3) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=2026,
        )),
    ])


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def run_group_cv(df: pd.DataFrame, y: pd.Series, feature_cols: List[str], task: str, family: str, n_splits: int = 5) -> pd.DataFrame:
    rows = []
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    groups = df["subject_id_norm"].map(normalize_subject_id).values
    yy = y.astype(int).values

    if len(np.unique(yy)) < 2 or len(np.unique(groups)) < n_splits:
        return pd.DataFrame()

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=2026)
    fold = 0
    for train_idx, test_idx in cv.split(X, yy, groups):
        fold += 1
        if len(np.unique(yy[train_idx])) < 2 or len(np.unique(yy[test_idx])) < 2:
            continue

        model = make_model()
        model.fit(X.iloc[train_idx], yy[train_idx])
        prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)

        rows.append({
            "cv_type": "subject_grouped_cv",
            "task": task,
            "feature_family": family,
            "fold": fold,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_features": int(len(feature_cols)),
            "positive_rate_test": float(np.mean(yy[test_idx])),
            "balanced_accuracy": float(balanced_accuracy_score(yy[test_idx], pred)),
            "roc_auc": safe_auc(yy[test_idx], prob),
        })

    return pd.DataFrame(rows)


def run_subject_cv(df: pd.DataFrame, y: pd.Series, feature_cols: List[str], task: str, family: str, min_class_n: int = 8) -> pd.DataFrame:
    rows = []
    tmp = df.copy()
    tmp["label"] = y.astype(int).values

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
            rows.append({
                "cv_type": "within_subject_cv",
                "task": task,
                "feature_family": family,
                "subject_id": subject,
                "group": g["group"].iloc[0] if "group" in g.columns else "",
                "n_trials": int(len(g)),
                "n_positive": int(counts[1]),
                "n_negative": int(counts[0]),
                "n_features": int(len(feature_cols)),
                "balanced_accuracy": float(np.nanmean(baccs)),
                "roc_auc": float(np.nanmean(aucs)),
            })

    return pd.DataFrame(rows)


def summarize_results(group_cv: pd.DataFrame, subject_cv: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if not group_cv.empty:
        for (task, family), g in group_cv.groupby(["task", "feature_family"]):
            for metric in ["balanced_accuracy", "roc_auc"]:
                vals = pd.to_numeric(g[metric], errors="coerce").dropna().values
                if len(vals) < 2:
                    continue
                t, p = stats.ttest_1samp(vals, 0.5)
                ci = stats.t.interval(0.95, df=len(vals)-1, loc=float(np.mean(vals)), scale=float(stats.sem(vals))) if len(vals) > 1 else (np.nan, np.nan)
                rows.append({
                    "cv_type": "subject_grouped_cv",
                    "task": task,
                    "feature_family": family,
                    "metric": metric,
                    "n_units": int(len(vals)),
                    "mean_score": float(np.mean(vals)),
                    "ci95_low": float(ci[0]),
                    "ci95_high": float(ci[1]),
                    "t_vs_0p5": float(t),
                    "p_value": float(p),
                })

    if not subject_cv.empty:
        for (task, family), g in subject_cv.groupby(["task", "feature_family"]):
            for metric in ["balanced_accuracy", "roc_auc"]:
                vals = pd.to_numeric(g[metric], errors="coerce").dropna().values
                if len(vals) < 8:
                    continue
                t, p = stats.ttest_1samp(vals, 0.5)
                ci = stats.t.interval(0.95, df=len(vals)-1, loc=float(np.mean(vals)), scale=float(stats.sem(vals))) if len(vals) > 1 else (np.nan, np.nan)
                rows.append({
                    "cv_type": "within_subject_cv",
                    "task": task,
                    "feature_family": family,
                    "metric": metric,
                    "n_units": int(len(vals)),
                    "mean_score": float(np.mean(vals)),
                    "ci95_low": float(ci[0]),
                    "ci95_high": float(ci[1]),
                    "t_vs_0p5": float(t),
                    "p_value": float(p),
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_fdr_bh"] = fdr_bh(out["p_value"].values)
        out = out.sort_values(["q_value_fdr_bh", "p_value"]).reset_index(drop=True)
    return out


def write_report(out_dir: Path, summary: pd.DataFrame, group_cv: pd.DataFrame, subject_cv: pd.DataFrame) -> None:
    lines = []
    lines.append("# EEG feature-decoding report")
    lines.append("")
    lines.append("This analysis tests whether trial-wise EEG features predict behavior or discriminate response-monitoring categories.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Subject-grouped CV rows: {len(group_cv)}")
    lines.append(f"- Within-subject CV rows: {len(subject_cv)}")
    lines.append(f"- Summary tests: {len(summary)}")
    lines.append(f"- FDR q < 0.05 summary rows: {int((summary['q_value_fdr_bh'] < 0.05).sum()) if not summary.empty else 0}")
    if not summary.empty:
        lines.append("")
        lines.append("| CV type | Task | Feature family | Metric | n | Mean | 95% CI | p | q |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
        for _, row in summary.head(30).iterrows():
            lines.append(
                f"| {row['cv_type']} | {row['task']} | {row['feature_family']} | {row['metric']} | "
                f"{int(row['n_units'])} | {row['mean_score']:.3f} | "
                f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] | "
                f"{row['p_value']:.3g} | {row['q_value_fdr_bh']:.3g} |"
            )
    lines.append("")
    lines.append("## Interpretation guardrail")
    lines.append("")
    lines.append("- Cue/target decoding tasks are closer to predictive neural evidence because they avoid response-locked post-decision signals.")
    lines.append("- Response-locked FA-vs-Hit decoding is a neural signature of error monitoring, not a pre-decision prediction.")
    lines.append("- Subject-grouped CV asks whether the effect generalizes to held-out participants; within-subject CV asks whether each person's EEG carries within-person predictive information.")
    lines.append("")
    (out_dir / "EEG_DECODING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--erp_features", type=str, required=True)
    parser.add_argument("--tf_features", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--min_class_n_subject_cv", type=int, default=8)
    args = parser.parse_args()

    erp_path = Path(args.erp_features).expanduser().resolve()
    tf_path = Path(args.tf_features).expanduser().resolve() if args.tf_features else None
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(erp_path, tf_path)

    task_specs = [
        ("press_prediction_cue_target", ["erp_cue_target", "tf_cue_target", "combined_cue_target"]),
        ("nogo_failure_prediction_cue_target", ["erp_cue_target", "tf_cue_target", "combined_cue_target"]),
        ("go_omission_prediction_cue_target", ["erp_cue_target", "tf_cue_target", "combined_cue_target"]),
        ("error_monitoring_signature_response", ["erp_response", "tf_response", "combined_response"]),
    ]

    group_rows = []
    subject_rows = []

    for task, families in task_specs:
        task_df, y = build_task(df, task)
        if len(task_df) == 0 or len(np.unique(y)) < 2:
            continue

        for family in families:
            feature_cols = select_features(task_df, family)
            # Drop features that are entirely missing or constant.
            usable = []
            for c in feature_cols:
                x = pd.to_numeric(task_df[c], errors="coerce")
                if x.notna().sum() >= 20 and x.nunique(dropna=True) > 1:
                    usable.append(c)
            feature_cols = usable

            if len(feature_cols) == 0:
                continue

            gcv = run_group_cv(task_df, y, feature_cols, task, family)
            scv = run_subject_cv(task_df, y, feature_cols, task, family, min_class_n=args.min_class_n_subject_cv)

            if not gcv.empty:
                group_rows.append(gcv)
            if not scv.empty:
                subject_rows.append(scv)

            print(f"{task} | {family}: group_cv={len(gcv)}, subject_cv={len(scv)}, features={len(feature_cols)}")

    group_cv = pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame()
    subject_cv = pd.concat(subject_rows, ignore_index=True) if subject_rows else pd.DataFrame()
    summary = summarize_results(group_cv, subject_cv)

    group_cv.to_csv(out_dir / "eeg_decoding_group_cv_results.csv", index=False)
    subject_cv.to_csv(out_dir / "eeg_decoding_subject_cv_results.csv", index=False)
    summary.to_csv(out_dir / "eeg_decoding_summary.csv", index=False)
    write_report(out_dir, summary, group_cv, subject_cv)

    print(f"Wrote outputs to: {out_dir}")
    if not summary.empty:
        print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
