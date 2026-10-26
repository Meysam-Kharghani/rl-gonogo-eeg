#!/usr/bin/env python3
# FORMALIZED HISTORICAL SOURCE SNAPSHOT
# Evidence status: HISTORICAL_OR_SUPERSEDED_DEVELOPMENT_EVIDENCE
# This file preserves a historical project stage with public-facing terminology standardized for release.
# The SHA-256 of the original executed/development source is recorded in the corresponding provenance registry.

"""
Sanitized EEG decoding validation for the EEG-computational study.

Why this script exists
----------------------
Feature-attribution can reveal leakage-like non-neural predictors. In the first
decoding/attribution pass, columns such as `cue_idx` may enter time-frequency
feature families because their names begin with `cue_`, even though they are
task-design/latent columns, not EEG measurements.

This script reruns decoding with stricter EEG-only feature selection and optional
fold-wise nuisance residualization.

Key protections
---------------
1. Strict EEG feature selection:
   - ERP features must have ERP suffixes such as _mean_uv, _min_uv, _max_uv,
     _slope_uv_per_s, _minlat_ms, _maxlat_ms.
   - Time-frequency features must explicitly contain theta/alpha/mu/beta.
   - Columns such as cue_idx, cue_type, trial_class, p_press, action_press,
     is_nogo, correct, error, fa, subject_index are excluded.

2. Subject-grouped cross-validation:
   - tests generalization to held-out participants.

3. Within-subject cross-validation:
   - tests whether each person's EEG carries within-person predictive signal.

4. Optional fold-wise nuisance residualization:
   - residualizes EEG features using nuisance variables fit on train folds only.
   - cue/target prediction tasks use cue_type and/or trial_class as nuisance
     variables where appropriate.

5. Empirical permutation validation:
   - labels are permuted within subject and globally.

Outputs
-------
- sanitized_eeg_decoding_feature_audit.csv
- sanitized_eeg_decoding_observed.csv
- sanitized_eeg_decoding_null.csv
- sanitized_eeg_decoding_summary.csv
- SANITIZED_EEG_DECODING_REPORT.md

Example
-------
python code/final/run_eeg_sanitized_decoding_validation_v1.py \
  --erp_features results/eeg_neural_dynamics_v1_2_corrected/eeg_trial_neural_features_latents_corrected.csv \
  --tf_features results/eeg_time_frequency_analysis_v1/tf_trial_bandpower_features.csv \
  --out results/eeg_sanitized_decoding_validation_v1 \
  --n_perm 200
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TASK_SPECS = [
    {
        "task": "press_prediction_cue_target",
        "families": ["erp_cue_target", "tf_cue_target", "combined_cue_target"],
        "positive_label": "press",
        "nuisance_cols": ["cue_type_norm", "trial_class_norm"],
        "description": "Can cue/target EEG predict response tendency?",
    },
    {
        "task": "nogo_failure_prediction_cue_target",
        "families": ["erp_cue_target", "tf_cue_target", "combined_cue_target"],
        "positive_label": "nogo_false_alarm",
        "nuisance_cols": ["cue_type_norm"],
        "description": "Can cue/target EEG predict false alarm vs correct rejection within NoGo trials?",
    },
    {
        "task": "go_omission_prediction_cue_target",
        "families": ["erp_cue_target", "tf_cue_target", "combined_cue_target"],
        "positive_label": "go_miss",
        "nuisance_cols": ["cue_type_norm"],
        "description": "Can cue/target EEG predict miss vs hit within Go trials?",
    },
    {
        "task": "error_monitoring_signature_response",
        "families": ["erp_response", "tf_response", "combined_response"],
        "positive_label": "nogo_false_alarm",
        "nuisance_cols": [],
        "description": "Can response-locked EEG distinguish false alarm from Go hit?",
    },
]

NON_EEG_EXCLUDE = {
    "subject_index", "group_latent", "cue_idx", "cue_type_latent",
    "trial_index", "trial_index_numeric", "trial_ordinal",
    "response_present", "rt_response_from_target_ms", "log_rt",
    "cue_onset_sec", "target_onset_sec", "response_onset_from_metadata_sec",
    "response_marker_onset_sec", "response_marker_error_ms",
    "condition_cue_match", "condition_trial_class_match",
    "p_press", "control_current", "actor_qdiff_current", "belief_nogo_current",
    "associability_current", "is_nogo", "action_press", "correct", "error", "fa",
    "latent_trial_index_offset_used",
}

ERP_SUFFIXES = (
    "_mean_uv", "_min_uv", "_max_uv",
    "_minlat_ms", "_maxlat_ms", "_slope_uv_per_s",
)

TF_BAND_TOKENS = ("theta", "alpha", "mu", "beta")


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
    erp["cue_type_norm"] = erp["cue_type"].map(norm_text) if "cue_type" in erp.columns else ""
    erp["response_type_norm"] = erp["response_type"].map(norm_text)
    erp["log_rt"] = np.log(pd.to_numeric(erp["rt_response_from_target_ms"], errors="coerce").where(lambda x: x > 0))

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

        # Keep only strict numeric TF EEG features. This intentionally excludes
        # cue_idx and any latent/design columns.
        tf_cols = []
        for c in tf.columns:
            if c in key_cols or c in NON_EEG_EXCLUDE:
                continue
            if not pd.api.types.is_numeric_dtype(tf[c]):
                continue
            cl = c.lower()
            if (c.startswith(("cue_", "target_", "resp_")) and any(tok in cl for tok in TF_BAND_TOKENS)):
                tf_cols.append(c)

        erp = erp.merge(tf[key_cols + sorted(tf_cols)], on=key_cols, how="left", suffixes=("", "_tf"))

    return erp


def is_erp_feature(c: str, allowed_prefixes: Tuple[str, ...]) -> bool:
    return c.startswith(allowed_prefixes) and c.endswith(ERP_SUFFIXES) and c not in NON_EEG_EXCLUDE


def is_tf_feature(c: str, allowed_prefixes: Tuple[str, ...]) -> bool:
    cl = c.lower()
    return c.startswith(allowed_prefixes) and any(tok in cl for tok in TF_BAND_TOKENS) and c not in NON_EEG_EXCLUDE


def select_features(df: pd.DataFrame, family: str) -> List[str]:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    erp_cue_target = [c for c in numeric if is_erp_feature(c, ("cue_", "target_"))]
    erp_response = [c for c in numeric if is_erp_feature(c, ("resp_",))]
    tf_cue_target = [c for c in numeric if is_tf_feature(c, ("cue_", "target_"))]
    tf_response = [c for c in numeric if is_tf_feature(c, ("resp_",))]

    if family == "erp_cue_target":
        cols = erp_cue_target
    elif family == "tf_cue_target":
        cols = tf_cue_target
    elif family == "combined_cue_target":
        cols = sorted(set(erp_cue_target + tf_cue_target))
    elif family == "erp_response":
        cols = erp_response
    elif family == "tf_response":
        cols = tf_response
    elif family == "combined_response":
        cols = sorted(set(erp_response + tf_response))
    else:
        raise ValueError(f"Unknown family: {family}")

    usable = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() >= 20 and x.nunique(dropna=True) > 1:
            usable.append(c)

    return sorted(usable)


def build_task(df: pd.DataFrame, task: str) -> Tuple[pd.DataFrame, pd.Series]:
    d = df.copy()
    d["outcome"] = d["outcome"].map(norm_text)

    if task == "press_prediction_cue_target":
        d = d[d["response_present"].notna()].copy()
        y = pd.to_numeric(d["response_present"], errors="coerce").fillna(0).astype(int)

    elif task == "nogo_failure_prediction_cue_target":
        d = d[d["outcome"].isin(["nogo_cr", "nogo_fa"])].copy()
        y = (d["outcome"] == "nogo_fa").astype(int)

    elif task == "go_omission_prediction_cue_target":
        d = d[d["outcome"].isin(["go_hit", "go_miss"])].copy()
        y = (d["outcome"] == "go_miss").astype(int)

    elif task == "error_monitoring_signature_response":
        d = d[d["outcome"].isin(["go_hit", "nogo_fa"])].copy()
        y = (d["outcome"] == "nogo_fa").astype(int)

    else:
        raise ValueError(task)

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


def make_design_matrix(meta: pd.DataFrame, nuisance_cols: List[str], reference_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    if not nuisance_cols:
        Z = pd.DataFrame(index=meta.index)
    else:
        parts = []
        for col in nuisance_cols:
            if col not in meta.columns:
                continue
            s = meta[col]
            if pd.api.types.is_numeric_dtype(s):
                parts.append(pd.DataFrame({col: pd.to_numeric(s, errors="coerce")}, index=meta.index))
            else:
                parts.append(pd.get_dummies(s.astype(str), prefix=col, drop_first=False, dtype=float))
        Z = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=meta.index)

    Z = Z.replace([np.inf, -np.inf], np.nan)
    if reference_columns is not None:
        Z = Z.reindex(columns=reference_columns, fill_value=0.0)
    else:
        reference_columns = list(Z.columns)

    for c in Z.columns:
        if Z[c].isna().any():
            Z[c] = Z[c].fillna(Z[c].median())

    return Z.astype(float), reference_columns


def residualize_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
    nuisance_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not nuisance_cols:
        return X_train, X_test

    Xtr = X_train.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    Xte = X_test.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    med = Xtr.median(axis=0)
    Xtr = Xtr.fillna(med)
    Xte = Xte.fillna(med)

    Ztr, cols = make_design_matrix(meta_train, nuisance_cols)
    Zte, _ = make_design_matrix(meta_test, nuisance_cols, reference_columns=cols)

    # Add intercept.
    Ztr_mat = np.column_stack([np.ones(len(Ztr)), Ztr.values])
    Zte_mat = np.column_stack([np.ones(len(Zte)), Zte.values])

    Xtr_mat = Xtr.values.astype(float)
    beta, *_ = np.linalg.lstsq(Ztr_mat, Xtr_mat, rcond=None)

    Xtr_res = Xtr_mat - Ztr_mat @ beta
    Xte_res = Xte.values.astype(float) - Zte_mat @ beta

    return (
        pd.DataFrame(Xtr_res, columns=X_train.columns, index=X_train.index),
        pd.DataFrame(Xte_res, columns=X_test.columns, index=X_test.index),
    )


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, score))
    except Exception:
        return np.nan


def fit_score_fold(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, float]:
    model = make_model()
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return float(balanced_accuracy_score(y_test, pred)), safe_auc(y_test, prob)


def grouped_cv_score(
    d: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
    variant: str,
    nuisance_cols: List[str],
    random_state: int,
) -> Tuple[float, float, int]:
    X = d[feature_cols].replace([np.inf, -np.inf], np.nan)
    yy = y.astype(int).values
    groups = d["subject_id_norm"].values

    if len(np.unique(yy)) < 2 or len(np.unique(groups)) < 5:
        return np.nan, np.nan, 0

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    baccs, aucs = [], []

    for train_idx, test_idx in cv.split(X, yy, groups):
        if len(np.unique(yy[train_idx])) < 2 or len(np.unique(yy[test_idx])) < 2:
            continue

        Xtr = X.iloc[train_idx].copy()
        Xte = X.iloc[test_idx].copy()

        if variant == "nuisance_residualized":
            Xtr, Xte = residualize_train_test(
                X_train=Xtr,
                X_test=Xte,
                meta_train=d.iloc[train_idx],
                meta_test=d.iloc[test_idx],
                nuisance_cols=nuisance_cols,
            )

        bacc, auc = fit_score_fold(Xtr, Xte, yy[train_idx], yy[test_idx])
        baccs.append(bacc)
        aucs.append(auc)

    return float(np.nanmean(baccs)), float(np.nanmean(aucs)), int(len(baccs))


def within_subject_score(
    d: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
    variant: str,
    nuisance_cols: List[str],
    random_state: int,
    min_class_n: int,
) -> Tuple[float, float, int]:
    tmp = d.copy()
    tmp["label"] = y.astype(int).values

    subj_baccs, subj_aucs = [], []

    for subject, g in tmp.groupby("subject_id_norm"):
        labels = g["label"].astype(int).values
        counts = np.bincount(labels, minlength=2)
        if counts.min() < min_class_n:
            continue

        n_splits = int(min(5, counts.min()))
        if n_splits < 2:
            continue

        X = g[feature_cols].replace([np.inf, -np.inf], np.nan)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        baccs, aucs = [], []
        for train_idx, test_idx in cv.split(X, labels):
            if len(np.unique(labels[train_idx])) < 2 or len(np.unique(labels[test_idx])) < 2:
                continue

            Xtr = X.iloc[train_idx].copy()
            Xte = X.iloc[test_idx].copy()

            if variant == "nuisance_residualized":
                Xtr, Xte = residualize_train_test(
                    X_train=Xtr,
                    X_test=Xte,
                    meta_train=g.iloc[train_idx],
                    meta_test=g.iloc[test_idx],
                    nuisance_cols=nuisance_cols,
                )

            bacc, auc = fit_score_fold(Xtr, Xte, labels[train_idx], labels[test_idx])
            baccs.append(bacc)
            aucs.append(auc)

        if baccs:
            subj_baccs.append(float(np.nanmean(baccs)))
            subj_aucs.append(float(np.nanmean(aucs)))

    return float(np.nanmean(subj_baccs)), float(np.nanmean(subj_aucs)), int(len(subj_baccs))


def permute_labels(y: pd.Series, subjects: pd.Series, rng: np.random.Generator, mode: str) -> pd.Series:
    vals = y.astype(int).values.copy()
    if mode == "global":
        rng.shuffle(vals)
        return pd.Series(vals)

    if mode == "within_subject":
        subj = subjects.values
        for s in np.unique(subj):
            idx = np.where(subj == s)[0]
            v = vals[idx].copy()
            rng.shuffle(v)
            vals[idx] = v
        return pd.Series(vals)

    raise ValueError(mode)


def empirical_p(observed: float, null_vals: np.ndarray) -> float:
    null_vals = np.asarray(null_vals, dtype=float)
    null_vals = null_vals[np.isfinite(null_vals)]
    if not np.isfinite(observed) or len(null_vals) == 0:
        return np.nan
    return float((np.sum(null_vals >= observed) + 1) / (len(null_vals) + 1))


def feature_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    families = [
        "erp_cue_target", "tf_cue_target", "combined_cue_target",
        "erp_response", "tf_response", "combined_response",
    ]
    forbidden = {"cue_idx", "subject_index", "action_press", "is_nogo", "correct", "error", "fa", "p_press"}
    for fam in families:
        cols = select_features(df, fam)
        for c in cols:
            rows.append({
                "feature_family": fam,
                "feature": c,
                "contains_forbidden_token": any(tok == c or tok in c for tok in forbidden),
                "is_tf_like": any(tok in c.lower() for tok in TF_BAND_TOKENS),
                "is_erp_like": c.endswith(ERP_SUFFIXES),
            })
    return pd.DataFrame(rows)


def run_all(
    df: pd.DataFrame,
    n_perm: int,
    seed: int,
    min_class_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    observed_rows = []
    null_rows = []
    audit = feature_audit(df)

    variants = ["raw_eeg_only", "nuisance_residualized"]

    for spec in TASK_SPECS:
        task = spec["task"]
        d, y = build_task(df, task)
        if len(d) == 0 or len(np.unique(y)) < 2:
            continue

        for family in spec["families"]:
            features = select_features(d, family)
            if not features:
                continue

            for variant in variants:
                nuisance_cols = spec["nuisance_cols"] if variant == "nuisance_residualized" else []

                gb, ga, gf = grouped_cv_score(d, y, features, variant, nuisance_cols, seed)
                sb, sa, ns = within_subject_score(d, y, features, variant, nuisance_cols, seed, min_class_n)

                observed_rows.extend([
                    {"cv_type": "subject_grouped_cv", "task": task, "feature_family": family, "variant": variant, "metric": "balanced_accuracy", "observed_score": gb, "n_features": len(features), "n_units": gf},
                    {"cv_type": "subject_grouped_cv", "task": task, "feature_family": family, "variant": variant, "metric": "roc_auc", "observed_score": ga, "n_features": len(features), "n_units": gf},
                    {"cv_type": "within_subject_cv", "task": task, "feature_family": family, "variant": variant, "metric": "balanced_accuracy", "observed_score": sb, "n_features": len(features), "n_units": ns},
                    {"cv_type": "within_subject_cv", "task": task, "feature_family": family, "variant": variant, "metric": "roc_auc", "observed_score": sa, "n_features": len(features), "n_units": ns},
                ])

                for perm_idx in range(1, n_perm + 1):
                    for mode in ["within_subject", "global"]:
                        yp = permute_labels(y, d["subject_id_norm"], rng, mode)

                        pgb, pga, _ = grouped_cv_score(d, yp, features, variant, nuisance_cols, seed + perm_idx)
                        psb, psa, _ = within_subject_score(d, yp, features, variant, nuisance_cols, seed + perm_idx, min_class_n)

                        null_rows.extend([
                            {"permutation_mode": mode, "perm_idx": perm_idx, "cv_type": "subject_grouped_cv", "task": task, "feature_family": family, "variant": variant, "metric": "balanced_accuracy", "null_score": pgb},
                            {"permutation_mode": mode, "perm_idx": perm_idx, "cv_type": "subject_grouped_cv", "task": task, "feature_family": family, "variant": variant, "metric": "roc_auc", "null_score": pga},
                            {"permutation_mode": mode, "perm_idx": perm_idx, "cv_type": "within_subject_cv", "task": task, "feature_family": family, "variant": variant, "metric": "balanced_accuracy", "null_score": psb},
                            {"permutation_mode": mode, "perm_idx": perm_idx, "cv_type": "within_subject_cv", "task": task, "feature_family": family, "variant": variant, "metric": "roc_auc", "null_score": psa},
                        ])

                print(f"Finished {task} | {family} | {variant} | features={len(features)}")

    observed = pd.DataFrame(observed_rows)
    null = pd.DataFrame(null_rows)

    summary_rows = []
    for _, obs in observed.iterrows():
        sub = null[
            (null["cv_type"] == obs["cv_type"]) &
            (null["task"] == obs["task"]) &
            (null["feature_family"] == obs["feature_family"]) &
            (null["variant"] == obs["variant"]) &
            (null["metric"] == obs["metric"])
        ]
        for mode, g in sub.groupby("permutation_mode"):
            nv = pd.to_numeric(g["null_score"], errors="coerce").dropna().values
            summary_rows.append({
                "permutation_mode": mode,
                "cv_type": obs["cv_type"],
                "task": obs["task"],
                "feature_family": obs["feature_family"],
                "variant": obs["variant"],
                "metric": obs["metric"],
                "observed_score": float(obs["observed_score"]),
                "null_mean": float(np.mean(nv)) if len(nv) else np.nan,
                "null_sd": float(np.std(nv, ddof=1)) if len(nv) > 1 else np.nan,
                "empirical_p_one_sided": empirical_p(float(obs["observed_score"]), nv),
                "n_permutations": int(len(nv)),
                "n_features": int(obs["n_features"]),
                "n_units": int(obs["n_units"]),
            })

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["q_value_fdr_bh"] = fdr_bh(summary["empirical_p_one_sided"].values)
        summary = summary.sort_values(["q_value_fdr_bh", "empirical_p_one_sided"]).reset_index(drop=True)

    return audit, observed, null, summary


def write_report(out_dir: Path, audit: pd.DataFrame, summary: pd.DataFrame, n_perm: int) -> None:
    lines = []
    lines.append("# Sanitized EEG decoding validation report")
    lines.append("")
    lines.append("This report reruns EEG decoding after strict removal of non-EEG design/latent columns such as cue_idx.")
    lines.append("")
    lines.append("## Feature audit")
    lines.append("")
    lines.append(f"- Selected EEG feature rows: {len(audit)}")
    if not audit.empty:
        lines.append(f"- Rows flagged as containing forbidden tokens: {int(audit['contains_forbidden_token'].sum())}")
        lines.append("")
        lines.append("| Feature family | n features | flagged forbidden |")
        lines.append("|---|---:|---:|")
        for fam, g in audit.groupby("feature_family"):
            lines.append(f"| {fam} | {len(g)} | {int(g['contains_forbidden_token'].sum())} |")
    lines.append("")
    lines.append("## Decoding summary")
    lines.append("")
    lines.append(f"- Permutations per model: {n_perm}")
    lines.append(f"- Summary rows: {len(summary)}")
    lines.append(f"- Empirical FDR q < 0.05 rows: {int((summary['q_value_fdr_bh'] < 0.05).sum()) if not summary.empty else 0}")
    if not summary.empty:
        lines.append("")
        lines.append("| Perm mode | CV type | Task | Family | Variant | Metric | observed | null mean | p | q |")
        lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|")
        for _, row in summary.head(60).iterrows():
            lines.append(
                f"| {row['permutation_mode']} | {row['cv_type']} | {row['task']} | {row['feature_family']} | "
                f"{row['variant']} | {row['metric']} | {row['observed_score']:.3f} | {row['null_mean']:.3f} | "
                f"{row['empirical_p_one_sided']:.4f} | {row['q_value_fdr_bh']:.4f} |"
            )
    lines.append("")
    lines.append("## Interpretation guardrail")
    lines.append("")
    lines.append("- If press-prediction drops after removing cue_idx and residualizing cue/trial-class, the earlier effect should be treated as task-structure-assisted rather than pure EEG prediction.")
    lines.append("- NoGo FA vs CR and Go Miss vs Hit are more behaviorally specific because they are tested within trial class.")
    lines.append("- Response-locked FA vs Hit remains a post-response error-monitoring signature, not a pre-decision prediction.")
    lines.append("")
    (out_dir / "SANITIZED_EEG_DECODING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


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
    audit, observed, null, summary = run_all(
        df=df,
        n_perm=int(args.n_perm),
        seed=int(args.seed),
        min_class_n=int(args.min_class_n_subject_cv),
    )

    audit.to_csv(out_dir / "sanitized_eeg_decoding_feature_audit.csv", index=False)
    observed.to_csv(out_dir / "sanitized_eeg_decoding_observed.csv", index=False)
    null.to_csv(out_dir / "sanitized_eeg_decoding_null.csv", index=False)
    summary.to_csv(out_dir / "sanitized_eeg_decoding_summary.csv", index=False)

    write_report(out_dir, audit, summary, int(args.n_perm))

    print(f"Wrote outputs to: {out_dir}")
    if not summary.empty:
        print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
