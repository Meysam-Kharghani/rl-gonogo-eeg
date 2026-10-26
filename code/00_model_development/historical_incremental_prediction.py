#!/usr/bin/env python3
# FORMALIZED HISTORICAL SOURCE SNAPSHOT
# Evidence status: HISTORICAL_OR_SUPERSEDED_DEVELOPMENT_EVIDENCE
# This file preserves a historical project stage with public-facing terminology standardized for release.
# The SHA-256 of the original executed/development source is recorded in the corresponding provenance registry.

"""
Historical incremental EEG-versus-computational prediction analysis for the EEG-computational study.

Purpose
-------
This script asks whether trial-wise EEG features add predictive information
beyond task design and model-derived RL/DDM latent states, and whether RL/DDM
latent states add information beyond task design.

It is retained as developmental evidence and does not replace the final cross-fitted computational
model comparison. Because supplied latent variables may be fit-derived rather
than generated in a strictly fold-wise out-of-sample pipeline, results involving
RL latent states should be described as computational-state informativeness or
association evidence unless fold-wise latents are provided.

Main tasks
----------
1. press_prediction_predecision:
   all trials; predict response_present from cue/target information only.
2. nogo_failure_predecision:
   NoGo trials; predict false alarm vs correct rejection from cue/target
   information only.
3. go_omission_predecision:
   Go trials; predict miss vs hit from cue/target information only.

Predictor blocks
----------------
- design: cue_type / trial_class / group as appropriate
- rl: p_press, control_current, actor_qdiff_current, belief_nogo_current,
      associability_current when present
- eeg_erp: strict cue/target ERP features only
- eeg_tf: strict cue/target theta/alpha/mu/beta features only
- eeg_combined: eeg_erp + eeg_tf
- design_plus_rl
- design_plus_eeg
- design_plus_rl_plus_eeg

Validation
----------
- subject-grouped CV: held-out participants
- within-subject CV: within-person trial-level decoding

Outputs
-------
- eeg_incremental_model_scores.csv
- eeg_incremental_model_summary.csv
- eeg_incremental_pairwise_comparisons.csv
- EEG_INCREMENTAL_PREDICTION_REPORT.md

Example
-------
python code/final/historical_incremental_prediction.py \
  --erp_features results/eeg_neural_dynamics_v1_2_corrected/eeg_trial_neural_features_latents_corrected.csv \
  --tf_features results/eeg_time_frequency_analysis_v1/tf_trial_bandpower_features.csv \
  --out results/model_development/historical_incremental_prediction
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


ERP_SUFFIXES = (
    "_mean_uv", "_min_uv", "_max_uv",
    "_minlat_ms", "_maxlat_ms", "_slope_uv_per_s",
)
TF_TOKENS = ("theta", "alpha", "mu", "beta")

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

RL_LATENT_COLS = [
    "p_press",
    "control_current",
    "actor_qdiff_current",
    "belief_nogo_current",
    "associability_current",
]


TASK_SPECS = [
    {
        "task": "press_prediction_predecision",
        "description": "all trials: response vs no response using cue/target information",
        "design_cols": ["cue_type_norm", "trial_class_norm", "group"],
    },
    {
        "task": "nogo_failure_predecision",
        "description": "NoGo trials: false alarm vs correct rejection using cue/target information",
        "design_cols": ["cue_type_norm", "group"],
    },
    {
        "task": "go_omission_predecision",
        "description": "Go trials: miss vs hit using cue/target information",
        "design_cols": ["cue_type_norm", "group"],
    },
]


MODEL_BLOCKS = {
    "design": ["design"],
    "rl": ["rl"],
    "eeg_erp": ["eeg_erp"],
    "eeg_tf": ["eeg_tf"],
    "eeg_combined": ["eeg_erp", "eeg_tf"],
    "design_plus_rl": ["design", "rl"],
    "design_plus_eeg": ["design", "eeg_erp", "eeg_tf"],
    "design_plus_rl_plus_eeg": ["design", "rl", "eeg_erp", "eeg_tf"],
}


PAIRWISE_COMPARISONS = [
    {
        "comparison": "rl_increment_over_design",
        "larger_model": "design_plus_rl",
        "smaller_model": "design",
        "interpretation": "model-derived computational states beyond task design",
    },
    {
        "comparison": "eeg_increment_over_design",
        "larger_model": "design_plus_eeg",
        "smaller_model": "design",
        "interpretation": "EEG cue/target information beyond task design",
    },
    {
        "comparison": "eeg_increment_over_design_plus_rl",
        "larger_model": "design_plus_rl_plus_eeg",
        "smaller_model": "design_plus_rl",
        "interpretation": "EEG cue/target information beyond task design and computational states",
    },
    {
        "comparison": "rl_increment_over_design_plus_eeg",
        "larger_model": "design_plus_rl_plus_eeg",
        "smaller_model": "design_plus_eeg",
        "interpretation": "computational states beyond task design and EEG",
    },
    {
        "comparison": "tf_over_erp",
        "larger_model": "eeg_tf",
        "smaller_model": "eeg_erp",
        "interpretation": "time-frequency EEG block relative to ERP block",
    },
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
    erp["cue_type_norm"] = erp["cue_type"].map(norm_text) if "cue_type" in erp.columns else ""
    erp["response_type_norm"] = erp["response_type"].map(norm_text)
    erp["group"] = erp["group"].map(norm_text) if "group" in erp.columns else ""

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
        tf_cols = []
        for c in tf.columns:
            if c in key_cols or c in NON_EEG_EXCLUDE:
                continue
            if not pd.api.types.is_numeric_dtype(tf[c]):
                continue
            cl = c.lower()
            if c.startswith(("cue_", "target_")) and any(tok in cl for tok in TF_TOKENS):
                tf_cols.append(c)

        erp = erp.merge(tf[key_cols + sorted(tf_cols)], on=key_cols, how="left", suffixes=("", "_tf"))

    return erp


def is_erp_feature(c: str) -> bool:
    return c.startswith(("cue_", "target_")) and c.endswith(ERP_SUFFIXES) and c not in NON_EEG_EXCLUDE


def is_tf_feature(c: str) -> bool:
    cl = c.lower()
    return c.startswith(("cue_", "target_")) and any(tok in cl for tok in TF_TOKENS) and c not in NON_EEG_EXCLUDE


def get_feature_blocks(df: pd.DataFrame, design_cols: List[str]) -> Dict[str, List[str]]:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    eeg_erp = []
    eeg_tf = []
    rl = []

    for c in numeric:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() < 20 or x.nunique(dropna=True) <= 1:
            continue
        if is_erp_feature(c):
            eeg_erp.append(c)
        elif is_tf_feature(c):
            eeg_tf.append(c)
        elif c in RL_LATENT_COLS:
            rl.append(c)

    design = [c for c in design_cols if c in df.columns]

    return {
        "design": sorted(design),
        "rl": sorted(rl),
        "eeg_erp": sorted(eeg_erp),
        "eeg_tf": sorted(eeg_tf),
    }


def build_task(df: pd.DataFrame, task: str) -> Tuple[pd.DataFrame, pd.Series]:
    d = df.copy()
    d["outcome"] = d["outcome"].map(norm_text)

    if task == "press_prediction_predecision":
        d = d[d["response_present"].notna()].copy()
        y = pd.to_numeric(d["response_present"], errors="coerce").fillna(0).astype(int)

    elif task == "nogo_failure_predecision":
        d = d[d["outcome"].isin(["nogo_cr", "nogo_fa"])].copy()
        y = (d["outcome"] == "nogo_fa").astype(int)

    elif task == "go_omission_predecision":
        d = d[d["outcome"].isin(["go_hit", "go_miss"])].copy()
        y = (d["outcome"] == "go_miss").astype(int)

    else:
        raise ValueError(f"Unknown task: {task}")

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


def make_train_test_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_blocks: Dict[str, List[str]],
    block_names: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    parts_train = []
    parts_test = []
    out_cols = []

    for block in block_names:
        cols = feature_blocks.get(block, [])
        if not cols:
            continue

        if block == "design":
            tr = pd.get_dummies(train_df[cols].astype(str), prefix=cols, drop_first=False, dtype=float)
            te = pd.get_dummies(test_df[cols].astype(str), prefix=cols, drop_first=False, dtype=float)
            te = te.reindex(columns=tr.columns, fill_value=0.0)
        else:
            tr = train_df[cols].apply(pd.to_numeric, errors="coerce")
            te = test_df[cols].apply(pd.to_numeric, errors="coerce")

        tr = tr.replace([np.inf, -np.inf], np.nan)
        te = te.replace([np.inf, -np.inf], np.nan)

        parts_train.append(tr)
        parts_test.append(te)
        out_cols.extend(list(tr.columns))

    if not parts_train:
        return pd.DataFrame(index=train_df.index), pd.DataFrame(index=test_df.index), []

    Xtr = pd.concat(parts_train, axis=1)
    Xte = pd.concat(parts_test, axis=1)

    # Remove all-missing/constant columns based on train.
    keep = []
    for c in Xtr.columns:
        x = pd.to_numeric(Xtr[c], errors="coerce")
        if x.notna().sum() >= 5 and x.nunique(dropna=True) > 1:
            keep.append(c)

    Xtr = Xtr[keep]
    Xte = Xte.reindex(columns=keep)

    return Xtr, Xte, keep


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, score))
    except Exception:
        return np.nan


def fit_score(X_train, X_test, y_train, y_test) -> Tuple[float, float]:
    if X_train.shape[1] == 0:
        return np.nan, np.nan
    model = make_model()
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return float(balanced_accuracy_score(y_test, pred)), safe_auc(y_test, prob)


def grouped_cv_scores(task_df: pd.DataFrame, y: pd.Series, blocks: Dict[str, List[str]], task: str, model_name: str, block_names: List[str], seed: int) -> List[dict]:
    rows = []
    yy = y.astype(int).values
    groups = task_df["subject_id_norm"].values

    if len(np.unique(yy)) < 2 or len(np.unique(groups)) < 5:
        return rows

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    fold = 0
    for train_idx, test_idx in cv.split(task_df, yy, groups):
        fold += 1
        if len(np.unique(yy[train_idx])) < 2 or len(np.unique(yy[test_idx])) < 2:
            continue

        train_df = task_df.iloc[train_idx].copy()
        test_df = task_df.iloc[test_idx].copy()
        Xtr, Xte, cols = make_train_test_matrix(train_df, test_df, blocks, block_names)

        bacc, auc = fit_score(Xtr, Xte, yy[train_idx], yy[test_idx])

        rows.append({
            "cv_type": "subject_grouped_cv",
            "task": task,
            "model": model_name,
            "unit_id": f"fold_{fold}",
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_features": int(len(cols)),
            "positive_rate_test": float(np.mean(yy[test_idx])),
            "balanced_accuracy": bacc,
            "roc_auc": auc,
        })

    return rows


def within_subject_scores(task_df: pd.DataFrame, y: pd.Series, blocks: Dict[str, List[str]], task: str, model_name: str, block_names: List[str], seed: int, min_class_n: int) -> List[dict]:
    rows = []
    tmp = task_df.copy()
    tmp["label"] = y.astype(int).values

    for subject, g in tmp.groupby("subject_id_norm"):
        labels = g["label"].astype(int).values
        counts = np.bincount(labels, minlength=2)
        if counts.min() < min_class_n:
            continue

        n_splits = int(min(5, counts.min()))
        if n_splits < 2:
            continue

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        baccs, aucs, n_features_seen = [], [], []

        for train_idx, test_idx in cv.split(g, labels):
            if len(np.unique(labels[train_idx])) < 2 or len(np.unique(labels[test_idx])) < 2:
                continue

            train_df = g.iloc[train_idx].copy()
            test_df = g.iloc[test_idx].copy()
            Xtr, Xte, cols = make_train_test_matrix(train_df, test_df, blocks, block_names)

            bacc, auc = fit_score(Xtr, Xte, labels[train_idx], labels[test_idx])
            if np.isfinite(bacc):
                baccs.append(bacc)
            if np.isfinite(auc):
                aucs.append(auc)
            n_features_seen.append(len(cols))

        if baccs:
            rows.append({
                "cv_type": "within_subject_cv",
                "task": task,
                "model": model_name,
                "unit_id": subject,
                "n_train": np.nan,
                "n_test": int(len(g)),
                "n_features": int(np.nanmedian(n_features_seen)) if n_features_seen else np.nan,
                "positive_rate_test": float(np.mean(labels)),
                "balanced_accuracy": float(np.nanmean(baccs)),
                "roc_auc": float(np.nanmean(aucs)) if aucs else np.nan,
            })

    return rows


def run_models(df: pd.DataFrame, seed: int, min_class_n: int) -> pd.DataFrame:
    rows = []

    for spec in TASK_SPECS:
        task = spec["task"]
        task_df, y = build_task(df, task)
        if len(task_df) == 0 or len(np.unique(y)) < 2:
            continue

        blocks = get_feature_blocks(task_df, spec["design_cols"])

        for model_name, block_names in MODEL_BLOCKS.items():
            # Skip impossible models.
            if any(len(blocks.get(block, [])) == 0 for block in block_names if block != "design"):
                continue
            if "design" in block_names and len(blocks.get("design", [])) == 0:
                continue

            rows.extend(grouped_cv_scores(task_df, y, blocks, task, model_name, block_names, seed))
            rows.extend(within_subject_scores(task_df, y, blocks, task, model_name, block_names, seed, min_class_n))

            print(f"{task} | {model_name} | blocks={block_names}")

    return pd.DataFrame(rows)


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if scores.empty:
        return pd.DataFrame()

    for keys, g in scores.groupby(["cv_type", "task", "model"]):
        for metric in ["balanced_accuracy", "roc_auc"]:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna().values
            if len(vals) < 2:
                continue
            t, p = stats.ttest_1samp(vals, 0.5)
            ci = stats.t.interval(0.95, df=len(vals)-1, loc=float(np.mean(vals)), scale=float(stats.sem(vals))) if len(vals) > 1 else (np.nan, np.nan)
            rows.append({
                "cv_type": keys[0],
                "task": keys[1],
                "model": keys[2],
                "metric": metric,
                "n_units": int(len(vals)),
                "mean_score": float(np.mean(vals)),
                "ci95_low": float(ci[0]),
                "ci95_high": float(ci[1]),
                "t_vs_0p5": float(t),
                "p_vs_0p5": float(p),
                "mean_n_features": float(pd.to_numeric(g["n_features"], errors="coerce").mean()),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_fdr_bh_vs_0p5"] = fdr_bh(out["p_vs_0p5"].values)
        out = out.sort_values(["task", "cv_type", "metric", "mean_score"], ascending=[True, True, True, False]).reset_index(drop=True)
    return out


def pairwise_comparisons(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if scores.empty:
        return pd.DataFrame()

    for comp in PAIRWISE_COMPARISONS:
        big = comp["larger_model"]
        small = comp["smaller_model"]

        for keys, g in scores.groupby(["cv_type", "task"]):
            cv_type, task = keys
            for metric in ["balanced_accuracy", "roc_auc"]:
                wide = g.pivot_table(index="unit_id", columns="model", values=metric, aggfunc="mean")
                if big not in wide.columns or small not in wide.columns:
                    continue
                diff = (wide[big] - wide[small]).dropna().astype(float).values
                if len(diff) < 2:
                    continue
                t, p = stats.ttest_1samp(diff, 0.0)
                ci = stats.t.interval(0.95, df=len(diff)-1, loc=float(np.mean(diff)), scale=float(stats.sem(diff))) if len(diff) > 1 else (np.nan, np.nan)
                rows.append({
                    "cv_type": cv_type,
                    "task": task,
                    "comparison": comp["comparison"],
                    "interpretation": comp["interpretation"],
                    "larger_model": big,
                    "smaller_model": small,
                    "metric": metric,
                    "n_units": int(len(diff)),
                    "mean_delta": float(np.mean(diff)),
                    "ci95_low": float(ci[0]),
                    "ci95_high": float(ci[1]),
                    "t_value": float(t),
                    "p_value": float(p),
                    "positive_units": int(np.sum(diff > 0)),
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value_fdr_bh"] = fdr_bh(out["p_value"].values)
        out = out.sort_values(["q_value_fdr_bh", "p_value"]).reset_index(drop=True)
    return out


def write_report(out_dir: Path, summary: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    lines = []
    lines.append("# EEG incremental predictive association analysis report")
    lines.append("")
    lines.append("This report tests whether cue/target EEG adds predictive information beyond task design and model-derived computational states.")
    lines.append("")
    lines.append("## Model performance")
    lines.append("")
    lines.append(f"- Summary rows: {len(summary)}")
    if not summary.empty:
        lines.append("")
        lines.append("| CV type | Task | Model | Metric | n | Mean | 95% CI | q vs .5 |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|")
        for _, row in summary.sort_values(["task", "cv_type", "metric", "mean_score"], ascending=[True, True, True, False]).head(60).iterrows():
            lines.append(
                f"| {row['cv_type']} | {row['task']} | {row['model']} | {row['metric']} | "
                f"{int(row['n_units'])} | {row['mean_score']:.3f} | [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] | "
                f"{row['q_value_fdr_bh_vs_0p5']:.3g} |"
            )

    lines.append("")
    lines.append("## Incremental comparisons")
    lines.append("")
    lines.append(f"- Pairwise rows: {len(pairwise)}")
    lines.append(f"- FDR q < 0.05 rows: {int((pairwise['q_value_fdr_bh'] < 0.05).sum()) if not pairwise.empty else 0}")
    if not pairwise.empty:
        lines.append("")
        lines.append("| CV type | Task | Comparison | Metric | n | Mean Δ | 95% CI | positive units | p | q |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
        for _, row in pairwise.head(80).iterrows():
            lines.append(
                f"| {row['cv_type']} | {row['task']} | {row['comparison']} | {row['metric']} | "
                f"{int(row['n_units'])} | {row['mean_delta']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
                f"{int(row['positive_units'])}/{int(row['n_units'])} | {row['p_value']:.3g} | {row['q_value_fdr_bh']:.3g} |"
            )

    lines.append("")
    lines.append("## Interpretation guardrail")
    lines.append("")
    lines.append("- Use this analysis as association evidence, not as the primary computational result.")
    lines.append("- If RL latent variables were generated from full-data fits, do not call RL-vs-EEG comparisons strictly leakage-free prediction.")
    lines.append("- The most defensible EEG claim is an incremental EEG effect that survives within-subject CV and subject-grouped CV beyond design variables.")
    lines.append("- For manuscript wording, distinguish predecision cue/target EEG prediction from response-locked error-monitoring signatures.")
    lines.append("")
    (out_dir / "EEG_INCREMENTAL_PREDICTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--erp_features", type=str, required=True)
    parser.add_argument("--tf_features", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min_class_n_subject_cv", type=int, default=8)
    args = parser.parse_args()

    erp_path = Path(args.erp_features).expanduser().resolve()
    tf_path = Path(args.tf_features).expanduser().resolve() if args.tf_features else None
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(erp_path, tf_path)

    scores = run_models(df, seed=int(args.seed), min_class_n=int(args.min_class_n_subject_cv))
    summary = summarize_scores(scores)
    pairwise = pairwise_comparisons(scores)

    scores.to_csv(out_dir / "eeg_incremental_model_scores.csv", index=False)
    summary.to_csv(out_dir / "eeg_incremental_model_summary.csv", index=False)
    pairwise.to_csv(out_dir / "eeg_incremental_pairwise_comparisons.csv", index=False)

    write_report(out_dir, summary, pairwise)

    print(f"Wrote outputs to: {out_dir}")
    if not pairwise.empty:
        print(pairwise.head(30).to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
