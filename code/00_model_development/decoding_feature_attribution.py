#!/usr/bin/env python3
# FORMALIZED HISTORICAL SOURCE SNAPSHOT
# Evidence status: HISTORICAL_OR_SUPERSEDED_DEVELOPMENT_EVIDENCE
# This file preserves a historical project stage with public-facing terminology standardized for release.
# The SHA-256 of the original executed/development source is recorded in the corresponding provenance registry.

"""
Feature attribution for EEG decoding analyses in the EEG-computational study.

This script explains which EEG features drive the decoding results from
run_eeg_feature_decoding_v1.py. It uses held-out-subject cross-validation and
summarizes:

1. Standardized logistic-regression coefficients
2. Coefficient sign stability across folds
3. Permutation importance on held-out test folds
4. Top interpretable feature families for each decoding task

Inputs:
- Trial-wise ERP feature table:
  results/eeg_neural_dynamics_v1_2_corrected/eeg_trial_neural_features_latents_corrected.csv
- Optional trial-wise time-frequency feature table:
  results/eeg_time_frequency_analysis_v1/tf_trial_bandpower_features.csv

Outputs:
- eeg_decoding_feature_attribution_fold_coefficients.csv
- eeg_decoding_feature_attribution_permutation_importance.csv
- eeg_decoding_feature_attribution_summary.csv
- EEG_DECODING_FEATURE_ATTRIBUTION_REPORT.md

Example:
python code/final/run_eeg_decoding_feature_attribution_v1.py \
  --erp_features results/eeg_neural_dynamics_v1_2_corrected/eeg_trial_neural_features_latents_corrected.csv \
  --tf_features results/eeg_time_frequency_analysis_v1/tf_trial_bandpower_features.csv \
  --out results/eeg_decoding_feature_attribution_v1
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TASK_SPECS = [
    {
        "task": "press_prediction_cue_target",
        "families": ["tf_cue_target", "erp_cue_target", "combined_cue_target"],
        "positive_class": "press",
        "interpretation": "cue/target EEG information about subsequent response tendency",
    },
    {
        "task": "nogo_failure_prediction_cue_target",
        "families": ["tf_cue_target", "erp_cue_target", "combined_cue_target"],
        "positive_class": "nogo_false_alarm",
        "interpretation": "cue/target EEG information about inhibitory-control failure",
    },
    {
        "task": "go_omission_prediction_cue_target",
        "families": ["erp_cue_target", "tf_cue_target", "combined_cue_target"],
        "positive_class": "go_miss",
        "interpretation": "cue/target EEG information about Go omission",
    },
    {
        "task": "error_monitoring_signature_response",
        "families": ["erp_response", "tf_response", "combined_response"],
        "positive_class": "nogo_false_alarm",
        "interpretation": "response-locked EEG signature of inhibitory-control failure",
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


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, score))
    except Exception:
        return np.nan


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
            "response_present", "response_type", "outcome",
            "rt_response_from_target_ms", "latent_trial_index_offset_used",
        }
        tf_feature_cols = [
            c for c in tf.columns
            if c not in meta_tf_cols and c not in key_cols
            and pd.api.types.is_numeric_dtype(tf[c])
        ]
        erp = erp.merge(
            tf[key_cols + tf_feature_cols],
            on=key_cols,
            how="left",
            suffixes=("", "_tf"),
        )

    return erp


def select_features(df: pd.DataFrame, feature_family: str) -> List[str]:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    meta_exclude = {
        "trial_index", "trial_index_numeric", "response_present",
        "rt_response_from_target_ms", "cue_onset_sec", "target_onset_sec",
        "response_onset_from_metadata_sec", "response_marker_onset_sec",
        "response_marker_error_ms", "condition_cue_match",
        "condition_trial_class_match", "p_press", "control_current",
        "actor_qdiff_current", "belief_nogo_current", "associability_current",
        "is_nogo", "action_press", "correct", "error", "fa",
        "latent_trial_index_offset_used",
    }

    erp_suffixes = (
        "_mean_uv", "_min_uv", "_max_uv",
        "_minlat_ms", "_maxlat_ms", "_slope_uv_per_s",
    )

    erp_cue_target = [
        c for c in numeric
        if (c.startswith("cue_") or c.startswith("target_"))
        and c not in meta_exclude
        and c.endswith(erp_suffixes)
    ]

    erp_response = [
        c for c in numeric
        if c.startswith("resp_")
        and c not in meta_exclude
        and c.endswith(erp_suffixes)
    ]

    tf_cue_target = [
        c for c in numeric
        if (c.startswith("cue_") or c.startswith("target_"))
        and c not in meta_exclude
        and not c.endswith(erp_suffixes)
    ]

    tf_response = [
        c for c in numeric
        if c.startswith("resp_")
        and c not in meta_exclude
        and not c.endswith(erp_suffixes)
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
        raise ValueError(f"Unknown task: {task_name}")

    return d.reset_index(drop=True), y.reset_index(drop=True)


def make_model(random_state: int = 2026) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.3,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )),
    ])


def feature_family_label(feature: str) -> str:
    if feature.startswith("resp_ern"):
        return "ERN"
    if feature.startswith("resp_pe"):
        return "Pe"
    if feature.startswith("resp_rp"):
        return "RP"
    if "theta" in feature:
        return "theta"
    if "beta" in feature:
        return "beta"
    if "alpha" in feature:
        return "alpha"
    if "mu" in feature:
        return "mu"
    if "cnv" in feature:
        return "CNV"
    if "p300" in feature or "_p3_" in feature:
        return "P300/P3"
    if "_n2_" in feature or "n2" in feature:
        return "N2"
    return "other"


def permutation_importance_on_test(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_cols: List[str],
    metric: str,
    n_repeats: int,
    rng: np.random.Generator,
) -> List[dict]:
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    if metric == "roc_auc":
        baseline = safe_auc(y_test, prob)
    elif metric == "balanced_accuracy":
        baseline = float(balanced_accuracy_score(y_test, pred))
    else:
        raise ValueError(metric)

    rows = []
    if not np.isfinite(baseline):
        return rows

    for feature in feature_cols:
        drops = []
        for _ in range(n_repeats):
            Xp = X_test.copy()
            vals = Xp[feature].values.copy()
            rng.shuffle(vals)
            Xp[feature] = vals

            pp = model.predict_proba(Xp)[:, 1]
            predp = (pp >= 0.5).astype(int)

            if metric == "roc_auc":
                scorep = safe_auc(y_test, pp)
            else:
                scorep = float(balanced_accuracy_score(y_test, predp))

            if np.isfinite(scorep):
                drops.append(baseline - scorep)

        if drops:
            rows.append({
                "feature": feature,
                "metric": metric,
                "baseline_score": float(baseline),
                "mean_score_drop": float(np.mean(drops)),
                "sd_score_drop": float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0,
                "n_repeats": int(len(drops)),
            })

    return rows


def run_attribution(
    df: pd.DataFrame,
    n_repeats: int,
    seed: int,
    max_features_for_permutation: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    coef_rows = []
    perm_rows = []
    performance_rows = []

    for spec in TASK_SPECS:
        task_name = spec["task"]
        task_df, y = build_task(df, task_name)

        if len(task_df) == 0 or len(np.unique(y)) < 2:
            continue

        groups = task_df["subject_id_norm"].map(normalize_subject_id).values
        yy = y.astype(int).values

        if len(np.unique(groups)) < 5:
            continue

        for family in spec["families"]:
            feature_cols = select_features(task_df, family)
            if not feature_cols:
                continue

            X = task_df[feature_cols].replace([np.inf, -np.inf], np.nan)

            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            fold = 0

            for train_idx, test_idx in cv.split(X, yy, groups):
                fold += 1
                if len(np.unique(yy[train_idx])) < 2 or len(np.unique(yy[test_idx])) < 2:
                    continue

                model = make_model(random_state=seed + fold)
                model.fit(X.iloc[train_idx], yy[train_idx])

                prob = model.predict_proba(X.iloc[test_idx])[:, 1]
                pred = (prob >= 0.5).astype(int)

                auc = safe_auc(yy[test_idx], prob)
                bacc = float(balanced_accuracy_score(yy[test_idx], pred))

                performance_rows.append({
                    "task": task_name,
                    "feature_family": family,
                    "fold": fold,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_features": int(len(feature_cols)),
                    "positive_rate_test": float(np.mean(yy[test_idx])),
                    "roc_auc": auc,
                    "balanced_accuracy": bacc,
                })

                clf = model.named_steps["clf"]
                coef = clf.coef_[0]

                for feature, weight in zip(feature_cols, coef):
                    coef_rows.append({
                        "task": task_name,
                        "feature_family": family,
                        "fold": fold,
                        "feature": feature,
                        "feature_type": feature_family_label(feature),
                        "standardized_logistic_coef": float(weight),
                        "abs_standardized_logistic_coef": float(abs(weight)),
                        "roc_auc_fold": auc,
                        "balanced_accuracy_fold": bacc,
                    })

                # To keep runtime manageable, run permutation importance on the
                # top absolute coefficient features if the model has many features.
                coef_order = np.argsort(np.abs(coef))[::-1]
                selected_idx = coef_order[:min(max_features_for_permutation, len(coef_order))]
                selected_features = [feature_cols[i] for i in selected_idx]

                for metric in ["roc_auc", "balanced_accuracy"]:
                    rows = permutation_importance_on_test(
                        model=model,
                        X_test=X.iloc[test_idx].copy(),
                        y_test=yy[test_idx],
                        feature_cols=selected_features,
                        metric=metric,
                        n_repeats=n_repeats,
                        rng=rng,
                    )
                    for row in rows:
                        row.update({
                            "task": task_name,
                            "feature_family": family,
                            "fold": fold,
                            "feature_type": feature_family_label(row["feature"]),
                        })
                        perm_rows.append(row)

            print(f"{task_name} | {family}: features={len(feature_cols)}")

    coef_df = pd.DataFrame(coef_rows)
    perm_df = pd.DataFrame(perm_rows)
    performance_df = pd.DataFrame(performance_rows)

    summary = summarize_attribution(coef_df, perm_df, performance_df)
    return coef_df, perm_df, summary


def summarize_attribution(
    coef_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    performance_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    if coef_df.empty:
        return pd.DataFrame()

    grouped = coef_df.groupby(["task", "feature_family", "feature", "feature_type"])

    for keys, g in grouped:
        coefs = pd.to_numeric(g["standardized_logistic_coef"], errors="coerce").dropna().values
        if len(coefs) == 0:
            continue

        task, family, feature, ftype = keys
        row = {
            "task": task,
            "feature_family": family,
            "feature": feature,
            "feature_type": ftype,
            "n_folds_with_coef": int(len(coefs)),
            "mean_standardized_coef": float(np.mean(coefs)),
            "mean_abs_standardized_coef": float(np.mean(np.abs(coefs))),
            "sd_standardized_coef": float(np.std(coefs, ddof=1)) if len(coefs) > 1 else 0.0,
            "positive_sign_fraction": float(np.mean(coefs > 0)),
            "negative_sign_fraction": float(np.mean(coefs < 0)),
            "sign_stability": float(max(np.mean(coefs > 0), np.mean(coefs < 0))),
            "mean_fold_auc": float(pd.to_numeric(g["roc_auc_fold"], errors="coerce").mean()),
            "mean_fold_balanced_accuracy": float(pd.to_numeric(g["balanced_accuracy_fold"], errors="coerce").mean()),
        }

        if not perm_df.empty:
            sub_auc = perm_df[
                (perm_df["task"] == task) &
                (perm_df["feature_family"] == family) &
                (perm_df["feature"] == feature) &
                (perm_df["metric"] == "roc_auc")
            ]
            sub_bacc = perm_df[
                (perm_df["task"] == task) &
                (perm_df["feature_family"] == family) &
                (perm_df["feature"] == feature) &
                (perm_df["metric"] == "balanced_accuracy")
            ]
            row["mean_auc_drop_when_permuted"] = float(pd.to_numeric(sub_auc["mean_score_drop"], errors="coerce").mean()) if not sub_auc.empty else np.nan
            row["mean_bacc_drop_when_permuted"] = float(pd.to_numeric(sub_bacc["mean_score_drop"], errors="coerce").mean()) if not sub_bacc.empty else np.nan
            row["n_folds_with_permutation"] = int(sub_auc["fold"].nunique()) if not sub_auc.empty else 0
        else:
            row["mean_auc_drop_when_permuted"] = np.nan
            row["mean_bacc_drop_when_permuted"] = np.nan
            row["n_folds_with_permutation"] = 0

        # Composite ranking: interpretable, stable, and useful on held-out folds.
        auc_drop = row["mean_auc_drop_when_permuted"]
        if not np.isfinite(auc_drop):
            auc_drop = 0.0
        row["attribution_score"] = (
            row["mean_abs_standardized_coef"]
            * row["sign_stability"]
            + max(0.0, auc_drop) * 2.0
        )

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["task", "feature_family", "attribution_score"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    return out


def summarize_feature_types(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    rows = []
    for keys, g in summary.groupby(["task", "feature_family", "feature_type"]):
        rows.append({
            "task": keys[0],
            "feature_family": keys[1],
            "feature_type": keys[2],
            "n_features": int(len(g)),
            "mean_abs_coef": float(g["mean_abs_standardized_coef"].mean()),
            "mean_sign_stability": float(g["sign_stability"].mean()),
            "mean_auc_drop": float(pd.to_numeric(g["mean_auc_drop_when_permuted"], errors="coerce").mean()),
            "max_attribution_score": float(g["attribution_score"].max()),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["task", "feature_family", "max_attribution_score"], ascending=[True, True, False])
    return out


def write_report(out_dir: Path, summary: pd.DataFrame, type_summary: pd.DataFrame) -> None:
    lines = []
    lines.append("# EEG decoding feature-attribution report")
    lines.append("")
    lines.append("This report explains which standardized EEG features contributed most to held-out-subject decoding models.")
    lines.append("")
    lines.append("## Top feature-level attributions")
    lines.append("")

    if summary.empty:
        lines.append("- No attribution rows were generated.")
    else:
        for (task, family), g in summary.groupby(["task", "feature_family"]):
            lines.append(f"### {task} — {family}")
            lines.append("")
            lines.append("| Rank | Feature | Type | mean coef | sign stability | AUC drop when permuted | attribution score |")
            lines.append("|---:|---|---|---:|---:|---:|---:|")
            for rank, (_, row) in enumerate(g.head(12).iterrows(), start=1):
                auc_drop = row["mean_auc_drop_when_permuted"]
                auc_text = f"{auc_drop:.4f}" if np.isfinite(auc_drop) else "NA"
                lines.append(
                    f"| {rank} | {row['feature']} | {row['feature_type']} | "
                    f"{row['mean_standardized_coef']:.4f} | {row['sign_stability']:.2f} | "
                    f"{auc_text} | {row['attribution_score']:.4f} |"
                )
            lines.append("")

    lines.append("## Feature-type summary")
    lines.append("")

    if type_summary.empty:
        lines.append("- No feature-type summary rows were generated.")
    else:
        lines.append("| Task | Feature family | Type | n features | mean abs coef | mean sign stability | mean AUC drop | max attribution score |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for _, row in type_summary.head(80).iterrows():
            auc_drop = row["mean_auc_drop"]
            auc_text = f"{auc_drop:.4f}" if np.isfinite(auc_drop) else "NA"
            lines.append(
                f"| {row['task']} | {row['feature_family']} | {row['feature_type']} | "
                f"{int(row['n_features'])} | {row['mean_abs_coef']:.4f} | "
                f"{row['mean_sign_stability']:.2f} | {auc_text} | {row['max_attribution_score']:.4f} |"
            )

    lines.append("")
    lines.append("## Interpretation guardrail")
    lines.append("")
    lines.append("- Standardized coefficients show model direction after scaling, but correlated EEG features can share weights.")
    lines.append("- Permutation importance is evaluated on held-out folds and is safer for saying which feature was useful for prediction.")
    lines.append("- For manuscript claims, prioritize features that show stable coefficient signs and positive held-out permutation importance.")
    lines.append("- Response-locked attribution supports post-response error-monitoring signatures; cue/target attribution is closer to pre-response predictive information.")
    lines.append("")

    (out_dir / "EEG_DECODING_FEATURE_ATTRIBUTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--erp_features", type=str, required=True)
    parser.add_argument("--tf_features", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_features_for_permutation", type=int, default=25)
    args = parser.parse_args()

    erp_path = Path(args.erp_features).expanduser().resolve()
    tf_path = Path(args.tf_features).expanduser().resolve() if args.tf_features else None
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(erp_path, tf_path)

    coef_df, perm_df, summary = run_attribution(
        df=df,
        n_repeats=int(args.n_repeats),
        seed=int(args.seed),
        max_features_for_permutation=int(args.max_features_for_permutation),
    )
    type_summary = summarize_feature_types(summary)

    coef_df.to_csv(out_dir / "eeg_decoding_feature_attribution_fold_coefficients.csv", index=False)
    perm_df.to_csv(out_dir / "eeg_decoding_feature_attribution_permutation_importance.csv", index=False)
    summary.to_csv(out_dir / "eeg_decoding_feature_attribution_summary.csv", index=False)
    type_summary.to_csv(out_dir / "eeg_decoding_feature_type_summary.csv", index=False)

    write_report(out_dir, summary, type_summary)

    print(f"Wrote outputs to: {out_dir}")
    if not summary.empty:
        print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
