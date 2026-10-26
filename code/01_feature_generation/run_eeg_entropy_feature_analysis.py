#!/usr/bin/env python3
"""EEG entropy and band-fraction feature analysis for EEG-computational study.

The program derives spectral-entropy summaries and normalized band fractions
from the time-frequency feature table. Entropy is computed at exact
lock/channel/window templates when multiple bands are available, at
lock-channel level across available windows, and at lock-global level across
available channels and windows.

Policy entropy and confidence are model-derived predecision quantities. Action
surprise depends on the observed action and is therefore treated as a
post-outcome model quantity. EEG spectral-entropy measures are derived from the
time-frequency representation."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
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


BANDS = ["theta", "alpha", "mu", "beta"]

RL_PRIMARY_COLS = [
    "p_press",
    "control_current",
    "actor_qdiff_current",
    "belief_nogo_current",
    "associability_current",
]

COMPUTATIONAL_ENTROPY_PREDICTORS = [
    "policy_entropy",
    "policy_confidence",
    "belief_nogo_entropy",
    "belief_nogo_confidence",
    "trial_class_surprise",
    "action_surprise",
]

PREDECISION_ENTROPY_COLS = [
    "policy_entropy",
    "policy_confidence",
    "belief_nogo_entropy",
    "belief_nogo_confidence",
]

ERP_SUFFIXES = (
    "_mean_uv", "_min_uv", "_max_uv",
    "_minlat_ms", "_maxlat_ms", "_slope_uv_per_s",
)

NON_EEG_EXCLUDE = {
    "subject_index", "group_latent", "cue_idx", "cue_type_latent",
    "trial_index", "trial_index_numeric", "trial_ordinal",
    "response_present", "response_present_numeric",
    "rt_response_from_target_ms", "rt_ms", "log_rt",
    "cue_onset_sec", "target_onset_sec", "response_onset_from_metadata_sec",
    "response_marker_onset_sec", "response_marker_error_ms",
    "condition_cue_match", "condition_trial_class_match",
    "p_press", "control_current", "actor_qdiff_current", "belief_nogo_current",
    "associability_current", "is_nogo", "action_press", "correct", "error", "fa",
    "latent_trial_index_offset_used",
}

TASK_SPECS = [
    {
        "task": "press_prediction_predecision",
        "design_cols": ["cue_type_norm", "trial_class_norm", "group"],
    },
    {
        "task": "nogo_failure_predecision",
        "design_cols": ["cue_type_norm", "group"],
    },
    {
        "task": "go_omission_predecision",
        "design_cols": ["cue_type_norm", "group"],
    },
]

MODEL_BLOCKS = {
    "design_plus_rl": ["design", "rl"],
    "design_plus_rl_plus_comp_entropy": ["design", "rl", "comp_entropy"],
    "design_plus_rl_plus_eeg_entropy": ["design", "rl", "eeg_entropy"],
    "design_plus_rl_plus_all_entropy": ["design", "rl", "comp_entropy", "eeg_entropy"],
    "comp_entropy_only": ["comp_entropy"],
    "eeg_entropy_only": ["eeg_entropy"],
    "all_entropy_only": ["comp_entropy", "eeg_entropy"],
}

PAIRWISE_COMPARISONS = [
    ("comp_entropy_increment_over_design_rl", "design_plus_rl_plus_comp_entropy", "design_plus_rl"),
    ("eeg_entropy_increment_over_design_rl", "design_plus_rl_plus_eeg_entropy", "design_plus_rl"),
    ("all_entropy_increment_over_design_rl", "design_plus_rl_plus_all_entropy", "design_plus_rl"),
    ("eeg_entropy_vs_comp_entropy", "eeg_entropy_only", "comp_entropy_only"),
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


def binary_entropy(p: pd.Series, eps: float = 1e-6) -> pd.Series:
    x = pd.to_numeric(p, errors="coerce").clip(eps, 1.0 - eps)
    return -(x * np.log2(x) + (1.0 - x) * np.log2(1.0 - x))


def binary_confidence(p: pd.Series) -> pd.Series:
    x = pd.to_numeric(p, errors="coerce")
    return (x - 0.5).abs() * 2.0


def negative_log2_prob(p: pd.Series, eps: float = 1e-6) -> pd.Series:
    x = pd.to_numeric(p, errors="coerce").clip(eps, 1.0)
    return -np.log2(x)


def load_erp(erp_path: Path) -> pd.DataFrame:
    df = pd.read_csv(erp_path)

    if "subject_id_norm" not in df.columns:
        df["subject_id_norm"] = df["subject_id"].map(normalize_subject_id)
    if "trial_index_numeric" not in df.columns:
        df["trial_index_numeric"] = pd.to_numeric(df["trial_index"], errors="coerce")

    if "group" in df.columns:
        df["group"] = df["group"].map(norm_text)
    else:
        df["group"] = df["subject_id_norm"].map(lambda s: "high" if str(s).startswith("H") else "low")

    df["trial_class_norm"] = df["trial_class"].map(norm_text) if "trial_class" in df.columns else ""
    df["cue_type_norm"] = df["cue_type"].map(norm_text) if "cue_type" in df.columns else ""
    df["response_type_norm"] = df["response_type"].map(norm_text) if "response_type" in df.columns else ""

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


def is_tf_band_col(c: str) -> bool:
    if c in NON_EEG_EXCLUDE:
        return False
    if not c.startswith(("cue_", "target_", "resp_")):
        return False
    cl = c.lower()
    return any(re.search(rf"(^|_){b}(_|$)", cl) for b in BANDS)


def load_tf(tf_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if tf_path is None or not tf_path.exists():
        return None

    tf = pd.read_csv(tf_path)
    if "subject_id_norm" not in tf.columns:
        tf["subject_id_norm"] = tf["subject_id"].map(normalize_subject_id)
    if "trial_index_numeric" not in tf.columns:
        tf["trial_index_numeric"] = pd.to_numeric(tf["trial_index"], errors="coerce")

    keep = ["subject_id_norm", "trial_index_numeric"]
    for c in tf.columns:
        if c in keep:
            continue
        if pd.api.types.is_numeric_dtype(tf[c]) and is_tf_band_col(c):
            keep.append(c)

    return tf[keep].copy()


def merge_inputs(erp: pd.DataFrame, tf: Optional[pd.DataFrame]) -> pd.DataFrame:
    if tf is None:
        return erp.copy()
    return erp.merge(tf, on=["subject_id_norm", "trial_index_numeric"], how="left", suffixes=("", "_tf"))


def add_computational_entropy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "p_press" in out.columns:
        p = pd.to_numeric(out["p_press"], errors="coerce")
        out["policy_entropy"] = binary_entropy(p)
        out["policy_confidence"] = binary_confidence(p)

        press = pd.to_numeric(out["response_present_numeric"], errors="coerce")
        action_prob = pd.Series(np.nan, index=out.index, dtype=float)
        action_prob.loc[press == 1] = p.loc[press == 1]
        action_prob.loc[press == 0] = 1.0 - p.loc[press == 0]
        out["action_surprise"] = negative_log2_prob(action_prob)

    if "belief_nogo_current" in out.columns:
        b = pd.to_numeric(out["belief_nogo_current"], errors="coerce")
        out["belief_nogo_entropy"] = binary_entropy(b)
        out["belief_nogo_confidence"] = binary_confidence(b)

        is_nogo = out["trial_class_norm"] == "nogo"
        is_go = out["trial_class_norm"] == "go"
        class_prob = pd.Series(np.nan, index=out.index, dtype=float)
        class_prob.loc[is_nogo] = b.loc[is_nogo]
        class_prob.loc[is_go] = 1.0 - b.loc[is_go]
        out["trial_class_surprise"] = negative_log2_prob(class_prob)

    return out


def parse_tf_col(c: str) -> Optional[Dict[str, str]]:
    # Expected names like:
    # cue_theta_fz_200_500
    # resp_beta_cz_m400_0
    m = re.match(r"^(cue|target|resp)_(theta|alpha|mu|beta)_([a-zA-Z0-9]+)_(.+)$", c)
    if not m:
        return None
    return {"lock": m.group(1), "band": m.group(2), "channel": m.group(3).lower(), "window": m.group(4)}


def entropy_from_power_columns(out: pd.DataFrame, band_to_cols: Dict[str, List[str]], prefix: str, mode: str) -> Tuple[pd.DataFrame, List[dict]]:
    rows = []
    available = [b for b in BANDS if b in band_to_cols and len(band_to_cols[b]) > 0]
    if len(available) < 2:
        return out, rows

    # Average positive power ratios within each band.
    band_power = []
    for b in available:
        cols = band_to_cols[b]
        X = out[cols].apply(pd.to_numeric, errors="coerce").astype(float).values
        # Existing TF features are baseline-corrected log power. exp makes them
        # positive ratios before relative normalization.
        P = np.exp(np.clip(X, -50, 50))
        band_mean = np.nanmean(P, axis=1)
        band_power.append(band_mean)

    M = np.vstack(band_power).T
    denom = np.nansum(M, axis=1)
    frac = M / denom[:, None]
    frac[~np.isfinite(frac)] = np.nan

    H = -np.nansum(frac * np.log2(np.clip(frac, 1e-12, 1.0)), axis=1)
    H_norm = H / np.log2(len(available))

    ent_col = f"{prefix}_spectral_entropy"
    out[ent_col] = H_norm

    for i, b in enumerate(available):
        out[f"{prefix}_{b}_fraction"] = frac[:, i]

    rows.append({
        "mode": mode,
        "entropy_column": ent_col,
        "available_bands": ",".join(available),
        "n_bands": len(available),
        "source_columns": ";".join([f"{b}:{','.join(band_to_cols[b])}" for b in available]),
    })
    return out, rows


def add_spectral_entropy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    tf_cols = [
        c for c in out.columns
        if pd.api.types.is_numeric_dtype(out[c]) and is_tf_band_col(c) and parse_tf_col(c) is not None
    ]

    parsed = {c: parse_tf_col(c) for c in tf_cols}
    inventory_rows = []

    # 1. exact template: same lock/channel/window
    exact = defaultdict(lambda: defaultdict(list))
    for c, info in parsed.items():
        key = f"{info['lock']}_{info['channel']}_{info['window']}"
        exact[key][info["band"]].append(c)
    for key, band_to_cols in sorted(exact.items()):
        out, rows = entropy_from_power_columns(out, band_to_cols, prefix=key, mode="exact_lock_channel_window")
        inventory_rows.extend(rows)

    # 2. lock-channel: same lock and channel, averaging across windows.
    lock_channel = defaultdict(lambda: defaultdict(list))
    for c, info in parsed.items():
        key = f"{info['lock']}_{info['channel']}_allwindows"
        lock_channel[key][info["band"]].append(c)
    for key, band_to_cols in sorted(lock_channel.items()):
        out, rows = entropy_from_power_columns(out, band_to_cols, prefix=key, mode="lock_channel_allwindows")
        inventory_rows.extend(rows)

    # 3. lock-global: cue/target/resp.
    lock_global = defaultdict(lambda: defaultdict(list))
    for c, info in parsed.items():
        key = f"{info['lock']}_global"
        lock_global[key][info["band"]].append(c)
    for key, band_to_cols in sorted(lock_global.items()):
        out, rows = entropy_from_power_columns(out, band_to_cols, prefix=key, mode="lock_global")
        inventory_rows.extend(rows)

    inventory = pd.DataFrame(inventory_rows)
    return out, inventory


def usable_numeric_cols(df: pd.DataFrame, cols: List[str], min_nonmissing: int = 20) -> List[str]:
    out = []
    for c in cols:
        if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() >= min_nonmissing and x.nunique(dropna=True) > 1:
            out.append(c)
    return out


def select_eeg_entropy_cols(df: pd.DataFrame, locks: Tuple[str, ...] = ("cue_", "target_", "resp_")) -> List[str]:
    cols = []
    for c in df.columns:
        if not c.startswith(locks):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cl = c.lower()
        if "spectral_entropy" in cl or "theta_fraction" in cl or "beta_fraction" in cl or "mu_fraction" in cl or "alpha_fraction" in cl:
            cols.append(c)
    return usable_numeric_cols(df, sorted(cols), min_nonmissing=20)


def select_targeted_eeg_cols(df: pd.DataFrame) -> List[str]:
    keywords = [
        "cue_cnv", "cue_p3", "cue_n2",
        "target_p300", "target_p3", "target_n2",
        "resp_ern", "resp_pe", "resp_rp",
        "spectral_entropy", "theta_fraction", "beta_fraction", "mu_fraction", "alpha_fraction",
    ]
    cols = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cl = c.lower()
        if any(k in cl for k in keywords):
            cols.append(c)
    return usable_numeric_cols(df, sorted(cols), min_nonmissing=50)


def fisher_z(r: float) -> float:
    if not np.isfinite(r):
        return np.nan
    return float(np.arctanh(np.clip(r, -0.999999, 0.999999)))


def safe_pearson(x: pd.Series, y: pd.Series, min_n: int) -> Tuple[float, int]:
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


def targeted_encoding(df: pd.DataFrame, min_trials: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    predictor_cols = usable_numeric_cols(df, [c for c in COMPUTATIONAL_ENTROPY_PREDICTORS if c in df.columns], min_nonmissing=50)
    target_cols = select_targeted_eeg_cols(df)

    rows = []
    min_n = max(50, min_trials)
    for subject, g in df.groupby("subject_id_norm"):
        for pred in predictor_cols:
            for target in target_cols:
                r, n = safe_pearson(g[pred], g[target], min_n=min_n)
                if np.isfinite(r):
                    rows.append({
                        "subject_id": subject,
                        "group": g["group"].iloc[0] if "group" in g.columns else "",
                        "predictor": pred,
                        "target": target,
                        "n_trials": n,
                        "r": r,
                        "fisher_z": fisher_z(r),
                    })

    rs = pd.DataFrame(rows)
    summary_rows = []
    if not rs.empty:
        for keys, g in rs.groupby(["predictor", "target"]):
            z = pd.to_numeric(g["fisher_z"], errors="coerce").dropna().values
            if len(z) < 8:
                continue
            t, p = stats.ttest_1samp(z, 0.0)
            ci = stats.t.interval(0.95, df=len(z)-1, loc=float(np.mean(z)), scale=float(stats.sem(z)))
            summary_rows.append({
                "predictor": keys[0],
                "target": keys[1],
                "n_subjects": int(len(z)),
                "mean_r": float(np.tanh(np.mean(z))),
                "ci95_low_r": float(np.tanh(ci[0])),
                "ci95_high_r": float(np.tanh(ci[1])),
                "t_value": float(t),
                "p_value": float(p),
            })

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["q_value_fdr_bh"] = fdr_bh(summary["p_value"].values)
        summary = summary.sort_values(["q_value_fdr_bh", "p_value"]).reset_index(drop=True)

    return rs, summary


def contrast_for_subject(g: pd.DataFrame, col: str, a: str, b: str, min_trials: int) -> Tuple[float, int, int]:
    va = pd.to_numeric(g.loc[g["outcome"] == a, col], errors="coerce").dropna()
    vb = pd.to_numeric(g.loc[g["outcome"] == b, col], errors="coerce").dropna()
    if len(va) < min_trials or len(vb) < min_trials:
        return np.nan, int(len(va)), int(len(vb))
    return float(va.mean() - vb.mean()), int(len(va)), int(len(vb))


def outcome_contrasts(df: pd.DataFrame, min_trials: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = select_eeg_entropy_cols(df)

    # Keep a compact set of already validated ERP features as positive controls.
    positive_controls = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and (
            c.startswith("resp_ern_fz_50_150")
            or c.startswith("resp_pe_fz_200_400")
            or c.startswith("resp_pe_cz_200_400")
            or c.startswith("target_p300_cz_250_500")
            or c.startswith("target_p300_pz_250_500")
        )
    ]
    feature_cols = sorted(set(feature_cols + positive_controls))

    contrast_specs = [
        ("nogo_fa_minus_go_hit", "nogo_fa", "go_hit"),
        ("nogo_cr_minus_nogo_fa", "nogo_cr", "nogo_fa"),
        ("go_miss_minus_go_hit", "go_miss", "go_hit"),
    ]

    subject_rows = []
    for subject, g in df.groupby("subject_id_norm"):
        for cname, a, b in contrast_specs:
            for col in feature_cols:
                if cname == "nogo_fa_minus_go_hit" and not col.startswith("resp_"):
                    continue
                if cname == "nogo_cr_minus_nogo_fa" and not (col.startswith("target_") or col.startswith("cue_")):
                    continue
                if cname == "go_miss_minus_go_hit" and not (col.startswith("target_") or col.startswith("cue_")):
                    continue
                val, na, nb = contrast_for_subject(g, col, a, b, min_trials=min_trials)
                if np.isfinite(val):
                    subject_rows.append({
                        "subject_id": subject,
                        "group": g["group"].iloc[0] if "group" in g.columns else "",
                        "contrast": cname,
                        "feature": col,
                        "outcome_a": a,
                        "outcome_b": b,
                        "n_a": na,
                        "n_b": nb,
                        "difference": val,
                    })

    subj = pd.DataFrame(subject_rows)
    rows = []
    if not subj.empty:
        for keys, g in subj.groupby(["contrast", "feature"]):
            vals = pd.to_numeric(g["difference"], errors="coerce").dropna().values
            if len(vals) < 8 or np.nanstd(vals) <= 1e-12:
                continue
            t, p = stats.ttest_1samp(vals, 0.0)
            ci = stats.t.interval(0.95, df=len(vals)-1, loc=float(np.mean(vals)), scale=float(stats.sem(vals)))
            rows.append({
                "contrast": keys[0],
                "feature": keys[1],
                "n_subjects": int(len(vals)),
                "mean_difference": float(np.mean(vals)),
                "ci95_low": float(ci[0]),
                "ci95_high": float(ci[1]),
                "t_value": float(t),
                "p_value": float(p),
                "positive_subjects": int(np.sum(vals > 0)),
                "negative_subjects": int(np.sum(vals < 0)),
                "min_trials": int(min_trials),
            })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["q_value_fdr_bh"] = fdr_bh(summary["p_value"].values)
        summary = summary.sort_values(["q_value_fdr_bh", "p_value"]).reset_index(drop=True)

    return subj, summary


def subject_summary(df: pd.DataFrame, min_trials: int) -> pd.DataFrame:
    rows = []
    entropy_cols = usable_numeric_cols(df, [c for c in COMPUTATIONAL_ENTROPY_PREDICTORS if c in df.columns], min_nonmissing=20)
    eeg_entropy_cols = select_eeg_entropy_cols(df)

    preferred_eeg = []
    for c in eeg_entropy_cols:
        if c.startswith("resp_") or c.startswith("resp_") or c.startswith("target_") or c.startswith("cue_"):
            preferred_eeg.append(c)

    for subject, g in df.groupby("subject_id_norm"):
        row = {
            "subject_id": subject,
            "group": g["group"].iloc[0] if "group" in g.columns else "",
            "n_trials": int(len(g)),
            "n_go_hit": int((g["outcome"] == "go_hit").sum()),
            "n_go_miss": int((g["outcome"] == "go_miss").sum()),
            "n_nogo_cr": int((g["outcome"] == "nogo_cr").sum()),
            "n_nogo_fa": int((g["outcome"] == "nogo_fa").sum()),
        }

        n_go = row["n_go_hit"] + row["n_go_miss"]
        n_nogo = row["n_nogo_cr"] + row["n_nogo_fa"]
        row["press_rate"] = float(pd.to_numeric(g["response_present_numeric"], errors="coerce").mean())
        row["go_miss_rate"] = row["n_go_miss"] / n_go if n_go else np.nan
        row["nogo_fa_rate"] = row["n_nogo_fa"] / n_nogo if n_nogo else np.nan
        row["mean_rt_ms"] = float(pd.to_numeric(g.loc[g["response_present_numeric"] == 1, "rt_ms"], errors="coerce").mean())

        for col in entropy_cols + [c for c in RL_PRIMARY_COLS if c in g.columns]:
            row[f"mean_{col}"] = float(pd.to_numeric(g[col], errors="coerce").mean())
            row[f"mean_{col}_go"] = float(pd.to_numeric(g.loc[g["trial_class_norm"] == "go", col], errors="coerce").mean())
            row[f"mean_{col}_nogo"] = float(pd.to_numeric(g.loc[g["trial_class_norm"] == "nogo", col], errors="coerce").mean())

        for col in preferred_eeg:
            if col.startswith("resp_"):
                val, _, _ = contrast_for_subject(g, col, "nogo_fa", "go_hit", min_trials=min_trials)
                row[f"contrast_fa_minus_hit_{col}"] = val
            elif col.startswith(("target_", "cue_")):
                val, _, _ = contrast_for_subject(g, col, "nogo_cr", "nogo_fa", min_trials=min_trials)
                row[f"contrast_cr_minus_fa_{col}"] = val

        rows.append(row)

    return pd.DataFrame(rows)


def group_tests(subj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidate_cols = [
        c for c in subj.columns
        if (
            c.startswith("mean_policy_")
            or c.startswith("mean_belief_")
            or c.startswith("mean_trial_class_surprise")
            or c.startswith("mean_action_surprise")
            or c.startswith("contrast_")
            or c in ["press_rate", "go_miss_rate", "nogo_fa_rate", "mean_rt_ms"]
        )
    ]

    for col in candidate_cols:
        high = pd.to_numeric(subj.loc[subj["group"] == "high", col], errors="coerce").dropna()
        low = pd.to_numeric(subj.loc[subj["group"] == "low", col], errors="coerce").dropna()
        if len(high) < 5 or len(low) < 5:
            continue
        if high.std() <= 1e-12 and low.std() <= 1e-12:
            continue
        t, p = stats.ttest_ind(high.values, low.values, equal_var=False)
        rows.append({
            "variable": col,
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


def subject_correlations(subj: pd.DataFrame) -> pd.DataFrame:
    neural_cols = [c for c in subj.columns if c.startswith("contrast_")]
    predictor_cols = [
        c for c in subj.columns
        if c.startswith("mean_policy_")
        or c.startswith("mean_belief_")
        or c.startswith("mean_trial_class_surprise")
        or c.startswith("mean_action_surprise")
        or c in ["press_rate", "go_miss_rate", "nogo_fa_rate", "mean_rt_ms"]
    ]

    rows = []
    for ycol in neural_cols:
        y = pd.to_numeric(subj[ycol], errors="coerce")
        for xcol in predictor_cols:
            if xcol == ycol:
                continue
            x = pd.to_numeric(subj[xcol], errors="coerce")
            mask = np.isfinite(x) & np.isfinite(y)
            n = int(mask.sum())
            if n < 10:
                continue
            if x[mask].std() <= 1e-12 or y[mask].std() <= 1e-12:
                continue
            r, p = stats.pearsonr(x[mask], y[mask])
            rho, sp = stats.spearmanr(x[mask], y[mask])
            rows.append({
                "target": ycol,
                "predictor": xcol,
                "n_subjects": n,
                "pearson_r": float(r),
                "pearson_p": float(p),
                "spearman_rho": float(rho),
                "spearman_p": float(sp),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["pearson_q_fdr_bh"] = fdr_bh(out["pearson_p"].values)
        out["spearman_q_fdr_bh"] = fdr_bh(out["spearman_p"].values)
        out = out.sort_values(["pearson_q_fdr_bh", "pearson_p"]).reset_index(drop=True)
    return out


def build_task(df: pd.DataFrame, task: str) -> Tuple[pd.DataFrame, pd.Series]:
    d = df.copy()
    if task == "press_prediction_predecision":
        d = d[d["response_present_numeric"].notna()].copy()
        y = pd.to_numeric(d["response_present_numeric"], errors="coerce").fillna(0).astype(int)
    elif task == "nogo_failure_predecision":
        d = d[d["outcome"].isin(["nogo_cr", "nogo_fa"])].copy()
        y = (d["outcome"] == "nogo_fa").astype(int)
    elif task == "go_omission_predecision":
        d = d[d["outcome"].isin(["go_hit", "go_miss"])].copy()
        y = (d["outcome"] == "go_miss").astype(int)
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


def get_prediction_blocks(df: pd.DataFrame, design_cols: List[str]) -> Dict[str, List[str]]:
    design = [c for c in design_cols if c in df.columns]
    rl = usable_numeric_cols(df, [c for c in RL_PRIMARY_COLS if c in df.columns], min_nonmissing=20)
    comp_entropy = usable_numeric_cols(df, [c for c in PREDECISION_ENTROPY_COLS if c in df.columns], min_nonmissing=20)
    eeg_entropy = [
        c for c in select_eeg_entropy_cols(df, locks=("cue_", "target_"))
        if "spectral_entropy" in c or "theta_fraction" in c or "beta_fraction" in c or "alpha_fraction" in c or "mu_fraction" in c
    ]
    return {
        "design": design,
        "rl": rl,
        "comp_entropy": comp_entropy,
        "eeg_entropy": eeg_entropy,
    }


def make_train_test_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame, blocks: Dict[str, List[str]], block_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    train_parts, test_parts = [], []

    for block in block_names:
        cols = blocks.get(block, [])
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
        train_parts.append(tr)
        test_parts.append(te)

    if not train_parts:
        return pd.DataFrame(index=train_df.index), pd.DataFrame(index=test_df.index), []

    Xtr = pd.concat(train_parts, axis=1)
    Xte = pd.concat(test_parts, axis=1)

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


def fit_score(Xtr: pd.DataFrame, Xte: pd.DataFrame, ytr: np.ndarray, yte: np.ndarray) -> Tuple[float, float]:
    if Xtr.shape[1] == 0:
        return np.nan, np.nan
    model = make_model()
    model.fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return float(balanced_accuracy_score(yte, pred)), safe_auc(yte, prob)


def incremental_prediction(df: pd.DataFrame, seed: int, min_class_n: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []

    for spec in TASK_SPECS:
        task = spec["task"]
        task_df, y = build_task(df, task)
        yy = y.astype(int).values
        if len(np.unique(yy)) < 2:
            continue

        blocks = get_prediction_blocks(task_df, spec["design_cols"])

        for model_name, block_names in MODEL_BLOCKS.items():
            possible = all(len(blocks.get(b, [])) > 0 for b in block_names)
            if not possible:
                continue

            groups = task_df["subject_id_norm"].values
            if len(np.unique(groups)) >= 5:
                cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                fold = 0
                for train_idx, test_idx in cv.split(task_df, yy, groups):
                    fold += 1
                    if len(np.unique(yy[train_idx])) < 2 or len(np.unique(yy[test_idx])) < 2:
                        continue
                    Xtr, Xte, cols = make_train_test_matrix(task_df.iloc[train_idx], task_df.iloc[test_idx], blocks, block_names)
                    bacc, auc = fit_score(Xtr, Xte, yy[train_idx], yy[test_idx])
                    rows.append({
                        "cv_type": "subject_grouped_cv",
                        "task": task,
                        "model": model_name,
                        "unit_id": f"fold_{fold}",
                        "n_features": int(len(cols)),
                        "balanced_accuracy": bacc,
                        "roc_auc": auc,
                    })

            temp = task_df.copy()
            temp["label"] = yy
            for subject, g in temp.groupby("subject_id_norm"):
                labels = g["label"].astype(int).values
                counts = np.bincount(labels, minlength=2)
                if counts.min() < min_class_n:
                    continue
                n_splits = int(min(5, counts.min()))
                if n_splits < 2:
                    continue
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                baccs, aucs, feats = [], [], []
                for train_idx, test_idx in cv.split(g, labels):
                    if len(np.unique(labels[train_idx])) < 2 or len(np.unique(labels[test_idx])) < 2:
                        continue
                    Xtr, Xte, cols = make_train_test_matrix(g.iloc[train_idx], g.iloc[test_idx], blocks, block_names)
                    bacc, auc = fit_score(Xtr, Xte, labels[train_idx], labels[test_idx])
                    if np.isfinite(bacc):
                        baccs.append(bacc)
                    if np.isfinite(auc):
                        aucs.append(auc)
                    feats.append(len(cols))
                if baccs:
                    rows.append({
                        "cv_type": "within_subject_cv",
                        "task": task,
                        "model": model_name,
                        "unit_id": subject,
                        "n_features": int(np.nanmedian(feats)) if feats else np.nan,
                        "balanced_accuracy": float(np.nanmean(baccs)),
                        "roc_auc": float(np.nanmean(aucs)) if aucs else np.nan,
                    })

            print(f"Prediction: {task} | {model_name} | blocks={block_names}")

    scores = pd.DataFrame(rows)
    summary_rows = []
    if not scores.empty:
        for keys, g in scores.groupby(["cv_type", "task", "model"]):
            for metric in ["balanced_accuracy", "roc_auc"]:
                vals = pd.to_numeric(g[metric], errors="coerce").dropna().values
                if len(vals) < 2 or np.nanstd(vals) <= 1e-12:
                    continue
                t, p = stats.ttest_1samp(vals, 0.5)
                ci = stats.t.interval(0.95, df=len(vals)-1, loc=float(np.mean(vals)), scale=float(stats.sem(vals)))
                summary_rows.append({
                    "cv_type": keys[0],
                    "task": keys[1],
                    "model": keys[2],
                    "metric": metric,
                    "n_units": int(len(vals)),
                    "mean_score": float(np.mean(vals)),
                    "ci95_low": float(ci[0]),
                    "ci95_high": float(ci[1]),
                    "p_vs_chance": float(p),
                    "mean_n_features": float(pd.to_numeric(g["n_features"], errors="coerce").mean()),
                })
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["q_value_fdr_bh_vs_chance"] = fdr_bh(summary["p_vs_chance"].values)
        summary = summary.sort_values(["task", "cv_type", "metric", "mean_score"], ascending=[True, True, True, False]).reset_index(drop=True)

    pair_rows = []
    if not scores.empty:
        for comparison, big, small in PAIRWISE_COMPARISONS:
            for keys, g in scores.groupby(["cv_type", "task"]):
                cv_type, task = keys
                wide = g.pivot_table(index="unit_id", columns="model", values=["balanced_accuracy", "roc_auc"], aggfunc="mean")
                for metric in ["balanced_accuracy", "roc_auc"]:
                    if (metric, big) not in wide.columns or (metric, small) not in wide.columns:
                        continue
                    diff = (wide[(metric, big)] - wide[(metric, small)]).dropna().values.astype(float)
                    if len(diff) < 2 or np.nanstd(diff) <= 1e-12:
                        continue
                    t, p = stats.ttest_1samp(diff, 0.0)
                    ci = stats.t.interval(0.95, df=len(diff)-1, loc=float(np.mean(diff)), scale=float(stats.sem(diff)))
                    pair_rows.append({
                        "cv_type": cv_type,
                        "task": task,
                        "comparison": comparison,
                        "larger_model": big,
                        "smaller_model": small,
                        "metric": metric,
                        "n_units": int(len(diff)),
                        "mean_delta": float(np.mean(diff)),
                        "ci95_low": float(ci[0]),
                        "ci95_high": float(ci[1]),
                        "positive_units": int(np.sum(diff > 0)),
                        "p_value": float(p),
                    })
    pairwise = pd.DataFrame(pair_rows)
    if not pairwise.empty:
        pairwise["q_value_fdr_bh"] = fdr_bh(pairwise["p_value"].values)
        pairwise = pairwise.sort_values(["q_value_fdr_bh", "p_value"]).reset_index(drop=True)

    return scores, summary, pairwise


def write_report(
    out_dir: Path,
    inventory: pd.DataFrame,
    contrasts: pd.DataFrame,
    enc_summary: pd.DataFrame,
    subj: pd.DataFrame,
    group_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    pred_summary: pd.DataFrame,
    pred_pairwise: pd.DataFrame,
    min_trials: int,
) -> None:
    lines = []
    lines.append("# EEG/RL entropy-feature analysis report")
    lines.append("")
    lines.append("This version creates computational entropy and robust EEG spectral-entropy features.")
    lines.append("")
    lines.append("## Feature construction")
    lines.append("")
    lines.append(f"- Subjects in subject summary: {len(subj)}")
    lines.append(f"- Spectral-entropy features created: {len(inventory)}")
    lines.append(f"- Minimum trials per outcome contrast: {min_trials}")

    if not inventory.empty:
        lines.append("")
        lines.append("| Mode | Entropy column | Bands | Source columns |")
        lines.append("|---|---|---|---|")
        for _, row in inventory.head(60).iterrows():
            src = str(row["source_columns"])
            if len(src) > 120:
                src = src[:117] + "..."
            lines.append(f"| {row['mode']} | {row['entropy_column']} | {row['available_bands']} | {src} |")

    # Feature variance warning
    lines.append("")
    lines.append("## Degenerate computational features")
    lines.append("")
    for col in ["belief_nogo_current", "belief_nogo_entropy", "belief_nogo_confidence", "trial_class_surprise"]:
        if col in subj.columns or col in pred_summary.columns:
            continue
    lines.append("- Belief-derived entropy features are automatically excluded from models if they have zero variance.")

    lines.append("")
    lines.append("## Outcome contrasts")
    lines.append("")
    lines.append(f"- Tested rows: {len(contrasts)}")
    lines.append(f"- FDR q < 0.05 rows: {int((contrasts['q_value_fdr_bh'] < 0.05).sum()) if not contrasts.empty else 0}")
    if not contrasts.empty:
        lines.append("")
        lines.append("| Contrast | Feature | n | Mean diff | 95% CI | p | q |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for _, row in contrasts.head(45).iterrows():
            lines.append(
                f"| {row['contrast']} | {row['feature']} | {int(row['n_subjects'])} | "
                f"{row['mean_difference']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
                f"{row['p_value']:.3g} | {row['q_value_fdr_bh']:.3g} |"
            )

    lines.append("")
    lines.append("## Targeted encoding")
    lines.append("")
    lines.append(f"- Tested rows: {len(enc_summary)}")
    lines.append(f"- FDR q < 0.05 rows: {int((enc_summary['q_value_fdr_bh'] < 0.05).sum()) if not enc_summary.empty else 0}")
    if not enc_summary.empty:
        lines.append("")
        lines.append("| Predictor | Target | n subjects | mean r | 95% CI | p | q |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for _, row in enc_summary.head(50).iterrows():
            lines.append(
                f"| {row['predictor']} | {row['target']} | {int(row['n_subjects'])} | "
                f"{row['mean_r']:.4f} | [{row['ci95_low_r']:.4f}, {row['ci95_high_r']:.4f}] | "
                f"{row['p_value']:.3g} | {row['q_value_fdr_bh']:.3g} |"
            )

    lines.append("")
    lines.append("## Subject-level group/correlation results")
    lines.append("")
    lines.append(f"- Group-test rows: {len(group_df)}; FDR q < 0.05: {int((group_df['q_value_fdr_bh'] < 0.05).sum()) if not group_df.empty else 0}")
    lines.append(f"- Correlation rows: {len(corr_df)}; Pearson FDR q < 0.05: {int((corr_df['pearson_q_fdr_bh'] < 0.05).sum()) if not corr_df.empty else 0}")

    if not group_df.empty:
        lines.append("")
        lines.append("### Top group tests")
        lines.append("| Variable | high mean | low mean | high-low | p | q |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, row in group_df.head(25).iterrows():
            lines.append(
                f"| {row['variable']} | {row['mean_high']:.4f} | {row['mean_low']:.4f} | "
                f"{row['mean_difference_high_minus_low']:.4f} | {row['p_value']:.3g} | {row['q_value_fdr_bh']:.3g} |"
            )

    if not corr_df.empty:
        lines.append("")
        lines.append("### Top subject-level correlations")
        lines.append("| Neural target | Predictor | n | Pearson r | p | q |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for _, row in corr_df.head(25).iterrows():
            lines.append(
                f"| {row['target']} | {row['predictor']} | {int(row['n_subjects'])} | "
                f"{row['pearson_r']:.4f} | {row['pearson_p']:.3g} | {row['pearson_q_fdr_bh']:.3g} |"
            )

    lines.append("")
    lines.append("## Incremental predecision prediction")
    lines.append("")
    lines.append(f"- Summary rows: {len(pred_summary)}")
    lines.append(f"- Pairwise rows: {len(pred_pairwise)}")
    lines.append(f"- Pairwise FDR q < 0.05 rows: {int((pred_pairwise['q_value_fdr_bh'] < 0.05).sum()) if not pred_pairwise.empty else 0}")
    if not pred_pairwise.empty:
        lines.append("")
        lines.append("| CV | Task | Comparison | Metric | n | Mean delta | 95% CI | p | q |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
        for _, row in pred_pairwise.head(40).iterrows():
            lines.append(
                f"| {row['cv_type']} | {row['task']} | {row['comparison']} | {row['metric']} | "
                f"{int(row['n_units'])} | {row['mean_delta']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
                f"{row['p_value']:.3g} | {row['q_value_fdr_bh']:.3g} |"
            )

    lines.append("")
    lines.append("## Interpretation constraints")
    lines.append("")
    lines.append("- Entropy results should be interpreted within the predefined feature family and with the reported FDR-corrected inference.")
    lines.append("- Policy entropy and confidence are model-derived computational uncertainty features.")
    lines.append("- Belief entropy is not useful if belief_nogo_current is constant; this version filters zero-variance features.")
    lines.append("- Action surprise uses the observed response and must not be used as a predecision predictor.")
    lines.append("- EEG spectral entropy is derived from existing TF bandpower features and should be interpreted as a compact TF-distribution summary.")
    lines.append("- Inferential claims require the corresponding permutation/bootstrap validation and multiple-comparison control.")
    lines.append("")

    (out_dir / "ENTROPY_FEATURE_ANALYSIS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--erp_features", type=str, required=True)
    parser.add_argument("--tf_features", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--min_trials", type=int, default=5)
    parser.add_argument("--min_class_n_subject_cv", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    erp_path = Path(args.erp_features).expanduser().resolve()
    tf_path = Path(args.tf_features).expanduser().resolve() if args.tf_features else None
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    erp = load_erp(erp_path)
    tf = load_tf(tf_path)
    df = merge_inputs(erp, tf)

    df = add_computational_entropy(df)
    df, inventory = add_spectral_entropy(df)

    contrast_rows, contrasts = outcome_contrasts(df, min_trials=int(args.min_trials))
    enc_rs, enc_summary = targeted_encoding(df, min_trials=int(args.min_trials))
    subj = subject_summary(df, min_trials=int(args.min_trials))
    group_df = group_tests(subj)
    corr_df = subject_correlations(subj)
    pred_scores, pred_summary, pred_pairwise = incremental_prediction(
        df,
        seed=int(args.seed),
        min_class_n=int(args.min_class_n_subject_cv),
    )

    df.to_csv(out_dir / "entropy_trial_features.csv", index=False)
    inventory.to_csv(out_dir / "entropy_feature_inventory.csv", index=False)
    contrast_rows.to_csv(out_dir / "entropy_outcome_contrast_subject_rows.csv", index=False)
    contrasts.to_csv(out_dir / "entropy_outcome_contrasts_subject_level.csv", index=False)
    enc_rs.to_csv(out_dir / "entropy_targeted_encoding_subject_rs.csv", index=False)
    enc_summary.to_csv(out_dir / "entropy_targeted_encoding_summary.csv", index=False)
    subj.to_csv(out_dir / "entropy_subject_summary.csv", index=False)
    group_df.to_csv(out_dir / "entropy_subject_group_tests.csv", index=False)
    corr_df.to_csv(out_dir / "entropy_subject_correlations.csv", index=False)
    pred_scores.to_csv(out_dir / "entropy_incremental_prediction_scores.csv", index=False)
    pred_summary.to_csv(out_dir / "entropy_incremental_prediction_summary.csv", index=False)
    pred_pairwise.to_csv(out_dir / "entropy_incremental_prediction_pairwise.csv", index=False)

    write_report(
        out_dir=out_dir,
        inventory=inventory,
        contrasts=contrasts,
        enc_summary=enc_summary,
        subj=subj,
        group_df=group_df,
        corr_df=corr_df,
        pred_summary=pred_summary,
        pred_pairwise=pred_pairwise,
        min_trials=int(args.min_trials),
    )

    print(f"Wrote entropy feature analysis to: {out_dir}")
    print((out_dir / "ENTROPY_FEATURE_ANALYSIS_REPORT.md").read_text(encoding="utf-8")[:6000])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
