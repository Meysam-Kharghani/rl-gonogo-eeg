#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Censored one-boundary diffusion model for Cued Go/NoGo data with
reinforcement-learning latent predictors.

For response trials, the likelihood uses the first-passage-time density of the
upper response boundary. For no-response trials, the likelihood uses survival
up to the analysis deadline. This formulation represents omission trials as
censored events rather than as a second observed response category.

The likelihood is implemented analytically with NumPy and SciPy. The module
supports data inspection, model fitting, participant-level cross-validation,
and model-comparison summaries."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json
import math
import time
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

EPS = 1e-12


# =============================================================================
# Utilities
# =============================================================================

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(p: Path, obj) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(p: Path, default=None):
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def softplus(x: np.ndarray) -> np.ndarray:
    # stable softplus
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def zscore_train_apply(train: pd.Series, x: pd.Series) -> Tuple[np.ndarray, float, float]:
    tr = pd.to_numeric(train, errors="coerce").astype(float)
    mu = float(np.nanmean(tr))
    sd = float(np.nanstd(tr))
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1.0
    y = (pd.to_numeric(x, errors="coerce").astype(float).to_numpy() - mu) / sd
    y[~np.isfinite(y)] = 0.0
    return y, mu, sd


def safe_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


# =============================================================================
# Data preparation
# =============================================================================

def build_raw_trial_table(data_root: Path) -> pd.DataFrame:
    """Read raw subject xlsx files and return behavioral columns needed for DDM.

    The RL latent export contains model states and action labels, but some
    exports do not include raw RT. Because RT is required for DDM fitting, the
    behavioral RT columns are merged from the original *_metadata_features.xlsx files.
    """
    data_root = Path(data_root)
    paths = sorted(data_root.rglob("*_metadata_features.xlsx"))
    paths = [p for p in paths if "all_" not in p.name.lower() and "merged" not in p.name.lower()]
    if not paths:
        raise FileNotFoundError(f"No individual '*_metadata_features.xlsx' files found under {data_root}")

    rows = []
    required = ["trial_index", "trial_class", "cue_type", "response_present", "rt_response_from_target_ms", "response_type"]
    for fp in paths:
        subj_id = fp.stem.split("_metadata_features")[0]
        df = pd.read_excel(fp)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{fp} missing required columns: {missing}")
        tmp = df[required].copy()
        tmp["subj_id_ddm"] = str(subj_id)
        rows.append(tmp)

    raw = pd.concat(rows, ignore_index=True)
    raw = raw.rename(columns={
        "response_present": "raw_response_present",
        "rt_response_from_target_ms": "raw_rt_response_from_target_ms",
        "response_type": "raw_response_type",
        "trial_class": "raw_trial_class",
        "cue_type": "raw_cue_type",
    })
    raw["trial_index"] = pd.to_numeric(raw["trial_index"], errors="coerce").astype(int)
    return raw


def maybe_merge_raw_behavior(latents: pd.DataFrame, subj_col: str, data_root: Optional[Path]) -> pd.DataFrame:
    """Merge RT/response columns from raw data if they are absent in latent CSV.

    Trial-index alignment:
    Some exported latent files use zero-based trial_index (0..379), whereas the raw
    xlsx files may use one-based trial_index (1..380). Common trial-index offsets
    are evaluated and the alignment with the fewest unmatched rows is selected.
    """
    d0 = latents.copy()
    d0["subj_id_ddm"] = d0[subj_col].astype(str)
    if "trial_index" in d0.columns:
        d0["trial_index"] = pd.to_numeric(d0["trial_index"], errors="coerce").astype(int)

    has_action = safe_col(d0, ["response_present", "action", "press", "action_press"], required=False) is not None
    has_rt = safe_col(d0, ["rt_response_from_target_ms", "rt_ms", "rt", "raw_rt_response_from_target_ms"], required=False) is not None
    has_resp = safe_col(d0, ["response_type", "raw_response_type"], required=False) is not None

    if has_action and has_rt and has_resp:
        return d0

    if data_root is None:
        missing = []
        if not has_action: missing.append("response/action column")
        if not has_rt: missing.append("RT column")
        if not has_resp: missing.append("response_type column")
        raise ValueError(
            "Latent CSV is missing " + ", ".join(missing) + ".\n"
            "Pass --data_root data/raw/csv so the script can merge raw behavioral columns."
        )

    raw0 = build_raw_trial_table(Path(data_root))
    if "trial_index" not in d0.columns:
        raise ValueError("Latent CSV has no trial_index, so it cannot be merged with raw behavioral data.")

    before = len(d0)

    def attempt_merge(offset: int):
        d = d0.copy()
        raw = raw0.copy()
        d["__merge_trial_index"] = d["trial_index"] + int(offset)
        raw = raw.rename(columns={"trial_index": "__merge_trial_index"})
        merged = d.merge(raw, on=["subj_id_ddm", "__merge_trial_index"], how="left", validate="one_to_one")
        if len(merged) != before:
            raise RuntimeError("Merge changed number of rows; check subject_id/trial_index uniqueness.")
        miss_identity = int(merged["raw_response_present"].isna().sum()) if "raw_response_present" in merged.columns else before
        return merged, miss_identity

    # Evaluate common index conventions.
    attempts = []
    for off in [0, 1, -1]:
        try:
            merged, miss = attempt_merge(off)
            attempts.append((miss, off, merged))
        except Exception as e:
            attempts.append((before + 999, off, e))

    attempts_sorted = sorted(attempts, key=lambda x: x[0])
    best_miss, best_offset, best_obj = attempts_sorted[0]

    if not isinstance(best_obj, pd.DataFrame):
        raise RuntimeError(f"All raw merge attempts failed. Best offset {best_offset} error: {best_obj}")

    d = best_obj.drop(columns=["__merge_trial_index"], errors="ignore")
    d.attrs["raw_merge_trial_index_offset"] = int(best_offset)
    d.attrs["raw_merge_missing_identity"] = int(best_miss)

    if best_miss > 0:
        # Retain merge diagnostics for error reporting.
        unmatched = d.loc[d["raw_response_present"].isna(), ["subj_id_ddm", "trial_index"]].head(20).to_dict("records")
        raise ValueError(
            f"Raw merge failed for {best_miss} rows with missing response identity. "
            f"Tried offsets [0, +1, -1]; best offset was {best_offset}. "
            f"First unmatched rows: {unmatched}"
        )

    # If the latent file already has action_press, use it for press status;
    # otherwise use raw_response_present.
    action_col_tmp = safe_col(d, ["response_present", "action", "press", "action_press", "raw_response_present"], required=True)
    press_tmp = pd.to_numeric(d[action_col_tmp], errors="coerce").fillna(0).astype(int).clip(0, 1)

    if "raw_rt_response_from_target_ms" in d.columns:
        missing_press_rt = int(((press_tmp == 1) & d["raw_rt_response_from_target_ms"].isna()).sum())
        if missing_press_rt > 0:
            bad = d.loc[(press_tmp == 1) & d["raw_rt_response_from_target_ms"].isna(),
                        ["subj_id_ddm", "trial_index"]].head(20).to_dict("records")
            raise ValueError(
                f"Raw merge produced {missing_press_rt} press trials with missing RT. "
                f"Best trial_index offset was {best_offset}. First bad rows: {bad}"
            )
    return d

def load_latents(path: Path, deadline_ms: float, data_root: Optional[Path] = None) -> pd.DataFrame:
    d = pd.read_csv(path)
    d.columns = [str(c) for c in d.columns]

    subj_col = safe_col(d, ["subj_id", "subject_id", "subject", "subj"], required=True)
    d = maybe_merge_raw_behavior(d, subj_col=subj_col, data_root=data_root)

    group_col = safe_col(d, ["group", "risk_group", "group_label"], required=False)
    cue_col = safe_col(d, ["cue_type", "cue", "cue_label", "raw_cue_type"], required=True)
    cls_col = safe_col(d, ["trial_class", "is_nogo", "raw_trial_class"], required=True)
    action_col = safe_col(d, ["response_present", "action", "press", "action_press", "raw_response_present"], required=True)
    rt_col = safe_col(d, ["rt_response_from_target_ms", "rt_ms", "rt", "raw_rt_response_from_target_ms"], required=True)
    resp_col = safe_col(d, ["response_type", "raw_response_type"], required=False)

    out = d.copy()
    if "subj_id_ddm" not in out.columns:
        out["subj_id_ddm"] = out[subj_col].astype(str)

    if group_col is not None:
        g = out[group_col]
        if pd.api.types.is_numeric_dtype(g):
            out["group_high"] = pd.to_numeric(g, errors="coerce").fillna(0).astype(int).clip(0, 1)
        else:
            gs = g.astype(str).str.lower()
            out["group_high"] = gs.str.contains("high|h").astype(int)
    else:
        out["group_high"] = out["subj_id_ddm"].str.lower().str.startswith("h").astype(int)

    cue = out[cue_col].astype(str).str.lower()
    out["cue_down"] = (cue == "down").astype(int)
    out["cue_bi"] = cue.isin(["bi", "bidirectional", "both"]).astype(int)
    out["cue_up"] = (cue == "up").astype(int)
    out["cue_label_ddm"] = np.where(out["cue_up"] == 1, "up", np.where(out["cue_bi"] == 1, "bi", "down"))

    if cls_col == "is_nogo":
        out["is_nogo"] = pd.to_numeric(out[cls_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    else:
        out["is_nogo"] = out[cls_col].astype(str).str.lower().eq("nogo").astype(int)

    out["press"] = pd.to_numeric(out[action_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    out["rt_sec"] = pd.to_numeric(out[rt_col], errors="coerce").astype(float) / 1000.0
    out.loc[out["press"] == 0, "rt_sec"] = np.nan

    if resp_col is not None:
        rs = out[resp_col].astype(str).str.lower()
        out["fa"] = (rs == "fa").astype(int)
        out["hit"] = (rs == "hit").astype(int)
        out["miss"] = (rs == "miss").astype(int)
        out["cr"] = (rs == "cr").astype(int)
    else:
        out["fa"] = ((out["is_nogo"] == 1) & (out["press"] == 1)).astype(int)
        out["hit"] = ((out["is_nogo"] == 0) & (out["press"] == 1)).astype(int)
        out["miss"] = ((out["is_nogo"] == 0) & (out["press"] == 0)).astype(int)
        out["cr"] = ((out["is_nogo"] == 1) & (out["press"] == 0)).astype(int)

    out["correct"] = (((out["is_nogo"] == 0) & (out["press"] == 1)) |
                      ((out["is_nogo"] == 1) & (out["press"] == 0))).astype(int)
    out["error"] = 1 - out["correct"]
    out = out.sort_values(["subj_id_ddm", "trial_index"] if "trial_index" in out.columns else ["subj_id_ddm"]).reset_index(drop=True)
    out["prev_error"] = out.groupby("subj_id_ddm")["error"].shift(1).fillna(0).astype(int)
    out["prev_fa"] = out.groupby("subj_id_ddm")["fa"].shift(1).fillna(0).astype(int)
    out["prev_press"] = out.groupby("subj_id_ddm")["press"].shift(1).fillna(0).astype(int)

    # Online behavioral-summary controls. These use only trials preceding the
    # current trial within a subject, so they do not leak current/future behavior
    # into trial-wise DDM prediction. Laplace smoothing avoids undefined early
    # trials and extreme 0/1 rates.
    out["trial_n_prior"] = out.groupby("subj_id_ddm").cumcount().astype(float)

    # The shift is performed within participant. A bare
    # groupby(...).cumsum().shift(1) shifts across subject boundaries and can
    # otherwise transfers the final trial of one participant to the first trial of the next.
    # The transform/lambda pattern below keeps all online counts subject-safe.
    out["prior_press_count"] = (
        out.groupby("subj_id_ddm")["press"]
        .transform(lambda s: s.cumsum().shift(1))
        .fillna(0.0)
        .astype(float)
    )
    out["prior_press_rate"] = (out["prior_press_count"] + 0.5) / (out["trial_n_prior"] + 1.0)

    out["prior_nogo_count"] = (
        out.groupby("subj_id_ddm")["is_nogo"]
        .transform(lambda s: s.cumsum().shift(1))
        .fillna(0.0)
        .astype(float)
    )
    out["prior_fa_count"] = (
        out.groupby("subj_id_ddm")["fa"]
        .transform(lambda s: s.cumsum().shift(1))
        .fillna(0.0)
        .astype(float)
    )
    out["prior_fa_rate"] = (out["prior_fa_count"] + 0.5) / (out["prior_nogo_count"] + 1.0)
    out["deadline_sec"] = float(deadline_ms) / 1000.0

    for c in [
        "control_current", "actor_qdiff_current", "p_press",
        "belief_nogo_current", "associability_current",
        "cnv_z", "p300_z", "ern_strength_z", "pe_strength_z"
    ]:
        if c not in out.columns:
            out[c] = 0.0

    out["valid_press_rt"] = (out["press"] == 1) & np.isfinite(out["rt_sec"]) & (out["rt_sec"] > 0.05)
    return out


def inspect(path: Path, deadline_ms: float, data_root: Optional[Path] = None,
            rt_min_ms: float = 100.0, rt_max_ms: float = 1000.0) -> None:
    d = load_latents(path, deadline_ms=deadline_ms, data_root=data_root)
    d2, filt = filter_ddm_trials(d, rt_min_ms=rt_min_ms, rt_max_ms=rt_max_ms)
    print(json.dumps({
        "path": str(path),
        "n_trials_raw": int(len(d)),
        "n_trials_after_filter": int(len(d2)),
        "filter": filt,
        "n_subjects": int(d2["subj_id_ddm"].nunique()),
        "group_counts_subjects": d2.groupby("subj_id_ddm")["group_high"].first().value_counts().sort_index().to_dict(),
        "press_rate": float(d2["press"].mean()),
        "hit_rate_go": float(d2.loc[d2["is_nogo"] == 0, "press"].mean()),
        "fa_rate_nogo": float(d2.loc[d2["is_nogo"] == 1, "press"].mean()),
        "valid_press_rt": int(d2["valid_press_rt"].sum()),
        "deadline_ms": deadline_ms,
        "columns": list(d2.columns),
        "latent_means": {
            c: float(pd.to_numeric(d2[c], errors="coerce").mean())
            for c in ["control_current", "actor_qdiff_current", "belief_nogo_current", "associability_current", "p_press"]
            if c in d2.columns
        },
    }, indent=2, ensure_ascii=False))


def filter_ddm_trials(d: pd.DataFrame, rt_min_ms: float = 100.0, rt_max_ms: float = 1000.0) -> Tuple[pd.DataFrame, Dict]:
    """Remove invalid/anticipatory press RT trials while keeping no-response trials.

    No-response trials legitimately have missing RT and are retained as censored.
    Press trials require a finite RT in [rt_min_ms, rt_max_ms].
    """
    df = d.copy()
    rt_min = float(rt_min_ms) / 1000.0
    rt_max = float(rt_max_ms) / 1000.0
    press = df["press"].astype(int).to_numpy() == 1
    rt = df["rt_sec"].to_numpy(dtype=float)
    valid_press = press & np.isfinite(rt) & (rt >= rt_min) & (rt <= rt_max)
    keep = (~press) | valid_press
    out = df.loc[keep].reset_index(drop=True)
    info = {
        "rt_min_ms": float(rt_min_ms),
        "rt_max_ms": float(rt_max_ms),
        "n_raw": int(len(df)),
        "n_kept": int(len(out)),
        "n_removed": int((~keep).sum()),
        "n_removed_press_invalid_rt": int((press & ~valid_press).sum()),
        "press_raw": int(press.sum()),
        "press_kept": int(out["press"].sum()),
        "note": "Only press trials with invalid/anticipatory RT are removed; no-response trials are retained as censored."
    }
    return out, info


# =============================================================================
# Model specifications
# =============================================================================

@dataclass(frozen=True)
class DDMVariant:
    key: str
    name: str
    drift_cols: Tuple[str, ...]
    dist_cols: Tuple[str, ...]
    t0_cols: Tuple[str, ...] = ("intercept",)


VARIANTS: Dict[str, DDMVariant] = {
    # Baseline: categorical/task variables only.
    "D0": DDMVariant(
        key="D0",
        name="baseline_cue_nogo_group",
        drift_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high"),
        dist_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high", "prev_error"),
    ),

    # Post-error DDM baseline: stronger previous error/FA terms in distance.
    "D1": DDMVariant(
        key="D1",
        name="post_error_caution",
        drift_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high"),
        dist_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high", "prev_error", "prev_fa"),
    ),

    # Main RL-informed DDM with M6 latents.
    # Historical model name retained for compatibility; D2 contains p_press,
    # actor Q-difference, and control and is therefore the full RL block.
    "D2": DDMVariant(
        key="D2",
        name="rl_actor_control",
        drift_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high", "actor_qdiff_z", "p_press_z"),
        dist_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high", "prev_error", "control_z"),
    ),

    # RL + core ERP covariates: CNV for response preparation/caution, P300 for target/inhibition processing.
    "D3": DDMVariant(
        key="D3",
        name="rl_actor_control_erp",
        drift_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high", "actor_qdiff_z", "p_press_z", "p300_z_global"),
        dist_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high", "prev_error", "control_z", "cnv_z_global"),
    ),

    # M7 sensitivity: belief and associability/volatility.
    "D4": DDMVariant(
        key="D4",
        name="rl_actor_control_belief_volatility",
        drift_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high", "actor_qdiff_z", "p_press_z", "belief_z"),
        dist_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high", "prev_error", "control_z", "assoc_z"),
    ),

    # More compact RL version without p_press to avoid circularity concerns.
    "D5": DDMVariant(
        key="D5",
        name="rl_actor_control_no_ppress",
        drift_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high", "actor_qdiff_z"),
        dist_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high", "prev_error", "control_z"),
    ),

    # Behavioral-summary control. This replaces RL policy states with simple,
    # online behavioral summaries available before each trial. It tests whether
    # the reinforcement-learning increment can be explained by press frequency or behavioral history.
    "D6": DDMVariant(
        key="D6",
        name="online_behavioral_summary_control",
        drift_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high", "prev_press", "prior_press_rate_z", "prior_fa_rate_z"),
        dist_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high", "prev_error", "prev_fa", "prior_press_rate_z", "prior_fa_rate_z"),
    ),

    # Train-fold behavioral-rate control. This is a less adaptive benchmark than
    # D6: it uses only rates estimated from the training subjects within
    # group x cue x target bins and then applies those rates to the held-out
    # subjects. It tests whether RL-policy latents outperform non-RL behavioral
    # summaries that do not adapt online to each test subject's previous choices.
    "D6A": DDMVariant(
        key="D6A",
        name="train_fold_behavioral_rate_control",
        drift_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high", "train_bin_press_rate_z", "train_bin_fa_rate_z"),
        dist_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high", "prev_error", "train_bin_press_rate_z", "train_bin_fa_rate_z"),
    ),

    # Residualized-policy control. p_press is residualized against task, group,
    # previous-trial behavior, and online behavioral summaries using train-fold
    # coefficients only; the residual is then used in place of p_press.
    "D7": DDMVariant(
        key="D7",
        name="rl_actor_control_residualized_policy",
        drift_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high", "actor_qdiff_z", "p_press_resid_z"),
        dist_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high", "prev_error", "control_z"),
    ),

    # Trial-alignment placebo. This keeps the marginal distribution of p_press
    # within subject x cue x target bins but deterministically rotates it across
    # trials, breaking trial-wise alignment while preserving coarse structure.
    "D8": DDMVariant(
        key="D8",
        name="within_subject_policy_shuffled_placebo",
        drift_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high", "actor_qdiff_z", "p_press_shuf_z"),
        dist_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high", "prev_error", "control_z"),
    ),
}


CONTINUOUS_SOURCE_MAP = {
    "control_z": "control_current",
    "actor_qdiff_z": "actor_qdiff_current",
    "p_press_z": "p_press",
    "belief_z": "belief_nogo_current",
    "assoc_z": "associability_current",
    "cnv_z_global": "cnv_z",
    "p300_z_global": "p300_z",
    "prior_press_rate_z": "prior_press_rate",
    "prior_fa_rate_z": "prior_fa_rate",
    "train_bin_press_rate_z": "train_bin_press_rate",
    "train_bin_fa_rate_z": "train_bin_fa_rate",
    "p_press_shuf_z": "p_press_shuffled",
}


# =============================================================================
# Design matrices
# =============================================================================


def deterministic_within_bin_rotation(df: pd.DataFrame, value_col: str = "p_press") -> np.ndarray:
    """Deterministically rotate a value within subject x cue x target bins.

    This is a placebo for trial-wise policy alignment. It preserves within-bin
    marginal distributions while assigning each trial another trial's policy
    value. It is deterministic, so CV results are exactly reproducible.
    """
    if value_col not in df.columns:
        return np.zeros(len(df), dtype=float)
    out = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0).astype(float).to_numpy().copy()
    work = df.copy().reset_index(drop=True)
    sort_cols = [c for c in ["subj_id_ddm", "cue_label_ddm", "is_nogo", "trial_index"] if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols)
    group_cols = [c for c in ["subj_id_ddm", "cue_label_ddm", "is_nogo"] if c in work.columns]
    if not group_cols:
        return np.roll(out, 1)
    for _, idx in work.groupby(group_cols, sort=False).groups.items():
        idx = list(idx)
        if len(idx) >= 2:
            vals = pd.to_numeric(df.iloc[idx][value_col], errors="coerce").fillna(0.0).astype(float).to_numpy()
            # Rotate by approximately half the bin size rather than one step to
            # avoid only shifting to the immediately adjacent trial.
            shift = max(1, len(vals) // 2)
            out[np.asarray(idx, dtype=int)] = np.roll(vals, shift)
    return out




def add_train_fold_behavioral_rates(train: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Add train-derived behavioral-rate covariates to train and target rows.

    Rates are estimated only from the training fold and then merged onto both
    train and target rows by group x cue x target bins.  Held-out subjects do
    not contribute to their own rate covariates, so this control does not use
    online behavior from the test subject.  Laplace smoothing is used to avoid
    extreme 0/1 rates in sparse bins.
    """
    train = train.copy().reset_index(drop=True)
    target = target.copy().reset_index(drop=True)
    keys = [c for c in ["group_high", "cue_label_ddm", "is_nogo"] if c in train.columns and c in target.columns]
    if not keys:
        global_press = (float(pd.to_numeric(train.get("press", 0), errors="coerce").fillna(0).sum()) + 0.5) / (len(train) + 1.0)
        global_nogo = pd.to_numeric(train.get("is_nogo", 0), errors="coerce").fillna(0).astype(int) == 1
        global_fa_count = float(pd.to_numeric(train.loc[global_nogo, "fa"], errors="coerce").fillna(0).sum()) if "fa" in train.columns else 0.0
        global_nogo_count = float(global_nogo.sum())
        global_fa_rate = (global_fa_count + 0.5) / (global_nogo_count + 1.0)
        for frame in (train, target):
            frame["train_bin_press_rate"] = global_press
            frame["train_bin_fa_rate"] = global_fa_rate
        return train, target, {"keys": keys, "fallback": "global", "global_press_rate": global_press, "global_fa_rate": global_fa_rate}

    tr = train.copy()
    tr["press_num"] = pd.to_numeric(tr.get("press", 0), errors="coerce").fillna(0.0).astype(float)
    tr["fa_num"] = pd.to_numeric(tr.get("fa", 0), errors="coerce").fillna(0.0).astype(float)
    tr["nogo_num"] = pd.to_numeric(tr.get("is_nogo", 0), errors="coerce").fillna(0.0).astype(float)

    grp = tr.groupby(keys, dropna=False)
    rates = grp.agg(
        n_trials=("press_num", "size"),
        press_count=("press_num", "sum"),
        nogo_count=("nogo_num", "sum"),
        fa_count=("fa_num", "sum"),
    ).reset_index()
    rates["train_bin_press_rate"] = (rates["press_count"] + 0.5) / (rates["n_trials"] + 1.0)
    rates["train_bin_fa_rate"] = (rates["fa_count"] + 0.5) / (rates["nogo_count"] + 1.0)

    global_press = (float(tr["press_num"].sum()) + 0.5) / (len(tr) + 1.0)
    global_fa_rate = (float(tr["fa_num"].sum()) + 0.5) / (float(tr["nogo_num"].sum()) + 1.0)

    def merge_rates(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.merge(rates[keys + ["train_bin_press_rate", "train_bin_fa_rate"]], on=keys, how="left")
        out["train_bin_press_rate"] = pd.to_numeric(out["train_bin_press_rate"], errors="coerce").fillna(global_press).astype(float)
        out["train_bin_fa_rate"] = pd.to_numeric(out["train_bin_fa_rate"], errors="coerce").fillna(global_fa_rate).astype(float)
        return out

    meta = {
        "keys": keys,
        "n_bins": int(len(rates)),
        "global_press_rate": float(global_press),
        "global_fa_rate": float(global_fa_rate),
        "smoothing": "Laplace: (count + 0.5)/(n + 1)",
    }
    return merge_rates(train), merge_rates(target), meta


def residualize_policy_from_train(train: pd.DataFrame, target: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Residualize p_press against non-RL behavioral/task summaries.

    Regression coefficients are estimated on the train fold and then applied to
    both train and target rows. This prevents test-fold outcomes from shaping the
    residualization model.
    """
    predictors = [
        "intercept", "is_nogo", "cue_up", "cue_bi", "group_high",
        "prev_error", "prev_press", "prior_press_rate_z", "prior_fa_rate_z",
    ]
    for frame in (train, target):
        if "intercept" not in frame.columns:
            frame["intercept"] = 1.0
        for c in predictors:
            if c not in frame.columns:
                frame[c] = 0.0
    Xtr = train[predictors].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    Xte = target[predictors].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ytr = pd.to_numeric(train.get("p_press", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    yte = pd.to_numeric(target.get("p_press", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    resid_train = ytr - Xtr @ beta
    resid_target = yte - Xte @ beta
    mu = float(np.nanmean(resid_train))
    sd = float(np.nanstd(resid_train))
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1.0
    resid_train_z = (resid_train - mu) / sd
    resid_target_z = (resid_target - mu) / sd
    resid_train_z[~np.isfinite(resid_train_z)] = 0.0
    resid_target_z[~np.isfinite(resid_target_z)] = 0.0
    meta = {"predictors": predictors, "beta": beta.tolist(), "resid_mean": mu, "resid_sd": sd}
    return resid_train_z, resid_target_z, meta


def prepare_designs(train_df: pd.DataFrame, target_df: pd.DataFrame, variant: DDMVariant) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    df = target_df.copy().reset_index(drop=True)
    train = train_df.copy().reset_index(drop=True)

    # Derived placebo/control sources.
    train["p_press_shuffled"] = deterministic_within_bin_rotation(train, "p_press")
    df["p_press_shuffled"] = deterministic_within_bin_rotation(df, "p_press")

    # Train-fold behavioral-rate covariates for D6A. These are estimated from
    # train only and then applied to target rows.
    train, df, train_rate_meta = add_train_fold_behavioral_rates(train, df)

    # Add continuous standardized columns to df and to train. The train-standardized
    # columns are needed for residualized-policy controls.
    standardization = {}
    for new_col, src_col in CONTINUOUS_SOURCE_MAP.items():
        if src_col in df.columns and src_col in train.columns:
            z, mu, sd = zscore_train_apply(train[src_col], df[src_col])
            tr_raw = pd.to_numeric(train[src_col], errors="coerce").astype(float).to_numpy()
            ztr = (tr_raw - mu) / sd
            ztr[~np.isfinite(ztr)] = 0.0
            df[new_col] = z
            train[new_col] = ztr
            standardization[new_col] = {"source": src_col, "mean": mu, "sd": sd}
        else:
            df[new_col] = 0.0
            train[new_col] = 0.0
            standardization[new_col] = {"source": src_col, "mean": 0.0, "sd": 1.0, "missing": True}

    df["intercept"] = 1.0
    train["intercept"] = 1.0

    # Residualized policy: residualize p_press against non-RL behavioral summaries.
    train_resid_z, target_resid_z, resid_meta = residualize_policy_from_train(train, df)
    train["p_press_resid_z"] = train_resid_z
    df["p_press_resid_z"] = target_resid_z

    def mat(cols: Tuple[str, ...]) -> np.ndarray:
        X = []
        for c in cols:
            if c not in df.columns:
                raise ValueError(f"Missing design column {c} for variant {variant.key}")
            X.append(pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float).to_numpy())
        return np.column_stack(X).astype(float)

    Xv = mat(variant.drift_cols)
    Xd = mat(variant.dist_cols)
    Xt0 = mat(variant.t0_cols)

    meta = {
        "drift_cols": list(variant.drift_cols),
        "dist_cols": list(variant.dist_cols),
        "t0_cols": list(variant.t0_cols),
        "standardization": standardization,
        "residualized_policy": resid_meta,
        "placebo_policy": {
            "method": "deterministic rotation within subject x cue x target bins",
            "source": "p_press",
            "column": "p_press_shuf_z",
        },
        "train_fold_behavioral_rates": train_rate_meta,
    }
    return Xv, Xd, Xt0, meta


# =============================================================================
# One-boundary DDM likelihood
# =============================================================================

def unpack_params(theta: np.ndarray, nv: int, nd: int, nt0: int):
    beta_v = theta[:nv]
    beta_d = theta[nv:nv+nd]
    beta_t0 = theta[nv+nd:nv+nd+nt0]
    return beta_v, beta_d, beta_t0


def firstpass_logpdf(t: np.ndarray, d: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Log density of first passage time to upper boundary d for BM drift v, noise=1."""
    t = np.asarray(t, dtype=float)
    d = np.asarray(d, dtype=float)
    v = np.asarray(v, dtype=float)
    out = np.full_like(t, -1e6, dtype=float)
    ok = (t > 1e-6) & (d > 1e-6) & np.isfinite(t) & np.isfinite(d) & np.isfinite(v)
    if np.any(ok):
        tt = t[ok]
        dd = d[ok]
        vv = v[ok]
        out[ok] = np.log(dd) - 0.5*np.log(2*np.pi) - 1.5*np.log(tt) - ((dd - vv*tt)**2)/(2*tt)
    return out


def firstpass_cdf(T: np.ndarray, d: np.ndarray, v: np.ndarray) -> np.ndarray:
    """CDF P(hit upper boundary before T), BM drift v, noise=1, boundary d."""
    T = np.asarray(T, dtype=float)
    d = np.asarray(d, dtype=float)
    v = np.asarray(v, dtype=float)
    F = np.zeros_like(T, dtype=float)
    ok = (T > 1e-6) & (d > 1e-6) & np.isfinite(T) & np.isfinite(d) & np.isfinite(v)
    if np.any(ok):
        TT = T[ok]
        dd = d[ok]
        vv = v[ok]
        sqrtT = np.sqrt(TT)
        term1 = norm.cdf((vv*TT - dd)/sqrtT)
        expo = np.exp(np.clip(2*vv*dd, -50, 50))
        term2 = expo * norm.cdf(-(vv*TT + dd)/sqrtT)
        Fok = term1 + term2
        F[ok] = np.clip(Fok, EPS, 1-EPS)
    return np.clip(F, EPS, 1-EPS)


def predict_components(theta: np.ndarray, Xv: np.ndarray, Xd: np.ndarray, Xt0: np.ndarray,
                       t0_min: float, t0_max: float, dist_min: float):
    nv, nd, nt0 = Xv.shape[1], Xd.shape[1], Xt0.shape[1]
    bv, bd, bt0 = unpack_params(theta, nv, nd, nt0)
    v = Xv @ bv
    d = softplus(Xd @ bd) + dist_min
    t0 = t0_min + (t0_max - t0_min) * expit(Xt0 @ bt0)
    return v, d, t0


def neg_loglik(theta: np.ndarray, Xv: np.ndarray, Xd: np.ndarray, Xt0: np.ndarray,
               press: np.ndarray, rt: np.ndarray, deadline: np.ndarray,
               t0_min: float, t0_max: float, dist_min: float, l2: float) -> float:
    v, d, t0 = predict_components(theta, Xv, Xd, Xt0, t0_min, t0_max, dist_min)

    press = press.astype(bool)
    ll = np.zeros(len(press), dtype=float)

    # Press RT likelihood.
    if np.any(press):
        td = rt[press] - t0[press]
        ll[press] = firstpass_logpdf(td, d[press], v[press])

    # No response: survival up to deadline.
    noresp = ~press
    if np.any(noresp):
        T = deadline[noresp] - t0[noresp]
        F = firstpass_cdf(T, d[noresp], v[noresp])
        ll[noresp] = np.log(np.clip(1 - F, EPS, 1))

    if not np.all(np.isfinite(ll)):
        return 1e30

    # Apply a small ridge penalty to stabilize numerical optimization.
    penalty = l2 * float(np.sum(theta**2))
    nll = -float(np.sum(ll)) + penalty
    if not np.isfinite(nll):
        return 1e30
    return nll


def fit_one_model(train_df: pd.DataFrame, variant: DDMVariant, deadline_ms: float,
                  n_starts: int, seed: int, l2: float,
                  t0_min: float, t0_max: float, dist_min: float) -> Dict:
    Xv, Xd, Xt0, design_meta = prepare_designs(train_df, train_df, variant)
    press = train_df["press"].to_numpy(dtype=int)
    rt = train_df["rt_sec"].to_numpy(dtype=float)
    deadline = train_df["deadline_sec"].to_numpy(dtype=float)

    npar = Xv.shape[1] + Xd.shape[1] + Xt0.shape[1]
    rng = np.random.default_rng(seed)

    starts = []
    base = np.zeros(npar)
    # Drift intercept positive-ish, distance intercept mild, t0 maps mid.
    if Xv.shape[1] > 0:
        base[0] = 0.8
    if Xd.shape[1] > 0:
        base[Xv.shape[1]] = 0.0
    starts.append(base.copy())

    for _ in range(max(0, n_starts-1)):
        starts.append(base + rng.normal(0, 0.25, size=npar))

    bounds = [(-6, 6)] * npar
    best = None
    history = []
    for i, x0 in enumerate(starts):
        res = minimize(
            neg_loglik, x0=x0,
            args=(Xv, Xd, Xt0, press, rt, deadline, t0_min, t0_max, dist_min, l2),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-8, "maxls": 40},
        )
        history.append({"start": i, "success": bool(res.success), "fun": float(res.fun), "message": str(res.message)})
        if best is None or res.fun < best.fun:
            best = res

    assert best is not None
    nll = float(best.fun)
    k = int(npar)
    n = int(len(train_df))
    aic = 2*k + 2*nll
    bic = k*np.log(n) + 2*nll

    return {
        "variant": variant.key,
        "variant_name": variant.name,
        "success": bool(best.success),
        "message": str(best.message),
        "nll": nll,
        "loglik": -nll,
        "n_params": k,
        "n_trials": n,
        "aic": float(aic),
        "bic": float(bic),
        "theta": best.x.tolist(),
        "design": design_meta,
        "optimizer_history": history,
        "settings": {
            "deadline_ms": deadline_ms,
            "n_starts": n_starts,
            "seed": seed,
            "l2": l2,
            "t0_min": t0_min,
            "t0_max": t0_max,
            "dist_min": dist_min,
        }
    }


def score_model(test_df: pd.DataFrame, train_df: pd.DataFrame, fit: Dict,
                t0_min: float, t0_max: float, dist_min: float, l2: float = 0.0) -> Dict:
    variant = VARIANTS[fit["variant"]]
    Xv, Xd, Xt0, _ = prepare_designs(train_df, test_df, variant)
    theta = np.asarray(fit["theta"], dtype=float)
    press = test_df["press"].to_numpy(dtype=int)
    rt = test_df["rt_sec"].to_numpy(dtype=float)
    deadline = test_df["deadline_sec"].to_numpy(dtype=float)

    nll = neg_loglik(theta, Xv, Xd, Xt0, press, rt, deadline, t0_min, t0_max, dist_min, l2=0.0)
    ppress = predict_press_probability(theta, Xv, Xd, Xt0, deadline, t0_min, t0_max, dist_min)
    ppc = ppc_metrics(test_df, ppress)
    return {"elpd": -float(nll), "nll": float(nll), "ppc": ppc}


def predict_press_probability(theta: np.ndarray, Xv: np.ndarray, Xd: np.ndarray, Xt0: np.ndarray,
                              deadline: np.ndarray, t0_min: float, t0_max: float, dist_min: float) -> np.ndarray:
    v, d, t0 = predict_components(theta, Xv, Xd, Xt0, t0_min, t0_max, dist_min)
    T = deadline - t0
    return firstpass_cdf(T, d, v)


def ppc_metrics(df: pd.DataFrame, ppress: np.ndarray) -> Dict:
    out = {}
    press = df["press"].to_numpy(dtype=float)
    nogo = df["is_nogo"].to_numpy(dtype=bool)
    go = ~nogo
    out["observed_press"] = float(np.mean(press))
    out["predicted_press"] = float(np.mean(ppress))
    out["observed_hit"] = float(np.mean(press[go])) if np.any(go) else None
    out["predicted_hit"] = float(np.mean(ppress[go])) if np.any(go) else None
    out["observed_fa"] = float(np.mean(press[nogo])) if np.any(nogo) else None
    out["predicted_fa"] = float(np.mean(ppress[nogo])) if np.any(nogo) else None
    out["by_group"] = {}
    for gval, gname in [(0, "low"), (1, "high")]:
        m = df["group_high"].to_numpy(dtype=int) == gval
        if not np.any(m):
            continue
        out["by_group"][gname] = {
            "observed_press": float(np.mean(press[m])),
            "predicted_press": float(np.mean(ppress[m])),
            "observed_fa": float(np.mean(press[m & nogo])) if np.any(m & nogo) else None,
            "predicted_fa": float(np.mean(ppress[m & nogo])) if np.any(m & nogo) else None,
            "observed_hit": float(np.mean(press[m & go])) if np.any(m & go) else None,
            "predicted_hit": float(np.mean(ppress[m & go])) if np.any(m & go) else None,
        }
    out["by_cue"] = {}
    for cue in ["down", "bi", "up"]:
        m = df["cue_label_ddm"].to_numpy() == cue
        if np.any(m):
            out["by_cue"][cue] = {
                "n": int(np.sum(m)),
                "observed_press": float(np.mean(press[m])),
                "predicted_press": float(np.mean(ppress[m])),
                "actual_nogo": float(np.mean(df.loc[m, "is_nogo"])),
            }
    return out


# =============================================================================
# Commands
# =============================================================================

def fit_suite(args) -> None:
    d_raw = load_latents(Path(args.latents), deadline_ms=args.deadline_ms, data_root=Path(args.data_root) if args.data_root else None)
    d, filter_info = filter_ddm_trials(d_raw, rt_min_ms=args.rt_min_ms, rt_max_ms=args.rt_max_ms)
    out = Path(args.out)
    ensure_dir(out)

    write_json(out/"input_qc.json", {
        "time": now_str(),
        "latents": str(args.latents),
        "n_trials_raw": int(len(d_raw)),
        "n_trials": int(len(d)),
        "n_subjects": int(d["subj_id_ddm"].nunique()),
        "deadline_ms": args.deadline_ms,
        "filter": filter_info,
        "variants": args.variants,
    })

    rows = []
    for vk in args.variants:
        if vk not in VARIANTS:
            print(f"[skip] Unknown variant {vk}")
            continue
        variant = VARIANTS[vk]
        print("="*90)
        print(f"[fit] {vk}: {variant.name}")
        fit = fit_one_model(
            d, variant,
            deadline_ms=args.deadline_ms,
            n_starts=args.n_starts,
            seed=args.seed,
            l2=args.l2,
            t0_min=args.t0_min,
            t0_max=args.t0_max,
            dist_min=args.dist_min,
        )
        model_dir = out/"models"/vk
        ensure_dir(model_dir)
        write_json(model_dir/"fit.json", fit)

        # PPC full data
        Xv, Xd, Xt0, _ = prepare_designs(d, d, variant)
        ppress = predict_press_probability(
            np.asarray(fit["theta"], dtype=float), Xv, Xd, Xt0,
            d["deadline_sec"].to_numpy(dtype=float),
            args.t0_min, args.t0_max, args.dist_min
        )
        ppc = ppc_metrics(d, ppress)
        write_json(model_dir/"ppc.json", ppc)

        rows.append({
            "variant": vk,
            "variant_name": variant.name,
            "loglik": fit["loglik"],
            "nll": fit["nll"],
            "aic": fit["aic"],
            "bic": fit["bic"],
            "n_params": fit["n_params"],
            "success": fit["success"],
            "observed_fa": ppc.get("observed_fa"),
            "predicted_fa": ppc.get("predicted_fa"),
            "observed_hit": ppc.get("observed_hit"),
            "predicted_hit": ppc.get("predicted_hit"),
        })

    comp = pd.DataFrame(rows).sort_values("aic")
    comp.to_csv(out/"fit_comparison.csv", index=False)
    print(comp.to_string(index=False))


def cv(args) -> None:
    d_raw = load_latents(Path(args.latents), deadline_ms=args.deadline_ms, data_root=Path(args.data_root) if args.data_root else None)
    d, filter_info = filter_ddm_trials(d_raw, rt_min_ms=args.rt_min_ms, rt_max_ms=args.rt_max_ms)
    out = Path(args.out)
    ensure_dir(out)
    write_json(out/"cv_input_qc.json", {
        "time": now_str(),
        "latents": str(args.latents),
        "n_trials_raw": int(len(d_raw)),
        "n_trials": int(len(d)),
        "n_subjects": int(d["subj_id_ddm"].nunique()),
        "deadline_ms": args.deadline_ms,
        "filter": filter_info,
        "variants": args.variants,
        "k": args.k,
        "seed": args.seed,
    })

    subj = np.array(sorted(d["subj_id_ddm"].unique()))
    # Stratified folds by group
    subj_group = d.groupby("subj_id_ddm")["group_high"].first()
    low = subj_group[subj_group == 0].index.to_numpy()
    high = subj_group[subj_group == 1].index.to_numpy()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(low)
    rng.shuffle(high)
    folds = [[] for _ in range(args.k)]
    for i, s in enumerate(low):
        folds[i % args.k].append(s)
    for i, s in enumerate(high):
        folds[i % args.k].append(s)
    folds = [sorted(f) for f in folds]

    all_rows = []
    for vk in args.variants:
        if vk not in VARIANTS:
            continue
        variant = VARIANTS[vk]
        print("="*90)
        print(f"[cv] {vk}: {variant.name}")
        model_dir = out/"cv"/vk
        ensure_dir(model_dir)
        fold_results = []

        for fi, test_subs in enumerate(folds):
            train_subs = [s for j, fold in enumerate(folds) if j != fi for s in fold]
            train = d[d["subj_id_ddm"].isin(train_subs)].reset_index(drop=True)
            test = d[d["subj_id_ddm"].isin(test_subs)].reset_index(drop=True)

            print(f"  fold {fi+1}/{args.k}: train_subj={len(train_subs)} test_subj={len(test_subs)}")
            fit = fit_one_model(
                train, variant,
                deadline_ms=args.deadline_ms,
                n_starts=args.n_starts,
                seed=args.seed + 1000*fi,
                l2=args.l2,
                t0_min=args.t0_min,
                t0_max=args.t0_max,
                dist_min=args.dist_min,
            )
            sc = score_model(test, train, fit, args.t0_min, args.t0_max, args.dist_min)
            fr = {
                "fold": fi,
                "train_subjects": list(map(str, train_subs)),
                "test_subjects": list(map(str, test_subs)),
                "fit": fit,
                "elpd": sc["elpd"],
                "nll": sc["nll"],
                "ppc": sc["ppc"],
            }
            fold_results.append(fr)
            write_json(model_dir/f"fold_{fi:02d}.json", fr)

        total_elpd = float(sum(fr["elpd"] for fr in fold_results))
        out_json = {
            "time": now_str(),
            "variant": vk,
            "variant_name": variant.name,
            "k": args.k,
            "seed": args.seed,
            "total_elpd": total_elpd,
            "folds": [
                {
                    "fold": fr["fold"],
                    "elpd": fr["elpd"],
                    "nll": fr["nll"],
                    "test_subjects": fr["test_subjects"],
                    "ppc": fr["ppc"],
                }
                for fr in fold_results
            ]
        }
        write_json(model_dir/"cv_kfold_subject.json", out_json)
        all_rows.append({
            "variant": vk,
            "variant_name": variant.name,
            "total_elpd": total_elpd,
            "mean_fold_elpd": float(np.mean([fr["elpd"] for fr in fold_results])),
            "sd_fold_elpd": float(np.std([fr["elpd"] for fr in fold_results], ddof=1)),
        })

    comp = pd.DataFrame(all_rows).sort_values("total_elpd", ascending=False)
    comp.to_csv(out/"cv_comparison.csv", index=False)
    print(comp.to_string(index=False))


def compare(args) -> None:
    out = Path(args.out)
    rows = []
    for p in sorted((out/"cv").glob("*/cv_kfold_subject.json")):
        obj = read_json(p)
        rows.append({
            "variant": obj["variant"],
            "variant_name": obj["variant_name"],
            "total_elpd": obj["total_elpd"],
            "k": obj["k"],
            "seed": obj["seed"],
        })
    if rows:
        df = pd.DataFrame(rows).sort_values("total_elpd", ascending=False)
        df.to_csv(out/"cv_comparison.csv", index=False)
        print(df.to_string(index=False))

    fitp = out/"fit_comparison.csv"
    if fitp.exists():
        print("\nFull-data fit comparison:")
        print(pd.read_csv(fitp).to_string(index=False))


def export_predictions(args) -> None:
    d_raw = load_latents(Path(args.latents), deadline_ms=args.deadline_ms, data_root=Path(args.data_root) if args.data_root else None)
    d, filter_info = filter_ddm_trials(d_raw, rt_min_ms=args.rt_min_ms, rt_max_ms=args.rt_max_ms)
    out = Path(args.out)
    model_dir = out/"models"/args.variant
    fit = read_json(model_dir/"fit.json")
    if fit is None:
        raise FileNotFoundError(f"Missing fit.json for {args.variant}: {model_dir}")

    variant = VARIANTS[args.variant]
    Xv, Xd, Xt0, _ = prepare_designs(d, d, variant)
    theta = np.asarray(fit["theta"], dtype=float)
    ppress = predict_press_probability(
        theta, Xv, Xd, Xt0,
        d["deadline_sec"].to_numpy(dtype=float),
        args.t0_min, args.t0_max, args.dist_min
    )
    v, dist, t0 = predict_components(theta, Xv, Xd, Xt0, args.t0_min, args.t0_max, args.dist_min)

    outdf = d.copy()
    outdf["ddm_p_press"] = ppress
    outdf["ddm_drift"] = v
    outdf["ddm_distance_to_bound"] = dist
    outdf["ddm_t0"] = t0
    keep = [
        "subj_id_ddm", "group_high", "trial_index", "cue_label_ddm", "is_nogo", "press", "rt_sec",
        "fa", "hit", "miss", "cr",
        "control_current", "actor_qdiff_current", "belief_nogo_current", "associability_current",
        "p_press", "cnv_z", "p300_z", "ern_strength_z", "pe_strength_z",
        "ddm_p_press", "ddm_drift", "ddm_distance_to_bound", "ddm_t0"
    ]
    keep = [c for c in keep if c in outdf.columns]
    outp = out/f"trial_ddm_predictions_{args.variant}.csv"
    outdf[keep].to_csv(outp, index=False)
    print(f"[export] wrote {outp}")


def build_argparser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--latents", required=True, help="RL latent CSV for DDM, e.g., outputs/ddm_latents/trial_latents_for_ddm_M6_seed2026.csv")
    common.add_argument("--data_root", default=None, help="Raw xlsx root for merging RT/response columns when needed.")
    common.add_argument("--deadline_ms", type=float, default=700.0)
    common.add_argument("--rt_min_ms", type=float, default=100.0, help="Exclude press trials with RT below this threshold; no-response trials are kept.")
    common.add_argument("--rt_max_ms", type=float, default=1000.0, help="Exclude press trials with RT above this threshold; no-response trials are kept.")
    common.add_argument("--t0_min", type=float, default=0.03)
    common.add_argument("--t0_max", type=float, default=0.08)
    common.add_argument("--dist_min", type=float, default=0.05)

    p = sub.add_parser("inspect", parents=[common])
    p.set_defaults(func=lambda args: inspect(Path(args.latents), args.deadline_ms, Path(args.data_root) if args.data_root else None, args.rt_min_ms, args.rt_max_ms))

    fit = sub.add_parser("fit_suite", parents=[common])
    fit.add_argument("--out", required=True)
    fit.add_argument("--variants", nargs="+", default=["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D6A", "D7", "D8"])
    fit.add_argument("--n_starts", type=int, default=5)
    fit.add_argument("--seed", type=int, default=2026)
    fit.add_argument("--l2", type=float, default=0.01)
    fit.set_defaults(func=fit_suite)

    cvp = sub.add_parser("cv", parents=[common])
    cvp.add_argument("--out", required=True)
    cvp.add_argument("--variants", nargs="+", default=["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D6A", "D7", "D8"])
    cvp.add_argument("--k", type=int, default=5)
    cvp.add_argument("--n_starts", type=int, default=3)
    cvp.add_argument("--seed", type=int, default=2026)
    cvp.add_argument("--l2", type=float, default=0.01)
    cvp.set_defaults(func=cv)

    comp = sub.add_parser("compare")
    comp.add_argument("--out", required=True)
    comp.set_defaults(func=compare)

    exp = sub.add_parser("export_predictions", parents=[common])
    exp.add_argument("--out", required=True)
    exp.add_argument("--variant", default="D2")
    exp.set_defaults(func=export_predictions)

    return ap


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = build_argparser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
