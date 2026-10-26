#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reinforcement-learning core for the Cued Go/NoGo behavioral analysis.

The module implements a family of adaptive-learning models (M0-M7), including
cue-belief, post-error control, volatility/associability, asymmetric learning,
and actor-control formulations. The primary downstream model in the companion
analyses is M6.

The module supports preprocessing, Bayesian model fitting, participant-level
cross-validation, posterior summaries, posterior-predictive evaluation, and
trial-level latent-state export for downstream action and diffusion-model
analyses. ERP measurements are not incorporated into the trial-wise
reinforcement-learning update; neural variables are analyzed separately in the
EEG companion analysis."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
import warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import pytensor
import pytensor.tensor as pt
import xarray as xr

pm = None
az = None


def require_bayes():
    global pm, az
    if pm is not None and az is not None:
        return pm, az
    import pymc as _pm
    import arviz as _az
    pm, az = _pm, _az
    return pm, az

warnings.filterwarnings("ignore", category=FutureWarning, module="arviz")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="arviz")

# =============================================================================
# Utilities
# =============================================================================


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(p: Path, default=None):
    return default if not p.exists() else json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_npz(p: Path, arrays: Dict[str, np.ndarray]) -> None:
    ensure_dir(p.parent)
    np.savez_compressed(p, **arrays)


def load_npz(p: Path) -> Dict[str, np.ndarray]:
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def get_meta(arr: Dict[str, np.ndarray]) -> Dict[str, Any]:
    return json.loads(arr["_meta_json"].item())


def set_meta(arr: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
    arr["_meta_json"] = np.array([json.dumps(meta)], dtype=object)


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def logsumexp_np(a: np.ndarray, axis: int = 0) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return float(default)


def default_cores() -> int:
    return max(1, min(8, os.cpu_count() or 4))

# =============================================================================
# Variants
# =============================================================================


@dataclass(frozen=True)
class VariantSpec:
    name: str
    use_belief: bool = False
    use_control: bool = False
    use_volatility: bool = False
    use_asym_belief: bool = False
    use_actor: bool = False
    include_prev_error: bool = True


VARIANTS: Dict[str, VariantSpec] = {
    "M0": VariantSpec("baseline_cue_target_prevError"),
    "M1": VariantSpec("cue_belief_learning", use_belief=True),
    "M2": VariantSpec("post_error_control", use_control=True),
    "M3": VariantSpec("belief_plus_control", use_belief=True, use_control=True),
    "M4": VariantSpec("volatility_belief_learning", use_belief=True, use_volatility=True),
    "M5": VariantSpec("asymmetric_belief_learning", use_belief=True, use_asym_belief=True),
    "M6": VariantSpec("actor_control_rl", use_actor=True, use_control=True),
    "M7": VariantSpec("actor_control_plus_volatility", use_actor=True, use_control=True, use_belief=True, use_volatility=True),
}

DEFAULT_LADDER = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7"]

# =============================================================================
# Preprocessing
# =============================================================================


@dataclass
class SubjectRow:
    subj_id: str
    group: int
    path: str


def infer_group_from_path(p: Path) -> int:
    parts = [x.lower() for x in p.parts]
    name = p.name.lower()
    stem = p.stem.lower()
    # Match explicit folder and file names rather than generic parent-directory names.
    if any(x in parts for x in ["high", "highrisk", "high_risk", "csv_high", "h"]):
        return 1
    if any(x in parts for x in ["low", "lowrisk", "low_risk", "csv_low", "l"]):
        return 0
    if re.match(r"^h\d+", stem) or "high" in name:
        return 1
    if re.match(r"^l\d+", stem) or "low" in name:
        return 0
    raise ValueError(f"Cannot infer group from file path: {p}")


def discover_subject_files(data_root: Path) -> List[SubjectRow]:
    paths = sorted(data_root.rglob("*_metadata_features.xlsx"))
    paths = [p for p in paths if "all_" not in p.name.lower() and "merged" not in p.name.lower()]
    if not paths:
        raise FileNotFoundError(f"No individual '*_metadata_features.xlsx' files found under {data_root}")
    rows = []
    for fp in paths:
        subj_id = fp.stem.split("_metadata_features")[0]
        rows.append(SubjectRow(subj_id=subj_id, group=infer_group_from_path(fp), path=str(fp)))

    def key(r: SubjectRow):
        m = re.search(r"(\d+)", r.subj_id)
        n = int(m.group(1)) if m else 10**9
        return (r.group, n, r.subj_id)

    rows.sort(key=key)
    groups = [r.group for r in rows]
    if len(set(groups)) < 2:
        raise ValueError(f"Only one group detected: {set(groups)}. Check folder names.")
    return rows


def exact_or_best(df: pd.DataFrame, exact: str, must: List[str], prefer: List[str]) -> Optional[str]:
    if exact in df.columns:
        return exact
    cols = df.columns.tolist()
    cands = []
    for c in cols:
        low = re.sub(r"[^a-z0-9_]+", "_", c.lower())
        if all(m.lower() in low for m in must):
            score = sum(5 for p in prefer if p.lower() in low) + max(0, 80 - len(low))/80
            cands.append((score, c))
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][1]


def pick_core_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "cnv": exact_or_best(df, "cue_cnv_mean_cz_200_500", ["cnv"], ["cue", "cz", "mean", "200", "500"]),
        "p300": exact_or_best(df, "target_p300_max_cz_250_500", ["p300"], ["target", "cz", "max", "250", "500"]),
        "ern": exact_or_best(df, "resp_ern_min_fz_50_150", ["ern"], ["resp", "fz", "min", "50", "150"]),
        "pe": exact_or_best(df, "resp_pe_max_fz_200_400", ["pe"], ["resp", "fz", "max", "200", "400"]),
    }


def col_float(df: pd.DataFrame, name: Optional[str]) -> np.ndarray:
    if not name or name not in df.columns:
        return np.full(len(df), np.nan)
    a = pd.to_numeric(df[name], errors="coerce").to_numpy(float)
    a[~np.isfinite(a)] = np.nan
    return a


def zscore_subject_vector(x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    mask = np.ones_like(x, dtype=bool) if mask is None else mask.astype(bool)
    m = mask & np.isfinite(x)
    if np.sum(m) < 2:
        return y
    mu = np.nanmean(x[m])
    sd = np.nanstd(x[m])
    if not np.isfinite(sd) or sd < 1e-8:
        return y
    y[m] = (x[m] - mu) / sd
    y[~np.isfinite(y)] = 0.0
    return y


def make_arrays(rows: List[SubjectRow]) -> Dict[str, np.ndarray]:
    all_dfs = []
    picked = None
    trial_counts = []
    for i, r in enumerate(rows):
        df = pd.read_excel(r.path).copy()
        if picked is None:
            picked = pick_core_columns(df)
        df = df.sort_values("trial_index").reset_index(drop=True)
        df["__subj"] = i
        df["__group"] = int(r.group)
        df["__subj_id"] = str(r.subj_id)
        all_dfs.append(df)
        trial_counts.append(len(df))
    if len(set(trial_counts)) != 1:
        raise ValueError(f"Subjects have unequal number of trials: {trial_counts}")
    T = trial_counts[0]
    S = len(rows)
    d = pd.concat(all_dfs, ignore_index=True)

    req = ["trial_index", "trial_class", "cue_type", "response_present", "response_type"]
    miss = [c for c in req if c not in d.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    # cue levels with stable task ordering
    cue_all = d["cue_type"].astype(str).str.lower().to_numpy()
    canonical = ["down", "bi", "up"]
    observed = list(pd.unique(cue_all))
    cue_levels = [c for c in canonical if c in observed] + sorted([c for c in observed if c not in canonical])
    cue_to_idx = {c: i for i, c in enumerate(cue_levels)}
    C = len(cue_levels)

    shape = (S, T)
    cue_idx = np.zeros(shape, dtype=np.int16)
    is_nogo = np.zeros(shape, dtype=np.int8)
    action = np.zeros(shape, dtype=np.int8)
    error = np.zeros(shape, dtype=np.int8)
    correct = np.zeros(shape, dtype=np.int8)
    fa = np.zeros(shape, dtype=np.int8)
    hit = np.zeros(shape, dtype=np.int8)
    miss_arr = np.zeros(shape, dtype=np.int8)
    cr = np.zeros(shape, dtype=np.int8)
    prev_error = np.zeros(shape, dtype=np.int8)
    logrt = np.zeros(shape, dtype=float)
    rt_obs = np.zeros(shape, dtype=np.int8)
    cnv_z = np.zeros(shape, dtype=float)
    p300_z = np.zeros(shape, dtype=float)
    ern_strength_z = np.zeros(shape, dtype=float)
    pe_strength_z = np.zeros(shape, dtype=float)

    subj_ids = []
    groups = np.array([r.group for r in rows], dtype=np.int8)
    for s, (r, df) in enumerate(zip(rows, all_dfs)):
        subj_ids.append(r.subj_id)
        trial_class = df["trial_class"].astype(str).str.lower().to_numpy()
        is_ng = (trial_class == "nogo").astype(np.int8)
        cue = df["cue_type"].astype(str).str.lower().to_numpy()
        ci = np.array([cue_to_idx[c] for c in cue], dtype=np.int16)
        act = pd.to_numeric(df["response_present"], errors="coerce").fillna(0).astype(int).clip(0, 1).to_numpy(np.int8)
        resp = df["response_type"].astype(str).str.lower().to_numpy()
        corr = (((is_ng == 0) & (act == 1)) | ((is_ng == 1) & (act == 0))).astype(np.int8)
        err = (1 - corr).astype(np.int8)
        cue_idx[s] = ci
        is_nogo[s] = is_ng
        action[s] = act
        correct[s] = corr
        error[s] = err
        fa[s] = (resp == "fa").astype(np.int8)
        hit[s] = (resp == "hit").astype(np.int8)
        miss_arr[s] = (resp == "miss").astype(np.int8)
        cr[s] = (resp == "cr").astype(np.int8)
        prev_error[s, 1:] = err[:-1]

        if "rt_response_from_target_ms" in df.columns:
            rt = pd.to_numeric(df["rt_response_from_target_ms"], errors="coerce").to_numpy(float)
            m = np.isfinite(rt) & (act == 1)
            rt_obs[s, m] = 1
            logrt[s, m] = np.log(np.maximum(rt[m], 50.0) / 1000.0)

        cnv_raw = col_float(df, picked["cnv"])
        p300_raw = col_float(df, picked["p300"])
        ern_raw = col_float(df, picked["ern"])
        pe_raw = col_float(df, picked["pe"])
        cnv_z[s] = zscore_subject_vector(cnv_raw)
        p300_z[s] = zscore_subject_vector(p300_raw)
        fa_mask = fa[s].astype(bool) & np.isfinite(ern_raw)
        pe_mask = fa[s].astype(bool) & np.isfinite(pe_raw)
        ern_z = zscore_subject_vector(ern_raw, mask=fa_mask)
        ern_strength_z[s] = -ern_z
        ern_strength_z[s, ~fa_mask] = 0.0
        pe_strength_z[s] = zscore_subject_vector(pe_raw, mask=pe_mask)
        pe_strength_z[s, ~pe_mask] = 0.0

    arr = dict(
        group=groups,
        cue_idx=cue_idx,
        is_nogo=is_nogo,
        action=action,
        correct=correct,
        error=error,
        prev_error=prev_error,
        fa=fa,
        hit=hit,
        miss=miss_arr,
        cr=cr,
        rt_obs=rt_obs,
        logrt=logrt,
        cnv_z=cnv_z,
        p300_z=p300_z,
        ern_strength_z=ern_strength_z,
        pe_strength_z=pe_strength_z,
    )
    meta = dict(
        created_at=now_str(),
        n_subj=int(S),
        n_trials_per_subj=int(T),
        n_trials=int(S*T),
        cue_levels=cue_levels,
        subj_ids=subj_ids,
        core_cols=picked,
        notes="ERN strength is -z(resp_ern_min_fz_50_150) within subject FA trials.",
    )
    set_meta(arr, meta)
    return arr


def preprocess(data_root: Path, out_dir: Path, rebuild: bool = False) -> None:
    ensure_dir(out_dir / "artifacts")
    rows = discover_subject_files(data_root)
    pd.DataFrame([asdict(r) for r in rows]).to_csv(out_dir / "artifacts" / "manifest.csv", index=False)
    core = out_dir / "artifacts" / "arrays__actor_control_model_family.npz"
    if rebuild or not core.exists():
        arr = make_arrays(rows)
        save_npz(core, arr)
        print("[preprocess] wrote", core)
    else:
        print("[preprocess] exists; skipping. Use --rebuild to overwrite.")
    doctor(out_dir)


def load_artifacts(out_dir: Path) -> Dict[str, np.ndarray]:
    p = out_dir / "artifacts" / "arrays__actor_control_model_family.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}; run preprocess.")
    return load_npz(p)


def doctor(out_dir: Path) -> None:
    arr = load_artifacts(out_dir)
    meta = get_meta(arr)
    g = arr["group"]
    qc = {
        "time": now_str(),
        "n_subjects": int(meta["n_subj"]),
        "n_trials": int(meta["n_trials"]),
        "group_counts": {"low": int(np.sum(g == 0)), "high": int(np.sum(g == 1))},
        "cue_levels": meta["cue_levels"],
        "core_cols": meta["core_cols"],
        "trial_counts": {
            "go": int(np.sum(arr["is_nogo"] == 0)),
            "nogo": int(np.sum(arr["is_nogo"] == 1)),
            "hit": int(np.sum(arr["hit"] == 1)),
            "fa": int(np.sum(arr["fa"] == 1)),
            "miss": int(np.sum(arr["miss"] == 1)),
            "cr": int(np.sum(arr["cr"] == 1)),
        },
        "erp_counts": {
            "ern_fa": int(np.sum(arr["fa"] == 1)),
            "pe_fa": int(np.sum(arr["fa"] == 1)),
        },
    }
    write_json(out_dir / "artifacts" / "qc.json", qc)
    print(json.dumps(qc, indent=2))

# =============================================================================
# PyMC model
# =============================================================================


def hier_normal(name: str, group_vec, sigma_mu=1.0, sigma_delta=0.5, sigma_sd=0.7):
    mu = pm.Normal(f"mu_{name}", 0, sigma_mu)
    delta = pm.Normal(f"delta_{name}_group", 0, sigma_delta)
    sigma = pm.HalfNormal(f"sigma_{name}", sigma_sd)
    raw = pm.Normal(f"{name}_raw", 0, 1, shape=group_vec.shape[0])
    lat = pm.Deterministic(f"{name}_lat", mu + delta * group_vec + sigma * raw)
    return lat


def zero_vec(n):
    return pt.zeros((n,), dtype="float64")


def build_model(arr: Dict[str, np.ndarray], variant: VariantSpec):
    global pm
    pm, _az = require_bayes()
    meta = get_meta(arr)
    S = int(meta["n_subj"])
    T = int(meta["n_trials_per_subj"])
    C = len(meta["cue_levels"])
    group = arr["group"].astype(float)

    # Data as T x S for scan.
    # PyTensor constants are required here rather than NumPy arrays.
    # Raw NumPy arrays are not used because pytensor.scan internally
    # slices sequences with a symbolic n_steps TensorVariable.
    cue_TS = pt.as_tensor_variable(arr["cue_idx"].T.astype("int32"))
    nogo_TS = pt.as_tensor_variable(arr["is_nogo"].T.astype("float64"))
    action_TS = pt.as_tensor_variable(arr["action"].T.astype("float64"))
    error_TS = pt.as_tensor_variable(arr["error"].T.astype("float64"))
    prev_error_TS = pt.as_tensor_variable(arr["prev_error"].T.astype("float64"))
    correct_TS = pt.as_tensor_variable(arr["correct"].T.astype("float64"))

    with pm.Model() as model:
        g = pm.Data("group", group)

        # Subject-level response bias, non-centered.
        mu_bias = pm.Normal("mu_bias", 0, 1.0)
        delta_bias_group = pm.Normal("delta_bias_group", 0, 0.6)
        sigma_bias = pm.HalfNormal("sigma_bias", 1.0)
        bias_raw = pm.Normal("bias_raw", 0, 1, shape=S)
        bias = pm.Deterministic("bias", mu_bias + delta_bias_group * g + sigma_bias * bias_raw)

        # Fixed effects.
        b_cue_raw = pm.Normal("b_cue_raw", 0, 0.5, shape=C)
        b_cue = pm.Deterministic("b_cue", b_cue_raw - pt.mean(b_cue_raw))
        b_nogo = pm.Deterministic("b_nogo", -pm.HalfNormal("b_nogo_abs", 1.0))
        if variant.include_prev_error:
            b_prev_error = pm.Normal("b_prev_error", 0, 0.7)
        else:
            b_prev_error = pm.Deterministic("b_prev_error", pt.as_tensor_variable(0.0))
        lapse_eps = pm.Beta("lapse_eps", 1.0, 12.0)

        # Belief parameters.
        if variant.use_belief and not variant.use_volatility and not variant.use_asym_belief:
            alpha_belief_lat = hier_normal("alpha_belief", g)
            alpha_belief = pm.Deterministic("alpha_belief", pm.math.sigmoid(alpha_belief_lat))
        else:
            alpha_belief = pm.Deterministic("alpha_belief", zero_vec(S))

        if variant.use_asym_belief:
            alpha_pos_lat = hier_normal("alpha_pos", g)
            alpha_neg_lat = hier_normal("alpha_neg", g)
            alpha_pos = pm.Deterministic("alpha_pos", pm.math.sigmoid(alpha_pos_lat))
            alpha_neg = pm.Deterministic("alpha_neg", pm.math.sigmoid(alpha_neg_lat))
        else:
            alpha_pos = pm.Deterministic("alpha_pos", zero_vec(S))
            alpha_neg = pm.Deterministic("alpha_neg", zero_vec(S))

        if variant.use_volatility:
            alpha_base_lat = hier_normal("alpha_base", g)
            eta_assoc_lat = hier_normal("eta_assoc", g)
            alpha_base = pm.Deterministic("alpha_base", alpha_base_lat)  # used on logit scale
            eta_assoc = pm.Deterministic("eta_assoc", pm.math.sigmoid(eta_assoc_lat))
            w_assoc = pm.HalfNormal("w_assoc", 1.0)
        else:
            alpha_base = pm.Deterministic("alpha_base", zero_vec(S))
            eta_assoc = pm.Deterministic("eta_assoc", zero_vec(S))
            w_assoc = pm.Deterministic("w_assoc", pt.as_tensor_variable(0.0))

        if variant.use_belief:
            w_belief = pm.Deterministic("w_belief", -pm.HalfNormal("w_belief_abs", 1.0))
        else:
            w_belief = pm.Deterministic("w_belief", pt.as_tensor_variable(0.0))

        # Control parameters.
        if variant.use_control:
            alpha_ctrl_lat = hier_normal("alpha_ctrl", g)
            alpha_ctrl = pm.Deterministic("alpha_ctrl", pm.math.sigmoid(alpha_ctrl_lat))
            w_control = pm.Deterministic("w_control", -pm.HalfNormal("w_control_abs", 1.0))
        else:
            alpha_ctrl = pm.Deterministic("alpha_ctrl", zero_vec(S))
            w_control = pm.Deterministic("w_control", pt.as_tensor_variable(0.0))

        # Actor parameters.
        if variant.use_actor:
            alpha_actor_lat = hier_normal("alpha_actor", g)
            logbeta_actor_lat = hier_normal("logbeta_actor", g, sigma_mu=0.8, sigma_delta=0.4, sigma_sd=0.6)
            alpha_actor = pm.Deterministic("alpha_actor", pm.math.sigmoid(alpha_actor_lat))
            beta_actor = pm.Deterministic("beta_actor", pt.exp(logbeta_actor_lat))
        else:
            alpha_actor = pm.Deterministic("alpha_actor", zero_vec(S))
            beta_actor = pm.Deterministic("beta_actor", zero_vec(S))

        belief0 = pt.full((S, C), 0.5, dtype="float64")
        assoc0 = pt.zeros((S, C), dtype="float64")
        control0 = pt.zeros((S,), dtype="float64")
        q0 = pt.zeros((S, C, 2), dtype="float64")
        cue_arange = pt.arange(C, dtype="int32")

        def step(cue_t, nogo_t, action_t, error_t, prev_error_t, correct_t,
                 belief_prev, assoc_prev, control_prev, q_prev,
                 cue_arange, bias, b_cue, b_nogo, b_prev_error, lapse_eps,
                 alpha_belief, alpha_pos, alpha_neg, alpha_base, eta_assoc, w_assoc,
                 w_belief, alpha_ctrl, w_control, alpha_actor, beta_actor):
            onehot = pt.eq(cue_t[:, None], cue_arange[None, :]).astype("float64")
            belief_cur = pt.sum(belief_prev * onehot, axis=1)
            assoc_cur = pt.sum(assoc_prev * onehot, axis=1)
            q_withhold = pt.sum(q_prev[:, :, 0] * onehot, axis=1)
            q_press = pt.sum(q_prev[:, :, 1] * onehot, axis=1)
            qdiff = q_press - q_withhold

            logit = bias + b_cue[cue_t] + b_nogo * nogo_t + b_prev_error * prev_error_t
            if variant.use_belief:
                logit = logit + w_belief * (belief_cur - 0.5)
            if variant.use_control:
                logit = logit + w_control * control_prev
            if variant.use_actor:
                logit = logit + beta_actor * qdiff

            p = pm.math.sigmoid(logit)
            p = (1.0 - lapse_eps) * p + lapse_eps * 0.5
            logp = action_t * pt.log(pt.clip(p, 1e-12, 1 - 1e-12)) + (1.0 - action_t) * pt.log(pt.clip(1.0 - p, 1e-12, 1 - 1e-12))

            # Belief update.
            belief_new = belief_prev
            assoc_new = assoc_prev
            surprise = pt.abs(nogo_t - belief_cur)
            pe = nogo_t - belief_cur
            if variant.use_belief:
                if variant.use_volatility:
                    alpha_t = pm.math.sigmoid(alpha_base + w_assoc * assoc_cur)
                    assoc_new = assoc_prev + eta_assoc[:, None] * onehot * (surprise - assoc_cur)[:, None]
                    assoc_new = pt.clip(assoc_new, 0.0, 1.0)
                elif variant.use_asym_belief:
                    alpha_t = pt.switch(pt.ge(pe, 0.0), alpha_pos, alpha_neg)
                else:
                    alpha_t = alpha_belief
                belief_new = belief_prev + alpha_t[:, None] * onehot * pe[:, None]
                belief_new = pt.clip(belief_new, 1e-4, 1.0 - 1e-4)

            # Control update.
            control_new = control_prev
            if variant.use_control:
                control_new = control_prev + alpha_ctrl * (error_t - control_prev)
                control_new = pt.clip(control_new, 0.0, 1.0)

            # Actor update.
            q_new = q_prev
            if variant.use_actor:
                q_chosen = action_t * q_press + (1.0 - action_t) * q_withhold
                delta = correct_t - q_chosen
                upd_press = alpha_actor[:, None] * onehot * action_t[:, None] * delta[:, None]
                upd_withhold = alpha_actor[:, None] * onehot * (1.0 - action_t)[:, None] * delta[:, None]
                q0_new = q_prev[:, :, 0] + upd_withhold
                q1_new = q_prev[:, :, 1] + upd_press
                q_new = pt.stack([q0_new, q1_new], axis=2)
                q_new = pt.clip(q_new, -5.0, 5.0)

            return belief_new, assoc_new, control_new, q_new, logp, p, belief_cur, assoc_cur, control_prev, qdiff

        outputs, _ = pytensor.scan(
            fn=step,
            sequences=[cue_TS, nogo_TS, action_TS, error_TS, prev_error_TS, correct_TS],
            outputs_info=[belief0, assoc0, control0, q0, None, None, None, None, None, None],
            non_sequences=[cue_arange, bias, b_cue, b_nogo, b_prev_error, lapse_eps,
                           alpha_belief, alpha_pos, alpha_neg, alpha_base, eta_assoc, w_assoc,
                           w_belief, alpha_ctrl, w_control, alpha_actor, beta_actor],
            strict=True,
        )
        logp_TS = outputs[4]
        loglik_subj = pm.Deterministic("loglik_subj", pt.sum(logp_TS, axis=0))
        pm.Potential("action_loglike", pt.sum(loglik_subj))

        # Lightweight deterministic summaries for posterior inspection.
        pm.Deterministic("mean_p_press", pt.mean(outputs[5]))
        pm.Deterministic("mean_belief_state", pt.mean(outputs[6]))
        pm.Deterministic("mean_assoc_state", pt.mean(outputs[7]))
        pm.Deterministic("mean_control_state", pt.mean(outputs[8]))
        pm.Deterministic("mean_actor_qdiff", pt.mean(outputs[9]))

    return model

# =============================================================================
# Numpy forward model for PPC, CV scoring, and latent export
# =============================================================================


def get_param(params: Dict[str, Any], name: str, default=0.0):
    v = params.get(name, default)
    return np.asarray(v, dtype=float)


def forward_numpy(arr: Dict[str, np.ndarray], variant: VariantSpec, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    meta = get_meta(arr)
    S = int(meta["n_subj"])
    T = int(meta["n_trials_per_subj"])
    C = len(meta["cue_levels"])
    cue = arr["cue_idx"].astype(int)
    nogo = arr["is_nogo"].astype(float)
    action = arr["action"].astype(float)
    error = arr["error"].astype(float)
    correct = arr["correct"].astype(float)
    prev_error = arr["prev_error"].astype(float)

    bias = get_param(params, "bias", np.zeros(S)).reshape(S)
    b_cue = get_param(params, "b_cue", np.zeros(C)).reshape(C)
    b_nogo = float(get_param(params, "b_nogo", 0.0))
    b_prev_error = float(get_param(params, "b_prev_error", 0.0))
    lapse = float(get_param(params, "lapse_eps", 0.0))

    alpha_belief = get_param(params, "alpha_belief", np.zeros(S)).reshape(S)
    alpha_pos = get_param(params, "alpha_pos", np.zeros(S)).reshape(S)
    alpha_neg = get_param(params, "alpha_neg", np.zeros(S)).reshape(S)
    alpha_base = get_param(params, "alpha_base", np.zeros(S)).reshape(S)
    eta_assoc = get_param(params, "eta_assoc", np.zeros(S)).reshape(S)
    w_assoc = float(get_param(params, "w_assoc", 0.0))
    w_belief = float(get_param(params, "w_belief", 0.0))
    alpha_ctrl = get_param(params, "alpha_ctrl", np.zeros(S)).reshape(S)
    w_control = float(get_param(params, "w_control", 0.0))
    alpha_actor = get_param(params, "alpha_actor", np.zeros(S)).reshape(S)
    beta_actor = get_param(params, "beta_actor", np.zeros(S)).reshape(S)

    belief = np.full((S, C), 0.5)
    assoc = np.zeros((S, C))
    control = np.zeros(S)
    Q = np.zeros((S, C, 2))
    p_press = np.zeros((S, T))
    belief_cur_all = np.zeros((S, T))
    assoc_cur_all = np.zeros((S, T))
    control_all = np.zeros((S, T))
    qdiff_all = np.zeros((S, T))
    logp = np.zeros((S, T))

    for t in range(T):
        ci = cue[:, t]
        bcur = belief[np.arange(S), ci]
        acur = assoc[np.arange(S), ci]
        qdiff = Q[np.arange(S), ci, 1] - Q[np.arange(S), ci, 0]
        belief_cur_all[:, t] = bcur
        assoc_cur_all[:, t] = acur
        control_all[:, t] = control
        qdiff_all[:, t] = qdiff

        logit = bias + b_cue[ci] + b_nogo * nogo[:, t] + b_prev_error * prev_error[:, t]
        if variant.use_belief:
            logit += w_belief * (bcur - 0.5)
        if variant.use_control:
            logit += w_control * control
        if variant.use_actor:
            logit += beta_actor * qdiff
        p = sigmoid_np(logit)
        p = (1 - lapse) * p + lapse * 0.5
        p = np.clip(p, 1e-12, 1 - 1e-12)
        p_press[:, t] = p
        logp[:, t] = action[:, t] * np.log(p) + (1 - action[:, t]) * np.log(1 - p)

        if variant.use_belief:
            pe = nogo[:, t] - bcur
            if variant.use_volatility:
                alpha_t = sigmoid_np(alpha_base + w_assoc * acur)
                assoc[np.arange(S), ci] += eta_assoc * (np.abs(pe) - acur)
                assoc = np.clip(assoc, 0, 1)
            elif variant.use_asym_belief:
                alpha_t = np.where(pe >= 0, alpha_pos, alpha_neg)
            else:
                alpha_t = alpha_belief
            belief[np.arange(S), ci] += alpha_t * pe
            belief = np.clip(belief, 1e-4, 1 - 1e-4)

        if variant.use_control:
            control += alpha_ctrl * (error[:, t] - control)
            control = np.clip(control, 0, 1)

        if variant.use_actor:
            chosen_q = np.where(action[:, t] > 0.5, Q[np.arange(S), ci, 1], Q[np.arange(S), ci, 0])
            delta = correct[:, t] - chosen_q
            press = action[:, t] > 0.5
            Q[np.arange(S)[press], ci[press], 1] += alpha_actor[press] * delta[press]
            Q[np.arange(S)[~press], ci[~press], 0] += alpha_actor[~press] * delta[~press]
            Q = np.clip(Q, -5, 5)

    return dict(
        p_press=p_press,
        logp=logp,
        loglik_subj=np.sum(logp, axis=1),
        belief=belief_cur_all,
        associability=assoc_cur_all,
        control=control_all,
        actor_qdiff=qdiff_all,
    )

# =============================================================================
# Fit / evaluation
# =============================================================================


@dataclass
class FitConfig:
    draws: int
    tune: int
    chains: int
    cores: int
    seed: int
    target_accept: float


def result_dir(seed_out: Path, variant_key: str) -> Path:
    return seed_out / "results" / f"{variant_key}_posterior"


def fit_nuts(seed_out: Path, variant_key: str, arr: Dict[str, np.ndarray], cfg: FitConfig, force_refit: bool = False) -> Path:
    pm, az = require_bayes()
    variant = VARIANTS[variant_key]
    rd = result_dir(seed_out, variant_key)
    ensure_dir(rd)
    idata_path = rd / "idata.nc"
    if force_refit and idata_path.exists():
        shutil.rmtree(rd)
        ensure_dir(rd)
    if idata_path.exists():
        return idata_path

    print(f"[{variant_key}] {variant.name}: NUTS draws={cfg.draws} tune={cfg.tune} chains={cfg.chains}")
    model = build_model(arr, variant)
    progress = dict(time=now_str(), variant=variant_key, variant_name=variant.name, cfg=asdict(cfg), status="running")
    write_json(rd / "progress.json", progress)
    with model:
        idata = pm.sample(
            draws=cfg.draws,
            tune=cfg.tune,
            chains=cfg.chains,
            cores=cfg.cores,
            random_seed=cfg.seed,
            target_accept=cfg.target_accept,
            return_inferencedata=True,
            progressbar=True,
        )
    safe = az.InferenceData(posterior=idata.posterior, sample_stats=idata.sample_stats)
    safe.to_netcdf(str(idata_path))
    progress["status"] = "done"
    progress["finished_at"] = now_str()
    write_json(rd / "progress.json", progress)
    return idata_path


def posterior_mean_params(idata) -> Dict[str, np.ndarray]:
    return {v: idata.posterior[v].mean(dim=("chain", "draw")).values for v in idata.posterior.data_vars}


def diagnostics_summary(idata) -> Dict[str, Any]:
    pm, az = require_bayes()
    try:
        summ = az.summary(idata, kind="diagnostics")
        rhat = pd.to_numeric(summ.get("r_hat", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        ess = pd.to_numeric(summ.get("ess_bulk", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        rhat = rhat[np.isfinite(rhat)]
        ess = ess[np.isfinite(ess)]
        div = int(idata.sample_stats["diverging"].sum().values) if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats else None
        return dict(
            rhat_max=float(np.max(rhat)) if rhat.size else None,
            rhat_frac_gt_1_05=float(np.mean(rhat > 1.05)) if rhat.size else None,
            ess_bulk_min=float(np.min(ess)) if ess.size else None,
            divergences=div,
        )
    except Exception as e:
        return {"error": str(e)}


def ppc_summary(arr: Dict[str, np.ndarray], variant: VariantSpec, params: Dict[str, Any]) -> Dict[str, Any]:
    pred = forward_numpy(arr, variant, params)
    p = pred["p_press"]
    action = arr["action"].astype(float)
    nogo = arr["is_nogo"].astype(bool)
    group = arr["group"].astype(int)
    out = {
        "observed": {
            "press": float(np.mean(action)),
            "hit": float(np.mean(action[~nogo])),
            "fa": float(np.mean(action[nogo])),
        },
        "predicted": {
            "press": float(np.mean(p)),
            "hit": float(np.mean(p[~nogo])),
            "fa": float(np.mean(p[nogo])),
        },
        "by_group": {},
    }
    for gv, gn in [(0, "low"), (1, "high")]:
        # arr fields are subject x trial matrices, while group is subject-level.
        # Broadcast the subject mask across trials for trial-type masks.
        m_subj = (group == gv)
        m_trial = m_subj[:, None]
        nogo_mask = m_trial & nogo
        go_mask = m_trial & (~nogo)
        out["by_group"][gn] = {
            "observed_press": float(np.mean(action[m_subj])),
            "predicted_press": float(np.mean(p[m_subj])),
            "observed_fa": float(np.mean(action[nogo_mask])) if np.any(nogo_mask) else float("nan"),
            "predicted_fa": float(np.mean(p[nogo_mask])) if np.any(nogo_mask) else float("nan"),
            "observed_hit": float(np.mean(action[go_mask])) if np.any(go_mask) else float("nan"),
            "predicted_hit": float(np.mean(p[go_mask])) if np.any(go_mask) else float("nan"),
        }
    out["abs_error"] = {k: abs(out["observed"][k] - out["predicted"][k]) for k in ["press", "hit", "fa"]}
    out["fa_gap"] = {
        "observed_high_minus_low": out["by_group"]["high"]["observed_fa"] - out["by_group"]["low"]["observed_fa"],
        "predicted_high_minus_low": out["by_group"]["high"]["predicted_fa"] - out["by_group"]["low"]["predicted_fa"],
    }
    return out


def add_loglik_group(idata):
    pm, az = require_bayes()
    if "loglik_subj" not in idata.posterior:
        raise ValueError("Missing loglik_subj")
    ll = idata.posterior["loglik_subj"]
    last = ll.dims[-1]
    ll_da = ll.rename({last: "subject"})
    out = idata.copy()
    out.add_groups({"log_likelihood": xr.Dataset({"y": ll_da})})
    return out


def evaluate(seed_out: Path, variant_key: str, arr: Dict[str, np.ndarray]) -> None:
    pm, az = require_bayes()
    rd = result_dir(seed_out, variant_key)
    idata = az.from_netcdf(str(rd / "idata.nc"))
    az.summary(idata).to_csv(rd / "summary.csv")
    diag = diagnostics_summary(idata)
    write_json(rd / "diagnostics.json", diag)
    params = posterior_mean_params(idata)
    ppc = ppc_summary(arr, VARIANTS[variant_key], params)
    ppc.update(time=now_str(), variant=variant_key, variant_name=VARIANTS[variant_key].name)
    write_json(rd / "ppc.json", ppc)

    loo_out = {"time": now_str(), "notes": "PSIS/WAIC are diagnostic only; use subject-level K-fold CV for model comparison."}
    try:
        id2 = add_loglik_group(idata)
        loo = az.loo(id2, var_name="y", pointwise=True)
        waic = az.waic(id2, var_name="y", pointwise=True)
        pk = np.asarray(loo.pareto_k.values).astype(float).ravel()
        pk = pk[np.isfinite(pk)]
        loo_out.update(
            elpd_loo=float(getattr(loo, "elpd_loo")),
            p_loo=float(getattr(loo, "p_loo")),
            elpd_waic=float(getattr(waic, "elpd_waic")),
            p_waic=float(getattr(waic, "p_waic")),
            pareto_k_max=float(np.max(pk)) if pk.size else None,
            pareto_k_frac_gt_0_7=float(np.mean(pk > 0.7)) if pk.size else None,
            psis_reliable=bool(pk.size and np.max(pk) <= 0.7),
        )
    except Exception as e:
        loo_out["error"] = str(e)
    write_json(rd / "loo_waic_subject.json", loo_out)
    print(f"[{variant_key}] evaluation saved in {rd}")

# =============================================================================
# CV scoring
# =============================================================================


def subset_subjects(arr: Dict[str, np.ndarray], keep: List[int]) -> Dict[str, np.ndarray]:
    meta = get_meta(arr)
    keep = list(map(int, keep))
    out = {}
    for k, v in arr.items():
        if k == "_meta_json":
            continue
        if isinstance(v, np.ndarray) and v.shape and v.shape[0] == int(meta["n_subj"]):
            out[k] = v[keep]
        else:
            out[k] = v
    meta2 = dict(meta)
    meta2["orig_subject_indices"] = keep
    meta2["subj_ids"] = [meta["subj_ids"][i] for i in keep]
    meta2["n_subj"] = len(keep)
    meta2["n_trials"] = int(len(keep) * meta["n_trials_per_subj"])
    set_meta(out, meta2)
    return out


def stratified_folds(arr: Dict[str, np.ndarray], k: int, seed: int) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    group = arr["group"].astype(int)
    low = np.where(group == 0)[0]
    high = np.where(group == 1)[0]
    rng.shuffle(low)
    rng.shuffle(high)
    folds = [[] for _ in range(k)]
    for i, s in enumerate(low):
        folds[i % k].append(int(s))
    for i, s in enumerate(high):
        folds[i % k].append(int(s))
    for f in folds:
        f.sort()
    return folds


def posterior_draw_dicts(idata, max_draws: int) -> List[Dict[str, Any]]:
    post = idata.posterior
    C = int(post.sizes.get("chain", 1))
    D = int(post.sizes.get("draw", 1))
    total = C * D
    step = max(1, total // max_draws) if total > max_draws else 1
    out = []
    idx = 0
    for c in range(C):
        for d in range(D):
            if idx % step == 0:
                dd = {}
                for v in post.data_vars:
                    val = np.asarray(post[v].isel(chain=c, draw=d).values)
                    dd[v] = float(val.item()) if val.shape == () else val
                out.append(dd)
                if len(out) >= max_draws:
                    return out
            idx += 1
    return out


def sample_subject_array(dd: Dict[str, Any], name: str, group: np.ndarray, transform: str = "identity") -> np.ndarray:
    S = len(group)
    mu = safe_float(dd.get(f"mu_{name}", 0.0), 0.0)
    delta = safe_float(dd.get(f"delta_{name}_group", 0.0), 0.0)
    sigma = abs(safe_float(dd.get(f"sigma_{name}", 0.0), 0.0))
    seed_material = f'{name}|{mu:.17g}|{delta:.17g}|{sigma:.17g}'.encode('utf-8')
    seed_value = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], 'little')
    rng = np.random.default_rng(seed_value)
    lat = rng.normal(mu + delta * group.astype(float), sigma + 1e-6, size=S)
    if transform == "sigmoid":
        return sigmoid_np(lat)
    if transform == "exp":
        return np.exp(lat)
    return lat


def params_for_new_subjects(dd: Dict[str, Any], variant: VariantSpec, group: np.ndarray, n_cue: int, seed: int) -> Dict[str, Any]:
    # Use a deterministic RNG per draw and re-sample arrays using hyperparameters.
    rng = np.random.default_rng(seed)
    S = len(group)
    def draw_lat(name, transform="identity"):
        mu = safe_float(dd.get(f"mu_{name}", 0.0), 0.0)
        delta = safe_float(dd.get(f"delta_{name}_group", 0.0), 0.0)
        sigma = abs(safe_float(dd.get(f"sigma_{name}", 0.0), 0.0))
        lat = rng.normal(mu + delta * group.astype(float), sigma + 1e-6, size=S)
        if transform == "sigmoid": return sigmoid_np(lat)
        if transform == "exp": return np.exp(lat)
        return lat

    p = {
        "bias": draw_lat("bias"),
        "b_cue": np.asarray(dd.get("b_cue", np.zeros(n_cue)), dtype=float).reshape(n_cue),
        "b_nogo": safe_float(dd.get("b_nogo", 0.0), 0.0),
        "b_prev_error": safe_float(dd.get("b_prev_error", 0.0), 0.0),
        "lapse_eps": safe_float(dd.get("lapse_eps", 0.0), 0.0),
        "alpha_belief": np.zeros(S),
        "alpha_pos": np.zeros(S),
        "alpha_neg": np.zeros(S),
        "alpha_base": np.zeros(S),
        "eta_assoc": np.zeros(S),
        "w_assoc": safe_float(dd.get("w_assoc", 0.0), 0.0),
        "w_belief": safe_float(dd.get("w_belief", 0.0), 0.0),
        "alpha_ctrl": np.zeros(S),
        "w_control": safe_float(dd.get("w_control", 0.0), 0.0),
        "alpha_actor": np.zeros(S),
        "beta_actor": np.zeros(S),
    }
    if variant.use_belief and not variant.use_volatility and not variant.use_asym_belief:
        p["alpha_belief"] = draw_lat("alpha_belief", "sigmoid")
    if variant.use_asym_belief:
        p["alpha_pos"] = draw_lat("alpha_pos", "sigmoid")
        p["alpha_neg"] = draw_lat("alpha_neg", "sigmoid")
    if variant.use_volatility:
        p["alpha_base"] = draw_lat("alpha_base")
        p["eta_assoc"] = draw_lat("eta_assoc", "sigmoid")
    if variant.use_control:
        p["alpha_ctrl"] = draw_lat("alpha_ctrl", "sigmoid")
    if variant.use_actor:
        p["alpha_actor"] = draw_lat("alpha_actor", "sigmoid")
        p["beta_actor"] = draw_lat("logbeta_actor", "exp")
    return p


def score_test(arr_test: Dict[str, np.ndarray], variant_key: str, idata, max_post_draws: int, seed: int) -> Dict[str, Any]:
    variant = VARIANTS[variant_key]
    meta = get_meta(arr_test)
    S = int(meta["n_subj"])
    C = len(meta["cue_levels"])
    group = arr_test["group"].astype(int)
    draws = posterior_draw_dicts(idata, max_draws=max_post_draws)
    ll = np.zeros((len(draws), S))
    for i, dd in enumerate(draws):
        params = params_for_new_subjects(dd, variant, group, C, seed + i * 1009)
        fwd = forward_numpy(arr_test, variant, params)
        ll[i] = fwd["loglik_subj"]
    lpd = logsumexp_np(ll, axis=0) - math.log(ll.shape[0])
    return {"elpd": float(np.sum(lpd)), "lpd_subj": lpd.tolist()}

# =============================================================================
# Orchestration
# =============================================================================


def suite(out: Path, data_root: Optional[Path], seeds: List[int], variants: List[str], cfg: FitConfig, rebuild: bool, force_refit: bool):
    ensure_dir(out)
    if data_root is not None:
        preprocess(data_root, out, rebuild=rebuild)
    arr = load_artifacts(out)
    for seed in seeds:
        seed_out = out / f"seed_{seed}"
        ensure_dir(seed_out)
        for vk in variants:
            fit_nuts(seed_out, vk, arr, cfg, force_refit=force_refit)
            evaluate(seed_out, vk, arr)


def compare_suite(out: Path, seed: int, variants: List[str]) -> None:
    rows = []
    for vk in variants:
        rd = result_dir(out / f"seed_{seed}", vk)
        if not rd.exists():
            continue
        ppc = read_json(rd / "ppc.json", default={})
        diag = read_json(rd / "diagnostics.json", default={})
        loo = read_json(rd / "loo_waic_subject.json", default={})
        rows.append({
            "variant": vk,
            "variant_name": VARIANTS[vk].name,
            "fa_abs_error": ppc.get("abs_error", {}).get("fa"),
            "hit_abs_error": ppc.get("abs_error", {}).get("hit"),
            "press_abs_error": ppc.get("abs_error", {}).get("press"),
            "obs_fa_gap": ppc.get("fa_gap", {}).get("observed_high_minus_low"),
            "pred_fa_gap": ppc.get("fa_gap", {}).get("predicted_high_minus_low"),
            "rhat_max": diag.get("rhat_max"),
            "rhat_frac_gt_1_05": diag.get("rhat_frac_gt_1_05"),
            "ess_bulk_min": diag.get("ess_bulk_min"),
            "divergences": diag.get("divergences"),
            "elpd_loo_diagnostic": loo.get("elpd_loo"),
            "psis_reliable": loo.get("psis_reliable"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["fa_abs_error", "variant"])
    out_csv = out / f"seed_{seed}" / "compare_suite_actor_control_rl.csv"
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print("[compare_suite] wrote", out_csv)


def cv(out: Path, seed: int, variants: List[str], k: int, cfg: FitConfig, max_post_draws: int, force_refit: bool):
    arr = load_artifacts(out)
    folds = stratified_folds(arr, k, seed)
    meta = get_meta(arr)
    manifest = pd.read_csv(out / "artifacts" / "manifest.csv")
    seed_out = out / f"seed_{seed}"
    ensure_dir(seed_out)
    for vk in variants:
        rd = result_dir(seed_out, vk)
        ensure_dir(rd)
        cv_path = rd / "cv_kfold_subject.json"
        if force_refit and cv_path.exists():
            cv_path.unlink()
        fold_results = []
        for fi, test_subs in enumerate(folds):
            train_subs = [s for j, f in enumerate(folds) if j != fi for s in f]
            fold_dir = rd / "cv_folds" / f"fold_{fi:02d}"
            fr_path = fold_dir / "fold_result.json"
            if fr_path.exists() and not force_refit:
                fr = read_json(fr_path)
                fold_results.append(fr)
                continue
            print("="*90)
            print(f"[cv] {vk} fold {fi+1}/{k} train={len(train_subs)} test={len(test_subs)}")
            arr_train = subset_subjects(arr, train_subs)
            arr_test = subset_subjects(arr, test_subs)
            fold_seed_out = fold_dir
            fold_cfg = FitConfig(draws=cfg.draws, tune=cfg.tune, chains=cfg.chains, cores=cfg.cores, seed=seed + 97*(fi+1), target_accept=cfg.target_accept)
            idata_path = fit_nuts(fold_seed_out, vk, arr_train, fold_cfg, force_refit=force_refit)
            pm, az = require_bayes()
            idata = az.from_netcdf(str(idata_path))
            sc = score_test(arr_test, vk, idata, max_post_draws=max_post_draws, seed=seed + 1000*(fi+1))
            fr = {
                "fold": fi,
                "status": "done",
                "n_train": len(train_subs),
                "n_test": len(test_subs),
                "test_subject_indices": [int(s) for s in test_subs],
                "test_subject_ids": [str(manifest.loc[int(s), "subj_id"]) for s in test_subs],
                "test_groups": [int(manifest.loc[int(s), "group"]) for s in test_subs],
                "elpd": sc["elpd"],
                "lpd_subj": sc["lpd_subj"],
                "draws": cfg.draws,
                "tune": cfg.tune,
                "max_post_draws": max_post_draws,
            }
            write_json(fr_path, fr)
            fold_results.append(fr)
        total = float(sum(fr["elpd"] for fr in fold_results))
        out_obj = {
            "time": now_str(),
            "variant": vk,
            "variant_name": VARIANTS[vk].name,
            "seed": seed,
            "k": k,
            "total_elpd": total,
            "folds": fold_results,
            "status": "done",
            "notes": "Subject-level K-fold CV; new-subject random effects integrated by posterior hyperparameter sampling.",
        }
        write_json(cv_path, out_obj)
        print("[cv] wrote", cv_path)


def compare_cv(out: Path, seeds: List[int], variants: List[str]) -> None:
    rows = []
    for seed in seeds:
        for vk in variants:
            p = result_dir(out / f"seed_{seed}", vk) / "cv_kfold_subject.json"
            if not p.exists():
                continue
            obj = read_json(p)
            rows.append({
                "seed": seed,
                "variant": vk,
                "variant_name": VARIANTS[vk].name,
                "total_elpd": float(obj["total_elpd"]),
                "k": int(obj["k"]),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        print("[compare_cv] no CV outputs found")
        return
    agg = df.groupby(["variant", "variant_name"], as_index=False).agg(
        mean_total_elpd=("total_elpd", "mean"),
        sd_total_elpd=("total_elpd", "std"),
        n_seeds=("seed", "nunique"),
    ).sort_values("mean_total_elpd", ascending=False)
    out_dir = out / "cv_comparison_actor_control_rl"
    ensure_dir(out_dir)
    df.to_csv(out_dir / "cv_by_seed.csv", index=False)
    agg.to_csv(out_dir / "cv_summary.csv", index=False)
    print(agg.to_string(index=False))
    print("[compare_cv] wrote", out_dir)

# =============================================================================
# Export parameters / latents / association table
# =============================================================================


def export_params(out: Path, seed: int, variant_key: str) -> None:
    pm, az = require_bayes()
    arr = load_artifacts(out)
    meta = get_meta(arr)
    rd = result_dir(out / f"seed_{seed}", variant_key)
    idata = az.from_netcdf(str(rd / "idata.nc"))
    params = posterior_mean_params(idata)
    group = arr["group"].astype(int)
    rows = []
    for s, sid in enumerate(meta["subj_ids"]):
        row = {"subject_index": s, "subject_id": sid, "group": group[s], "group_label": "high" if group[s] else "low"}
        for name in ["bias", "alpha_belief", "alpha_pos", "alpha_neg", "alpha_ctrl", "alpha_actor", "beta_actor"]:
            if name in params and np.asarray(params[name]).shape and len(np.asarray(params[name]).reshape(-1)) >= s+1:
                row[name] = float(np.asarray(params[name]).reshape(-1)[s])
        for name in ["w_belief", "w_control", "b_nogo", "b_prev_error", "lapse_eps", "w_assoc"]:
            if name in params:
                row[name] = float(np.asarray(params[name]).reshape(-1)[0])
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = rd / "subject_params.csv"
    df.to_csv(out_csv, index=False)
    print("[export_params] wrote", out_csv)


def export_latents(out: Path, seed: int, variant_key: str) -> None:
    pm, az = require_bayes()
    arr = load_artifacts(out)
    meta = get_meta(arr)
    rd = result_dir(out / f"seed_{seed}", variant_key)
    idata = az.from_netcdf(str(rd / "idata.nc"))
    params = posterior_mean_params(idata)
    variant = VARIANTS[variant_key]
    fwd = forward_numpy(arr, variant, params)
    rows = []
    cue_levels = meta["cue_levels"]
    for s, sid in enumerate(meta["subj_ids"]):
        for t in range(int(meta["n_trials_per_subj"])):
            rows.append({
                "subject_index": s,
                "subject_id": sid,
                "group": int(arr["group"][s]),
                "trial_index": t,
                "cue_idx": int(arr["cue_idx"][s, t]),
                "cue_type": cue_levels[int(arr["cue_idx"][s, t])],
                "is_nogo": int(arr["is_nogo"][s, t]),
                "action_press": int(arr["action"][s, t]),
                "correct": int(arr["correct"][s, t]),
                "error": int(arr["error"][s, t]),
                "fa": int(arr["fa"][s, t]),
                "p_press": float(fwd["p_press"][s, t]),
                "belief_nogo_current": float(fwd["belief"][s, t]),
                "associability_current": float(fwd["associability"][s, t]),
                "control_current": float(fwd["control"][s, t]),
                "actor_qdiff_current": float(fwd["actor_qdiff"][s, t]),
                "cnv_z": float(arr["cnv_z"][s, t]),
                "p300_z": float(arr["p300_z"][s, t]),
                "ern_strength_z": float(arr["ern_strength_z"][s, t]),
                "pe_strength_z": float(arr["pe_strength_z"][s, t]),
            })
    df = pd.DataFrame(rows)
    out_csv = rd / "trial_latents_for_ddm.csv"
    df.to_csv(out_csv, index=False)
    print("[export_latents] wrote", out_csv)


def association_table(out: Path, seed: int, variant_key: str) -> None:
    arr = load_artifacts(out)
    meta = get_meta(arr)
    rd = result_dir(out / f"seed_{seed}", variant_key)
    # Ensure params exist.
    if not (rd / "subject_params.csv").exists():
        export_params(out, seed, variant_key)
    params = pd.read_csv(rd / "subject_params.csv")
    rows = []
    for s, sid in enumerate(meta["subj_ids"]):
        fa_rate = np.mean(arr["action"][s][arr["is_nogo"][s].astype(bool)])
        hit_rate = np.mean(arr["action"][s][~arr["is_nogo"][s].astype(bool)])
        ern_vals = arr["ern_strength_z"][s][arr["fa"][s].astype(bool)]
        pe_vals = arr["pe_strength_z"][s][arr["fa"][s].astype(bool)]
        rows.append({
            "subject_index": s,
            "subject_id": sid,
            "group": int(arr["group"][s]),
            "fa_rate": float(fa_rate),
            "hit_rate": float(hit_rate),
            "ern_strength_mean": float(np.mean(ern_vals)) if ern_vals.size else np.nan,
            "ern_strength_sd": float(np.std(ern_vals)) if ern_vals.size else np.nan,
            "pe_strength_mean": float(np.mean(pe_vals)) if pe_vals.size else np.nan,
            "cnv_mean": float(np.mean(arr["cnv_z"][s])),
            "p300_mean": float(np.mean(arr["p300_z"][s])),
        })
    subj = pd.DataFrame(rows)
    merged = subj.merge(params, on=["subject_index", "subject_id", "group"], how="left")
    out_csv = rd / "subject_rl_erp_associations.csv"
    merged.to_csv(out_csv, index=False)
    print("[association_table] wrote", out_csv)

# =============================================================================
# CLI
# =============================================================================


def parse_args():
    ap = argparse.ArgumentParser(description="actor-control RL model family for Cued Go/Nogo")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preprocess")
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rebuild", action="store_true")

    p = sub.add_parser("doctor")
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("suite")
    p.add_argument("--data_root", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--variants", nargs="+", default=DEFAULT_LADDER)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--cores", type=int, default=4)
    p.add_argument("--target_accept", type=float, default=0.9)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--force_refit", action="store_true")

    p = sub.add_parser("compare_suite")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--variants", nargs="+", default=DEFAULT_LADDER)

    p = sub.add_parser("cv")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--variants", nargs="+", default=DEFAULT_LADDER)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--draws", type=int, default=700)
    p.add_argument("--tune", type=int, default=700)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--cores", type=int, default=4)
    p.add_argument("--target_accept", type=float, default=0.9)
    p.add_argument("--max_post_draws", type=int, default=500)
    p.add_argument("--force_refit", action="store_true")

    p = sub.add_parser("compare_cv")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--variants", nargs="+", default=DEFAULT_LADDER)

    p = sub.add_parser("export_params")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--variant", required=True)

    p = sub.add_parser("export_latents")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--variant", required=True)

    p = sub.add_parser("association_table")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--variant", required=True)

    return ap.parse_args()


def main():
    args = parse_args()
    if args.cmd == "preprocess":
        preprocess(args.data_root.resolve(), args.out.resolve(), rebuild=args.rebuild)
    elif args.cmd == "doctor":
        doctor(args.out.resolve())
    elif args.cmd == "suite":
        cfg = FitConfig(args.draws, args.tune, args.chains, args.cores, args.seeds[0], args.target_accept)
        # seed inside cfg is updated per seed below for reproducibility.
        out = args.out.resolve()
        ensure_dir(out)
        if args.data_root is not None:
            preprocess(args.data_root.resolve(), out, rebuild=args.rebuild)
        arr = load_artifacts(out)
        for seed in args.seeds:
            cfg_seed = FitConfig(args.draws, args.tune, args.chains, args.cores, seed, args.target_accept)
            seed_out = out / f"seed_{seed}"
            ensure_dir(seed_out)
            for vk in args.variants:
                fit_nuts(seed_out, vk, arr, cfg_seed, force_refit=args.force_refit)
                evaluate(seed_out, vk, arr)
    elif args.cmd == "compare_suite":
        compare_suite(args.out.resolve(), args.seed, args.variants)
    elif args.cmd == "cv":
        cfg = FitConfig(args.draws, args.tune, args.chains, args.cores, args.seed, args.target_accept)
        cv(args.out.resolve(), args.seed, args.variants, args.k, cfg, args.max_post_draws, args.force_refit)
    elif args.cmd == "compare_cv":
        compare_cv(args.out.resolve(), args.seeds, args.variants)
    elif args.cmd == "export_params":
        export_params(args.out.resolve(), args.seed, args.variant)
    elif args.cmd == "export_latents":
        export_latents(args.out.resolve(), args.seed, args.variant)
    elif args.cmd == "association_table":
        association_table(args.out.resolve(), args.seed, args.variant)


if __name__ == "__main__":
    main()
