#!/usr/bin/env python3
"""Shared utilities for the reproducibility analyses."""
from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def import_module(path: str | Path, name: str):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def norm_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def normalize_subject_id(x: Any) -> str:
    s = str(x).strip()
    s = re.sub(r"\.(set|fdt|xlsx|csv)$", "", s, flags=re.I)
    s = re.sub(r"[^A-Za-z0-9_-]+", "", s)
    return s


def fdr_bh(pvals: Sequence[float]) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    ok = np.isfinite(pvals)
    if not np.any(ok):
        return out
    p = pvals[ok]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty(n)
    tmp[order] = q
    out[ok] = tmp
    return out


def stratified_subject_folds(
    df: pd.DataFrame,
    subject_col: str,
    group_col: str | None,
    k: int,
    seed: int,
) -> list[list[str]]:
    """Make deterministic subject folds, stratified by a binary/group label if available."""
    if k < 2:
        raise ValueError("k must be >=2")
    subj = df[[subject_col] + ([group_col] if group_col and group_col in df.columns else [])].drop_duplicates(subject_col)
    subj[subject_col] = subj[subject_col].astype(str)
    rng = np.random.default_rng(seed)
    folds: list[list[str]] = [[] for _ in range(k)]
    if group_col and group_col in subj.columns:
        for _, g in subj.groupby(group_col, dropna=False, sort=True):
            ids = g[subject_col].astype(str).to_numpy()
            rng.shuffle(ids)
            for i, sid in enumerate(ids):
                folds[i % k].append(str(sid))
    else:
        ids = subj[subject_col].astype(str).to_numpy()
        rng.shuffle(ids)
        for i, sid in enumerate(ids):
            folds[i % k].append(str(sid))
    return [sorted(f) for f in folds]


def empirical_p_one_sided(observed: float, null: Sequence[float], greater: bool = True) -> float:
    a = np.asarray(null, dtype=float)
    a = a[np.isfinite(a)]
    if not np.isfinite(observed) or len(a) == 0:
        return np.nan
    n = np.sum(a >= observed) if greater else np.sum(a <= observed)
    return float((n + 1) / (len(a) + 1))


def empirical_p_two_sided(observed: float, null: Sequence[float]) -> float:
    a = np.asarray(null, dtype=float)
    a = a[np.isfinite(a)]
    if not np.isfinite(observed) or len(a) == 0:
        return np.nan
    return float((np.sum(np.abs(a) >= abs(observed)) + 1) / (len(a) + 1))


def signflip_p(values: Sequence[float], n_perm: int, seed: int) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    obs = float(np.mean(vals))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        null[i] = np.mean(vals * rng.choice([-1.0, 1.0], size=len(vals)))
    return empirical_p_two_sided(obs, null)


def bootstrap_mean_ci(values: Sequence[float], n_boot: int, seed: int, alpha: float = 0.05) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = np.mean(rng.choice(vals, size=len(vals), replace=True))
    return float(np.mean(vals)), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def within_subject_z(df: pd.DataFrame, col: str, subject_col: str, suffix: str = "_wpz") -> str:
    out_col = f"{col}{suffix}"
    x = pd.to_numeric(df[col], errors="coerce")
    means = x.groupby(df[subject_col]).transform("mean")
    sds = x.groupby(df[subject_col]).transform("std").replace(0, np.nan)
    df[out_col] = (x - means) / sds
    return out_col


def infer_outcome(df: pd.DataFrame) -> pd.Series:
    if "outcome" in df.columns:
        out = df["outcome"].map(norm_text)
        out = out.replace({"hit": "go_hit", "miss": "go_miss", "fa": "nogo_fa", "cr": "nogo_cr"})
        return out
    if "response_type" in df.columns:
        r = df["response_type"].map(norm_text)
        out = r.map({
            "hit": "go_hit", "miss": "go_miss", "fa": "nogo_fa", "cr": "nogo_cr",
            "false alarm": "nogo_fa", "false_alarm": "nogo_fa",
            "correct rejection": "nogo_cr", "correct_rejection": "nogo_cr",
        })
    else:
        out = pd.Series(np.nan, index=df.index, dtype=object)
    trial = df.get("trial_class", pd.Series("", index=df.index)).map(norm_text)
    press = pd.to_numeric(df.get("response_present", 0), errors="coerce").fillna(0).astype(int)
    out = out.astype(object)
    out.loc[out.isna() & trial.eq("go") & press.eq(1)] = "go_hit"
    out.loc[out.isna() & trial.eq("go") & press.eq(0)] = "go_miss"
    out.loc[out.isna() & trial.eq("nogo") & press.eq(1)] = "nogo_fa"
    out.loc[out.isna() & trial.eq("nogo") & press.eq(0)] = "nogo_cr"
    return out


def add_trial_history(
    df: pd.DataFrame,
    subject_col: str,
    trial_col: str,
    block_size: int = 0,
) -> pd.DataFrame:
    out = df.copy()
    out[subject_col] = out[subject_col].astype(str)
    out[trial_col] = pd.to_numeric(out[trial_col], errors="coerce")
    out = out.sort_values([subject_col, trial_col]).reset_index(drop=True)
    out["outcome"] = infer_outcome(out)
    out["response_present_numeric"] = pd.to_numeric(out.get("response_present", 0), errors="coerce").fillna(0).astype(int)
    out["is_nogo"] = out.get("trial_class", "").map(norm_text).eq("nogo").astype(int)
    out["error"] = out["outcome"].isin(["nogo_fa", "go_miss"]).astype(int)
    out["fa"] = out["outcome"].eq("nogo_fa").astype(int)
    out["prev_error"] = out.groupby(subject_col)["error"].shift(1).fillna(0).astype(int)
    out["prev_press"] = out.groupby(subject_col)["response_present_numeric"].shift(1).fillna(0).astype(int)
    out["trial_order"] = out.groupby(subject_col).cumcount()
    n_by_sub = out.groupby(subject_col)["trial_order"].transform("max").replace(0, 1)
    out["trial_progress"] = out["trial_order"] / n_by_sub
    if block_size and block_size > 0:
        out["block"] = (out["trial_order"] // int(block_size)).astype(int)
    elif "block" not in out.columns:
        out["block"] = 0
    out["next_block"] = out.groupby(subject_col)["block"].shift(-1)
    out["same_block_next"] = out["block"].eq(out["next_block"])
    return out


def safe_logit(p: Iterable[float], eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(list(p) if not isinstance(p, np.ndarray) else p, dtype=float)
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def binary_entropy(p: pd.Series | np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return -(x * np.log2(x) + (1 - x) * np.log2(1 - x))


def action_surprise(p_press: pd.Series | np.ndarray, press: pd.Series | np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p_press, dtype=float), eps, 1 - eps)
    a = np.asarray(press, dtype=float)
    pa = np.where(a > 0.5, p, 1 - p)
    return -np.log2(np.clip(pa, eps, 1))


def cue_probability_map(path: str | Path | None = None) -> dict[str, float]:
    if path:
        raw = read_json(path)
        return {norm_text(k): float(v) for k, v in raw.items()}
    return {"up": 0.38, "bi": 0.20, "bidirectional": 0.20, "both": 0.20, "down": 0.12}


def add_instructed_and_empirical_cue_beliefs(
    df: pd.DataFrame,
    subject_col: str,
    cue_col: str,
    target_col: str,
    trial_col: str,
    probs: dict[str, float],
    prior_strength: float = 10.0,
) -> pd.DataFrame:
    """Add instructed probability and strictly lagged Beta-Binomial cue belief.

    The empirical value at trial t uses only target outcomes before t for the same
    participant and cue. The Beta prior mean equals the instructed cue probability.
    """
    out = df.copy()
    out[subject_col] = out[subject_col].astype(str)
    out[cue_col] = out[cue_col].map(norm_text)
    out[target_col] = out[target_col].map(norm_text)
    out[trial_col] = pd.to_numeric(out[trial_col], errors="coerce")
    out = out.sort_values([subject_col, trial_col]).reset_index(drop=True)
    out["instructed_nogo_prob"] = out[cue_col].map(probs)
    if out["instructed_nogo_prob"].isna().any():
        missing = sorted(out.loc[out["instructed_nogo_prob"].isna(), cue_col].dropna().unique())
        raise ValueError(f"No instructed probability for cue labels: {missing}")
    out["instructed_nogo_logit"] = safe_logit(out["instructed_nogo_prob"].to_numpy())
    is_nogo = out[target_col].eq("nogo").astype(float)
    out["_is_nogo_tmp"] = is_nogo
    grp = out.groupby([subject_col, cue_col], sort=False)
    prior_n = grp.cumcount().astype(float)
    prior_ng = grp["_is_nogo_tmp"].transform(lambda s: s.cumsum().shift(1)).fillna(0.0)
    alpha0 = out["instructed_nogo_prob"] * float(prior_strength)
    beta0 = (1 - out["instructed_nogo_prob"]) * float(prior_strength)
    out["empirical_nogo_prob"] = (alpha0 + prior_ng) / (alpha0 + beta0 + prior_n)
    out["empirical_nogo_logit"] = safe_logit(out["empirical_nogo_prob"].to_numpy())
    out["cue_prior_n"] = prior_n
    return out.drop(columns=["_is_nogo_tmp"])


def choose_first(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"None of the required columns are present: {candidates}")
    return None


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "(no rows)"
    return df.head(max_rows).to_markdown(index=False)
