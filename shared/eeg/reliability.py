#!/usr/bin/env python3
"""Split-half reliability and trial-count curves for EEG features/contrasts.

For every participant, trials from each condition are randomly split into two
halves. The condition contrast is calculated independently in each half, then
correlated across participants. Spearman-Brown correction estimates full-length
reliability. Repeating the split gives an uncertainty distribution.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from common_utils import ensure_dir, infer_outcome  # noqa: E402

DEFAULT_PATTERNS = [
    r"^resp_ern_fz_50_150_(mean|min|max)_uv$", r"^resp_pe_(fz|cz)_200_400_(mean|min|max)_uv$",
    r"^resp_(theta|beta|mu)_", r"^target_(theta|alpha|beta)_", r"^cue_(theta|alpha|beta)_",
]
DEFAULT_CONTRASTS = [
    ("response_fa_minus_hit", "nogo_fa", "go_hit", "resp"),
    ("target_fa_minus_cr", "nogo_fa", "nogo_cr", "target"),
    ("target_nogo_minus_go_correct", "nogo_cr", "go_hit", "target"),
]


def load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".pq", ".parquet"} else pd.read_csv(path, low_memory=False)


def canonicalize(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    if "subject_id" not in d.columns and "subject_id_norm" in d.columns:
        d = d.rename(columns={"subject_id_norm": "subject_id"})
    if "subject_id" not in d.columns:
        raise ValueError("subject_id is required")
    d["subject_id"] = d["subject_id"].astype(str)
    d["outcome"] = infer_outcome(d)
    return d


def merge(paths: list[Path]) -> pd.DataFrame:
    tables = [canonicalize(load(p)) for p in paths]
    possible_trial = ["trial_index_numeric", "trial_index", "trial_ordinal"]
    trial = next((c for c in possible_trial if c in tables[0].columns), None)
    if trial is None:
        raise ValueError("A trial index column is required")
    if trial != "trial_index_numeric":
        tables[0] = tables[0].rename(columns={trial: "trial_index_numeric"})
    out = tables[0]
    for t in tables[1:]:
        tr = next((c for c in possible_trial if c in t.columns), None)
        if tr is None:
            raise ValueError("Every table needs a trial index")
        if tr != "trial_index_numeric":
            t = t.rename(columns={tr: "trial_index_numeric"})
        keys = ["subject_id", "trial_index_numeric"]
        keep = keys + [c for c in t.columns if c not in out.columns]
        out = out.merge(t[keep].drop_duplicates(keys), on=keys, how="left", validate="one_to_one")
    return canonicalize(out)


def select_features(d: pd.DataFrame, explicit: list[str] | None) -> list[str]:
    if explicit:
        miss = [c for c in explicit if c not in d.columns]
        if miss:
            raise ValueError(f"Missing features: {miss}")
        return explicit
    return [c for c in d.columns if any(re.search(p, c, flags=re.I) for p in DEFAULT_PATTERNS)]


def sb_correct(r: float) -> float:
    if not np.isfinite(r) or np.isclose(1 + r, 0):
        return np.nan
    return float(2 * r / (1 + r))


def one_split(
    d: pd.DataFrame, feature: str, a: str, b: str, min_trials: int, rng: np.random.Generator,
) -> tuple[float, float, int]:
    h1, h2 = [], []
    for sid, sd in d.groupby("subject_id", sort=False):
        av = pd.to_numeric(sd.loc[sd["outcome"].eq(a), feature], errors="coerce").dropna().to_numpy()
        bv = pd.to_numeric(sd.loc[sd["outcome"].eq(b), feature], errors="coerce").dropna().to_numpy()
        if len(av) < min_trials or len(bv) < min_trials or len(av) < 4 or len(bv) < 4:
            continue
        rng.shuffle(av); rng.shuffle(bv)
        ah = len(av) // 2; bh = len(bv) // 2
        if min(ah, len(av)-ah, bh, len(bv)-bh) < 2:
            continue
        h1.append(np.mean(av[:ah]) - np.mean(bv[:bh]))
        h2.append(np.mean(av[ah:]) - np.mean(bv[bh:]))
    if len(h1) < 8 or np.std(h1) == 0 or np.std(h2) == 0:
        return np.nan, np.nan, len(h1)
    pr = float(pearsonr(h1, h2).statistic)
    sr = float(spearmanr(h1, h2).statistic)
    return sb_correct(pr), sb_correct(sr), len(h1)


def run(d: pd.DataFrame, features: list[str], thresholds: list[int], reps: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    raw = []
    for name, a, b, prefix in DEFAULT_CONTRASTS:
        eligible_features = [f for f in features if f.lower().startswith(prefix)]
        for f in eligible_features:
            for minimum in thresholds:
                for rep in range(reps):
                    pr, sr, n = one_split(d, f, a, b, minimum, rng)
                    raw.append({"contrast": name, "condition_a": a, "condition_b": b, "feature": f,
                                "min_trials_each_condition": minimum, "rep": rep,
                                "n_subjects": n, "sb_pearson": pr, "sb_spearman": sr})
    raw_df = pd.DataFrame(raw)
    if raw_df.empty:
        return raw_df, pd.DataFrame()
    summary = raw_df.groupby(["contrast", "feature", "min_trials_each_condition"], as_index=False).agg(
        n_reps_valid=("sb_pearson", lambda x: int(np.isfinite(x).sum())),
        median_n_subjects=("n_subjects", "median"),
        median_sb_pearson=("sb_pearson", "median"),
        sb_pearson_q025=("sb_pearson", lambda x: np.nanquantile(x, .025) if np.isfinite(x).any() else np.nan),
        sb_pearson_q975=("sb_pearson", lambda x: np.nanquantile(x, .975) if np.isfinite(x).any() else np.nan),
        median_sb_spearman=("sb_spearman", "median"),
        sb_spearman_q025=("sb_spearman", lambda x: np.nanquantile(x, .025) if np.isfinite(x).any() else np.nan),
        sb_spearman_q975=("sb_spearman", lambda x: np.nanquantile(x, .975) if np.isfinite(x).any() else np.nan),
    )
    return raw_df, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial-features", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-cues", nargs="*", default=None)
    ap.add_argument("--features", nargs="*", default=None)
    ap.add_argument("--min-trials", nargs="*", type=int, default=[5, 10, 15, 20])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    out = ensure_dir(args.out)
    d = merge([Path(x) for x in args.trial_features])
    if args.include_cues and "cue_type" in d.columns:
        allowed = {str(x).strip().lower() for x in args.include_cues}
        d = d[d["cue_type"].astype(str).str.lower().isin(allowed)].copy()
    features = select_features(d, args.features)
    if not features:
        raise ValueError("No EEG features detected; pass --features explicitly")
    raw, summary = run(d, features, args.min_trials, args.reps, args.seed)
    raw.to_csv(out / "eeg_split_half_reliability_repetitions.csv.gz", index=False, compression="gzip")
    summary.to_csv(out / "eeg_split_half_reliability_summary.csv", index=False)
    lines = [
        "# EEG reliability report", "", summary.to_markdown(index=False) if not summary.empty else "No estimable contrasts.", "",
        "## Interpretation", "",
        "- Reliability is reported for the contrast used in inference, not merely the raw condition mean.",
        "- Values below roughly .50 imply substantial attenuation of individual-difference correlations; do not turn this into a rigid pass/fail threshold.",
        "- Trial-count curves show the precision/retention trade-off and should accompany minimum-trial sensitivity results.",
    ]
    (out / "EEG_RELIABILITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
