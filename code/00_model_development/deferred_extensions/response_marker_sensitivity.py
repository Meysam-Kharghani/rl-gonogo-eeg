#!/usr/bin/env python3
# FORMALIZED HISTORICAL SOURCE SNAPSHOT
# Evidence status: STOPPED_BY_DESIGN; NOT_PART_OF_FINAL_EVIDENCE
# This file preserves a historical project stage with public-facing terminology standardized for release.
# The SHA-256 of the original executed/development source is recorded in the corresponding provenance registry.

"""Sensitivity of response-locked effects to response-marker alignment quality.

The current extraction uses target onset + behavioral RT as the response lock and
records the distance to the nearest direct EEG response marker. This script asks
whether primary FA-vs-Hit response-locked contrasts remain stable after retaining
only trials whose metadata-derived onset is close to a direct marker.

It does not re-extract marker-locked EEG. If discrepancies are substantial, modify
the extractor to use direct markers and rerun response-locked analyses.
"""
from __future__ import annotations
import hashlib

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from common_utils import bootstrap_mean_ci, ensure_dir, infer_outcome, signflip_p  # noqa: E402

PATTERNS = [
    r"^resp_ern_fz_50_150_(mean|min|max)_uv$", r"^resp_pe_(fz|cz)_200_400_(mean|min|max)_uv$",
    r"^resp_(theta|beta|mu)_",
]


def load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".pq", ".parquet"} else pd.read_csv(path, low_memory=False)


def canonicalize(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    if "subject_id" not in d.columns and "subject_id_norm" in d.columns:
        d = d.rename(columns={"subject_id_norm": "subject_id"})
    if "response_marker_error_ms" not in d.columns:
        raise ValueError("response_marker_error_ms is absent; use eeg_trial_neural_features.csv or a table containing it")
    d["subject_id"] = d["subject_id"].astype(str)
    d["outcome"] = infer_outcome(d)
    d["response_marker_error_ms"] = pd.to_numeric(d["response_marker_error_ms"], errors="coerce")
    d["abs_marker_error_ms"] = d["response_marker_error_ms"].abs()
    return d


def merge(paths: list[Path]) -> pd.DataFrame:
    tables = [canonicalize(load(paths[0]))] + [load(p) for p in paths[1:]]
    out = tables[0]
    key_candidates = ["trial_index_numeric", "trial_index", "trial_ordinal"]
    key = next((x for x in key_candidates if x in out.columns), None)
    if key is None:
        raise ValueError("No trial index found")
    if key != "trial_index_numeric":
        out = out.rename(columns={key: "trial_index_numeric"})
    for t in tables[1:]:
        if "subject_id" not in t.columns and "subject_id_norm" in t.columns:
            t = t.rename(columns={"subject_id_norm": "subject_id"})
        k = next((x for x in key_candidates if x in t.columns), None)
        if k is None:
            raise ValueError("No trial index in an additional table")
        if k != "trial_index_numeric":
            t = t.rename(columns={k: "trial_index_numeric"})
        keys = ["subject_id", "trial_index_numeric"]
        keep = keys + [c for c in t.columns if c not in out.columns]
        out = out.merge(t[keep].drop_duplicates(keys), on=keys, how="left", validate="one_to_one")
    return canonicalize(out)


def select_features(d: pd.DataFrame, explicit: list[str] | None) -> list[str]:
    if explicit:
        missing = [c for c in explicit if c not in d.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return explicit
    return [c for c in d.columns if any(re.search(p, c, flags=re.I) for p in PATTERNS)]


def contrast(d: pd.DataFrame, feature: str, min_each: int, n_perm: int, n_boot: int, seed: int) -> dict:
    rows = []
    for sid, sd in d.groupby("subject_id"):
        a = pd.to_numeric(sd.loc[sd["outcome"].eq("nogo_fa"), feature], errors="coerce").dropna()
        b = pd.to_numeric(sd.loc[sd["outcome"].eq("go_hit"), feature], errors="coerce").dropna()
        if len(a) >= min_each and len(b) >= min_each:
            rows.append({"subject_id": sid, "difference_fa_minus_hit": float(a.mean() - b.mean()),
                         "n_fa": len(a), "n_hit": len(b)})
    rr = pd.DataFrame(rows)
    if rr.empty:
        return {"n_subjects": 0}
    vals = rr["difference_fa_minus_hit"].to_numpy()
    mean, lo, hi = bootstrap_mean_ci(vals, n_boot, seed)
    return {"n_subjects": len(rr), "mean_difference": mean, "ci95_low": lo, "ci95_high": hi,
            "signflip_p": signflip_p(vals, n_perm, seed + 1),
            "median_n_fa": float(rr["n_fa"].median()), "median_n_hit": float(rr["n_hit"].median())}



def stable_seed(base: int, *parts: object) -> int:
    material = (str(base) + "|" + "|".join(map(str, parts))).encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(material).digest()[:4], "little")
    return int((base + offset) % (2**32 - 1))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial-features", nargs="+", required=True, help="First table must contain response_marker_error_ms")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-cues", nargs="*", default=None)
    ap.add_argument("--features", nargs="*", default=None)
    ap.add_argument("--thresholds-ms", nargs="*", type=float, default=[4, 8, 12, 20, 40])
    ap.add_argument("--min-trials", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    out = ensure_dir(args.out)
    d = merge([Path(x) for x in args.trial_features])
    if args.include_cues and "cue_type" in d.columns:
        allowed = {str(x).strip().lower() for x in args.include_cues}
        d = d[d["cue_type"].astype(str).str.lower().isin(allowed)].copy()
    d = d[d["response_present"].fillna(0).astype(float).eq(1)].copy()
    features = select_features(d, args.features)
    if not features:
        raise ValueError("No response-locked features detected")

    qc = d.groupby("subject_id", as_index=False).agg(
        n_responded=("abs_marker_error_ms", "size"),
        n_with_marker=("abs_marker_error_ms", lambda x: int(x.notna().sum())),
        median_abs_error_ms=("abs_marker_error_ms", "median"),
        q95_abs_error_ms=("abs_marker_error_ms", lambda x: x.quantile(.95)),
        max_abs_error_ms=("abs_marker_error_ms", "max"),
    )
    qc["marker_coverage"] = qc["n_with_marker"] / qc["n_responded"]
    qc.to_csv(out / "response_marker_qc_by_subject.csv", index=False)

    rows = []
    filters = [("all_metadata_locked", np.inf)] + [(f"abs_error_le_{x:g}ms", x) for x in args.thresholds_ms]
    for label, threshold in filters:
        ds = d if np.isinf(threshold) else d[d["abs_marker_error_ms"].le(threshold)].copy()
        for f in features:
            r = contrast(ds, f, args.min_trials, args.n_perm, args.n_boot,
                         stable_seed(args.seed, label, f))
            rows.append({"filter": label, "threshold_ms": threshold, "feature": f,
                         "n_trials_retained": len(ds), "trial_retention": len(ds) / max(len(d), 1), **r})
    res = pd.DataFrame(rows)
    # Retained-effect ratio relative to the unrestricted estimate.
    base = res[res["filter"].eq("all_metadata_locked")][["feature", "mean_difference"]].rename(columns={"mean_difference": "base_mean_difference"})
    res = res.merge(base, on="feature", how="left")
    res["effect_ratio_vs_all"] = res["mean_difference"] / res["base_mean_difference"].replace(0, np.nan)
    res.to_csv(out / "response_marker_effect_sensitivity.csv", index=False)
    lines = [
        "# Response-marker sensitivity", "", "## Marker QC", "", qc.to_markdown(index=False), "",
        "## Effect stability", "", res.to_markdown(index=False), "", "## Decision rule", "",
        "- At 250 Hz, one sample is 4 ms. Stable estimates under 4–12 ms thresholds support the metadata-derived lock.",
        "- Large median errors, poor marker coverage, sign reversal, or strong effect inflation/attenuation requires direct-marker re-extraction.",
        "- Report this as a sensitivity check; it cannot prove equivalence of the two locking methods.",
    ]
    (out / "RESPONSE_MARKER_SENSITIVITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(res.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
