#!/usr/bin/env python3
"""Confound-aware EEG error analysis.

The published source analysis and current repository emphasize FA-vs-Go-hit response-
locked contrasts. That contrast mixes error status with Go/NoGo target class. This script
adds two complementary checks:

1. Within-subject matched FA-vs-Hit response-locked contrasts, matching cue, block,
   reaction time, trial position, and recent history.
2. Target/cue-locked factorial models testing target class x response occurrence with
   subject fixed effects and subject-clustered standard errors.
3. Within-subject matched NoGo-CR-vs-NoGo-FA target-locked contrasts.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common_utils import (  # noqa: E402
    add_trial_history,
    bootstrap_mean_ci,
    ensure_dir,
    fdr_bh,
    normalize_subject_id,
    signflip_p,
)



def stable_seed(base: int, *parts: object) -> int:
    material = "|".join(map(str, parts)).encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(material).digest()[:4], "little")
    return int((int(base) + offset) % (2**32 - 1))

DEFAULT_RESPONSE_FEATURES = [
    "resp_ern_fz_50_150_mean_uv", "resp_ern_fz_50_150_min_uv",
    "resp_pe_fz_200_400_mean_uv", "resp_pe_cz_200_400_mean_uv",
    "resp_theta_fz_0_200", "resp_theta_fz_0_400", "resp_theta_cz_0_400",
    "resp_beta_cz_200_500", "resp_mu_cz_m400_0", "resp_beta_cz_m400_0",
]
DEFAULT_TARGET_FEATURES = [
    "target_n2_fz_200_350_mean_uv", "target_n2_cz_200_350_mean_uv",
    "target_p300_cz_250_500_mean_uv", "target_p300_pz_250_500_mean_uv",
    "target_theta_fz_200_500", "target_theta_cz_200_500",
    "target_alpha_pz_250_500", "target_beta_cz_250_500",
]
DEFAULT_CUE_FEATURES = [
    "cue_cnv_late_cz_200_500_mean_uv", "cue_cnv_late_cz_200_500_min_uv",
    "cue_p3_cz_250_450_mean_uv", "cue_n2_fz_200_350_mean_uv",
    "cue_theta_fz_200_500", "cue_theta_cz_200_500", "cue_alpha_pz_200_500", "cue_beta_cz_200_500",
]


def load_merge(erp_path: Path, tf_path: Path | None) -> pd.DataFrame:
    erp = pd.read_csv(erp_path)
    subject_col = "subject_id_norm" if "subject_id_norm" in erp.columns else "subject_id"
    erp["subject_id"] = erp[subject_col].map(normalize_subject_id)
    if "trial_index_numeric" not in erp.columns:
        erp["trial_index_numeric"] = pd.to_numeric(erp["trial_index"], errors="coerce")
    if tf_path is None:
        return erp
    tf = pd.read_csv(tf_path)
    tsub = "subject_id_norm" if "subject_id_norm" in tf.columns else "subject_id"
    tf["subject_id"] = tf[tsub].map(normalize_subject_id)
    if "trial_index_numeric" not in tf.columns:
        tf["trial_index_numeric"] = pd.to_numeric(tf["trial_index"], errors="coerce")
    keep = ["subject_id", "trial_index_numeric"] + [c for c in tf.columns if c not in erp.columns and c not in {tsub, "trial_index"}]
    return erp.merge(tf[keep], on=["subject_id", "trial_index_numeric"], how="left")


def pooled_smd(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy()
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy()
    if len(x) < 2 or len(y) < 2:
        return np.nan
    sd = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    return float((np.mean(x) - np.mean(y)) / sd) if sd > 1e-12 else 0.0


def greedy_match(
    a: pd.DataFrame,
    b: pd.DataFrame,
    distance_cols: list[str],
    exact_cols: list[str],
    rt_col: str | None,
    rt_caliper_sd: float,
) -> list[tuple[int, int, float]]:
    pairs: list[tuple[int, int, float]] = []
    if a.empty or b.empty:
        return pairs
    exact_cols = [c for c in exact_cols if c in a.columns and c in b.columns]
    strata_a = a.groupby(exact_cols, dropna=False) if exact_cols else [((), a)]
    b_groups = {k: g for k, g in (b.groupby(exact_cols, dropna=False) if exact_cols else [((), b)])}

    for key, ga in strata_a:
        gb = b_groups.get(key)
        if gb is None or gb.empty:
            continue
        cols = [c for c in distance_cols if c in ga.columns and c in gb.columns]
        pooled = pd.concat([ga[cols], gb[cols]], axis=0).apply(pd.to_numeric, errors="coerce") if cols else pd.DataFrame(index=pd.concat([ga, gb]).index)
        mu = pooled.mean()
        sd = pooled.std().replace(0, 1).fillna(1)
        Za = ((ga[cols].apply(pd.to_numeric, errors="coerce") - mu) / sd).fillna(0).to_numpy() if cols else np.zeros((len(ga), 1))
        Zb = ((gb[cols].apply(pd.to_numeric, errors="coerce") - mu) / sd).fillna(0).to_numpy() if cols else np.zeros((len(gb), 1))
        candidates = []
        rt_idx = cols.index(rt_col) if rt_col and rt_col in cols else None
        for i in range(len(ga)):
            for j in range(len(gb)):
                if rt_idx is not None and abs(Za[i, rt_idx] - Zb[j, rt_idx]) > rt_caliper_sd:
                    continue
                dist = float(np.sqrt(np.sum((Za[i] - Zb[j]) ** 2)))
                candidates.append((dist, i, j))
        candidates.sort()
        used_a, used_b = set(), set()
        for dist, i, j in candidates:
            if i in used_a or j in used_b:
                continue
            used_a.add(i); used_b.add(j)
            pairs.append((int(ga.index[i]), int(gb.index[j]), dist))
    return pairs


def matched_analysis(
    df: pd.DataFrame,
    outcome_a: str,
    outcome_b: str,
    features: list[str],
    distance_cols: list[str],
    exact_cols: list[str],
    rt_col: str | None,
    rt_caliper_sd: float,
    n_perm: int,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows, subject_rows, balance_rows = [], [], []
    for sid, g in df.groupby("subject_id"):
        a = g[g["outcome"].eq(outcome_a)].copy()
        b = g[g["outcome"].eq(outcome_b)].copy()
        pairs = greedy_match(a, b, distance_cols, exact_cols, rt_col, rt_caliper_sd)
        if not pairs:
            continue
        for ia, ib, dist in pairs:
            row = {"subject_id": sid, "outcome_a": outcome_a, "outcome_b": outcome_b, "index_a": ia, "index_b": ib, "match_distance": dist}
            for c in distance_cols:
                if c in df.columns:
                    row[f"a__{c}"] = df.at[ia, c]
                    row[f"b__{c}"] = df.at[ib, c]
            for feat in features:
                if feat in df.columns:
                    va = pd.to_numeric(pd.Series([df.at[ia, feat]]), errors="coerce").iloc[0]
                    vb = pd.to_numeric(pd.Series([df.at[ib, feat]]), errors="coerce").iloc[0]
                    row[f"diff__{feat}"] = va - vb if np.isfinite(va) and np.isfinite(vb) else np.nan
            pair_rows.append(row)

        pm = pd.DataFrame([r for r in pair_rows if r["subject_id"] == sid and r["outcome_a"] == outcome_a and r["outcome_b"] == outcome_b])
        for feat in features:
            col = f"diff__{feat}"
            if col in pm.columns and pm[col].notna().sum() >= 3:
                subject_rows.append({"subject_id": sid, "contrast": f"{outcome_a}_minus_{outcome_b}", "feature": feat,
                                     "n_pairs": int(pm[col].notna().sum()), "mean_paired_difference": float(pm[col].mean())})

        # Balance within this subject before and after matching.
        idx_a = [p[0] for p in pairs]; idx_b = [p[1] for p in pairs]
        for cov in distance_cols:
            if cov not in df.columns:
                continue
            balance_rows.append({
                "subject_id": sid, "contrast": f"{outcome_a}_minus_{outcome_b}", "covariate": cov,
                "smd_before": pooled_smd(a[cov], b[cov]),
                "smd_after": pooled_smd(df.loc[idx_a, cov], df.loc[idx_b, cov]),
                "n_pairs": len(pairs),
            })

    pairs_df = pd.DataFrame(pair_rows)
    subj_df = pd.DataFrame(subject_rows)
    balance_df = pd.DataFrame(balance_rows)
    summary_rows = []
    if not subj_df.empty:
        for (contrast, feat), g in subj_df.groupby(["contrast", "feature"]):
            vals = g["mean_paired_difference"].dropna().to_numpy()
            if len(vals) < 10:
                continue
            mean, lo, hi = bootstrap_mean_ci(vals, n_boot, stable_seed(seed, contrast, feat))
            summary_rows.append({
                "contrast": contrast, "feature": feat, "n_subjects": len(vals), "total_pairs": int(g["n_pairs"].sum()),
                "mean_paired_difference": mean, "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
                "signflip_p": signflip_p(vals, n_perm, stable_seed(seed, feat)),
                "positive_subjects": int(np.sum(vals > 0)),
            })
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["q_fdr_bh"] = fdr_bh(summary["signflip_p"])
        summary = summary.sort_values(["q_fdr_bh", "signflip_p"])
    return pairs_df, summary, balance_df


def factorial_models(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    d = df.copy()
    d["cue_type"] = d["cue_type"].astype(str)
    for feat in features:
        if feat not in d.columns:
            continue
        x = d[[feat, "subject_id", "is_nogo", "response_present_numeric", "cue_type", "prev_error", "trial_progress"]].copy()
        x[feat] = pd.to_numeric(x[feat], errors="coerce")
        x = x.dropna()
        if len(x) < 200 or x["subject_id"].nunique() < 20:
            continue
        formula = f"Q('{feat}') ~ is_nogo * response_present_numeric + C(cue_type) + prev_error + trial_progress + C(subject_id)"
        try:
            fit = smf.ols(formula, data=x).fit(cov_type="cluster", cov_kwds={"groups": x["subject_id"]})
            term = "is_nogo:response_present_numeric"
            rows.append({
                "feature": feat, "n_trials": len(x), "n_subjects": x["subject_id"].nunique(),
                "interaction_beta": float(fit.params.get(term, np.nan)),
                "interaction_se_cluster": float(fit.bse.get(term, np.nan)),
                "interaction_p_cluster": float(fit.pvalues.get(term, np.nan)),
                "is_nogo_beta": float(fit.params.get("is_nogo", np.nan)),
                "response_beta": float(fit.params.get("response_present_numeric", np.nan)),
                "r_squared": float(fit.rsquared),
            })
        except Exception as exc:
            rows.append({"feature": feat, "error": str(exc)})
    out = pd.DataFrame(rows)
    if not out.empty and "interaction_p_cluster" in out.columns:
        out["interaction_q_fdr_bh"] = fdr_bh(out["interaction_p_cluster"])
        out = out.sort_values(["interaction_q_fdr_bh", "interaction_p_cluster"], na_position="last")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--erp-features", required=True)
    ap.add_argument("--tf-features", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-cues", nargs="*", default=None, help="Optional cue labels; use 'up down' for the confirmatory ERP analysis")
    ap.add_argument("--block-size", type=int, default=95, help="Set 0 if the task was not divided into 4x95-trial blocks")
    ap.add_argument("--rt-caliper-sd", type=float, default=0.50)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out = ensure_dir(args.out)
    d = load_merge(Path(args.erp_features), Path(args.tf_features) if args.tf_features else None)
    d = add_trial_history(d, "subject_id", "trial_index_numeric", block_size=args.block_size)
    d["cue_type"] = d["cue_type"].astype(str).str.lower()
    if args.include_cues:
        allowed = {str(x).strip().lower() for x in args.include_cues}
        d = d[d["cue_type"].isin(allowed)].copy()
    d["log_rt"] = np.log(pd.to_numeric(d.get("rt_response_from_target_ms"), errors="coerce").where(lambda x: x > 0))

    response_features = [c for c in DEFAULT_RESPONSE_FEATURES if c in d.columns]
    target_features = [c for c in DEFAULT_TARGET_FEATURES if c in d.columns]
    cue_features = [c for c in DEFAULT_CUE_FEATURES if c in d.columns]

    fa_hit_pairs, fa_hit_summary, fa_hit_balance = matched_analysis(
        d[d["outcome"].isin(["nogo_fa", "go_hit"])], "nogo_fa", "go_hit", response_features,
        distance_cols=["log_rt", "trial_progress", "prev_error", "prev_press"],
        exact_cols=["subject_id", "cue_type", "block"], rt_col="log_rt", rt_caliper_sd=args.rt_caliper_sd,
        n_perm=args.n_perm, n_boot=args.n_bootstrap, seed=args.seed,
    )
    cr_fa_pairs, cr_fa_summary, cr_fa_balance = matched_analysis(
        d[d["outcome"].isin(["nogo_cr", "nogo_fa"])], "nogo_cr", "nogo_fa", target_features + cue_features,
        distance_cols=["trial_progress", "prev_error", "prev_press"],
        exact_cols=["subject_id", "cue_type", "block"], rt_col=None, rt_caliper_sd=args.rt_caliper_sd,
        n_perm=args.n_perm, n_boot=args.n_bootstrap, seed=args.seed + 1,
    )
    factorial = factorial_models(d, target_features + cue_features)

    fa_hit_pairs.to_csv(out / "matched_fa_hit_pairs.csv.gz", index=False, compression="gzip")
    fa_hit_summary.to_csv(out / "matched_fa_hit_effects.csv", index=False)
    fa_hit_balance.to_csv(out / "matched_fa_hit_balance.csv", index=False)
    cr_fa_pairs.to_csv(out / "matched_cr_fa_pairs.csv.gz", index=False, compression="gzip")
    cr_fa_summary.to_csv(out / "matched_cr_fa_effects.csv", index=False)
    cr_fa_balance.to_csv(out / "matched_cr_fa_balance.csv", index=False)
    factorial.to_csv(out / "target_cue_factorial_models.csv", index=False)

    balance_summary = pd.concat([fa_hit_balance, cr_fa_balance], ignore_index=True)
    if not balance_summary.empty:
        balance_summary = balance_summary.groupby(["contrast", "covariate"], as_index=False).agg(
            n_subjects=("subject_id", "nunique"), median_abs_smd_before=("smd_before", lambda x: float(np.nanmedian(np.abs(x)))),
            median_abs_smd_after=("smd_after", lambda x: float(np.nanmedian(np.abs(x)))), total_pairs=("n_pairs", "sum"))
    balance_summary.to_csv(out / "matching_balance_summary.csv", index=False)

    lines = [
        "# Confound-aware matched and factorial EEG analysis", "", "## Matched FA minus Go-hit response effects", "",
        fa_hit_summary.to_markdown(index=False) if not fa_hit_summary.empty else "No eligible effects.", "",
        "## Matched NoGo CR minus NoGo FA target/cue effects", "",
        cr_fa_summary.to_markdown(index=False) if not cr_fa_summary.empty else "No eligible effects.", "",
        "## Factorial target x response interaction", "",
        factorial.to_markdown(index=False) if not factorial.empty else "No eligible models.", "",
        "## Matching balance", "", balance_summary.to_markdown(index=False) if not balance_summary.empty else "No balance rows.", "",
        "## Decision rule", "",
        "A response-locked error claim is strongest when FA-vs-Hit effects survive RT/history matching and target/cue-locked analyses show a compatible target x response interaction or NoGo CR-vs-FA difference.",
    ]
    (out / "MATCHED_FACTORIAL_EEG_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(fa_hit_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
