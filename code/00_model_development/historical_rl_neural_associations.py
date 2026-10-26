#!/usr/bin/env python3
# FORMALIZED HISTORICAL SOURCE SNAPSHOT
# Evidence status: HISTORICAL_OR_SUPERSEDED_DEVELOPMENT_EVIDENCE
# This file preserves a historical project stage with public-facing terminology standardized for release.
# The SHA-256 of the original executed/development source is recorded in the corresponding provenance registry.

"""
Robust subject-level RL–theta association analysis for the EEG-computational study.

Why this script exists
----------------------
The subject-level moderation screen found that the strongest corrected
individual-differences association was:

    FA-minus-Hit response theta at Fz, 0-200 ms
        vs
    mean model-derived control_current

This script validates that association more carefully with:
1. standardized partial regressions,
2. covariate-adjusted effects controlling group and behavior rates,
3. leave-one-subject-out stability,
4. permutation p-values,
5. bootstrap confidence intervals,
6. winsorized sensitivity.

This should be treated as a subject-level neural-computational association analysis, not as
causal evidence.

Inputs
------
- subject_neural_signature_table.csv from:
  results/eeg_subject_neural_signature_moderation_v1/

Outputs
-------
- robust_rl_theta_association_model_results.csv
- robust_rl_theta_association_leave_one_out.csv
- robust_rl_theta_association_bootstrap.csv
- ROBUST_RL_THETA_ASSOCIATION_REPORT.md

Example
-------
python code/final/historical_rl_neural_associations.py \
  --subject_table results/eeg_subject_neural_signature_moderation_v1/subject_neural_signature_table.csv \
  --out results/model_development/historical_rl_neural_associations \
  --n_perm 10000 \
  --n_boot 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


PRIMARY_TARGETS = [
    "sig_error_fa_minus_hit_resp_theta_fz_0_200",
    "sig_error_fa_minus_hit_resp_theta_fz_0_400",
    "sig_error_fa_minus_hit_resp_theta_cz_0_400",
    "composite_error_monitoring_signature_z",
    "sig_error_fa_minus_hit_resp_beta_cz_200_500",
]

PRIMARY_PREDICTORS = [
    "mean_control_current",
    "mean_control_current_go",
    "mean_control_current_nogo",
    "mean_actor_qdiff_current",
    "mean_actor_qdiff_current_go",
    "mean_actor_qdiff_current_nogo",
    "mean_p_press",
    "mean_p_press_go",
    "mean_p_press_nogo",
]

COVARIATE_SETS = {
    "none": [],
    "group": ["group"],
    "behavior": ["nogo_fa_rate", "go_miss_rate", "press_rate", "mean_rt_ms"],
    "group_plus_behavior": ["group", "nogo_fa_rate", "go_miss_rate", "press_rate", "mean_rt_ms"],
    "group_plus_behavior_counts": ["group", "nogo_fa_rate", "go_miss_rate", "press_rate", "mean_rt_ms", "n_nogo_fa", "n_go_hit"],
}


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
    tmp = np.empty(n)
    tmp[order] = q
    out[ok] = tmp
    return out


def winsorize_series(x: pd.Series, limits: Tuple[float, float] = (0.05, 0.95)) -> pd.Series:
    y = pd.to_numeric(x, errors="coerce").copy()
    lo, hi = y.quantile(limits[0]), y.quantile(limits[1])
    return y.clip(lower=lo, upper=hi)


def build_design(df: pd.DataFrame, predictor: str, covariates: List[str], winsorize: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    cols = [predictor] + [c for c in covariates if c in df.columns]
    data = df[["subject_id", "group"] + list(set(cols))].copy()

    parts = []
    names = []

    # Predictor first.
    x = pd.to_numeric(df[predictor], errors="coerce")
    if winsorize:
        x = winsorize_series(x)
    parts.append(x.rename(predictor))
    names.append(predictor)

    # Covariates.
    for cov in covariates:
        if cov not in df.columns:
            continue
        if cov == predictor:
            continue
        if cov == "group":
            g = df[cov].astype(str).str.lower().map({"high": 1.0, "low": 0.0})
            parts.append(g.rename("group_high"))
            names.append("group_high")
        elif pd.api.types.is_numeric_dtype(df[cov]):
            z = pd.to_numeric(df[cov], errors="coerce")
            if winsorize:
                z = winsorize_series(z)
            parts.append(z.rename(cov))
            names.append(cov)
        else:
            dummies = pd.get_dummies(df[cov].astype(str), prefix=cov, drop_first=True, dtype=float)
            for c in dummies.columns:
                parts.append(dummies[c].rename(c))
                names.append(c)

    Xraw = pd.concat(parts, axis=1)
    # Standardize non-intercept columns.
    Xstd = Xraw.copy()
    for c in Xstd.columns:
        s = pd.to_numeric(Xstd[c], errors="coerce")
        sd = s.std(skipna=True)
        if pd.notna(sd) and sd > 0:
            Xstd[c] = (s - s.mean(skipna=True)) / sd
        else:
            Xstd[c] = np.nan

    return Xstd.values.astype(float), None, names, Xstd


def ols_fit(y: np.ndarray, X_no_intercept: np.ndarray) -> Dict[str, np.ndarray]:
    mask = np.isfinite(y) & np.isfinite(X_no_intercept).all(axis=1)
    n = int(mask.sum())
    p = X_no_intercept.shape[1] + 1
    if n <= p + 1:
        return {"ok": False}

    X = np.column_stack([np.ones(n), X_no_intercept[mask]])
    yy = y[mask]
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    df = n - X.shape[1]
    if df <= 0:
        return {"ok": False}

    sigma2 = float((resid @ resid) / df)
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    tvals = beta / se
    pvals = 2 * stats.t.sf(np.abs(tvals), df=df)

    yhat = X @ beta
    ss_total = float(np.sum((yy - yy.mean()) ** 2))
    ss_resid = float(np.sum(resid ** 2))
    r2 = 1 - ss_resid / ss_total if ss_total > 0 else np.nan

    return {
        "ok": True,
        "mask": mask,
        "n": n,
        "df": df,
        "beta": beta,
        "se": se,
        "t": tvals,
        "p": pvals,
        "r2": r2,
        "resid": resid,
        "yhat": yhat,
    }


def standardized_target(y: pd.Series, winsorize: bool = False) -> pd.Series:
    z = pd.to_numeric(y, errors="coerce")
    if winsorize:
        z = winsorize_series(z)
    sd = z.std(skipna=True)
    if pd.notna(sd) and sd > 0:
        return (z - z.mean(skipna=True)) / sd
    return z * np.nan


def permutation_p_for_predictor(y: np.ndarray, Xstd: pd.DataFrame, predictor: str, observed_beta: float, n_perm: int, rng: np.random.Generator) -> float:
    if n_perm <= 0:
        return np.nan

    cols = list(Xstd.columns)
    if predictor not in cols:
        return np.nan
    pred_idx = cols.index(predictor)

    mask = np.isfinite(y) & np.isfinite(Xstd.values).all(axis=1)
    if int(mask.sum()) < 10:
        return np.nan

    y_obs = y.copy()
    X = Xstd.values.copy()
    null = []

    # Permute predictor relative to y/covariates; preserves predictor distribution.
    for _ in range(n_perm):
        Xp = X.copy()
        vals = Xp[mask, pred_idx].copy()
        rng.shuffle(vals)
        Xp[mask, pred_idx] = vals
        fit = ols_fit(y_obs, Xp)
        if fit.get("ok", False):
            null.append(fit["beta"][pred_idx + 1])

    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if len(null) == 0:
        return np.nan

    return float((np.sum(np.abs(null) >= abs(observed_beta)) + 1) / (len(null) + 1))


def bootstrap_ci(df: pd.DataFrame, target: str, predictor: str, covariates: List[str], n_boot: int, rng: np.random.Generator, winsorize: bool = False) -> Dict[str, float]:
    if n_boot <= 0:
        return {"boot_mean": np.nan, "boot_ci_low": np.nan, "boot_ci_high": np.nan, "boot_n": 0}

    vals = []
    n = len(df)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bdf = df.iloc[idx].reset_index(drop=True)
        y = standardized_target(bdf[target], winsorize=winsorize).values.astype(float)
        X, _, names, Xstd = build_design(bdf, predictor, covariates, winsorize=winsorize)
        if predictor not in names:
            continue
        pred_idx = names.index(predictor) + 1
        fit = ols_fit(y, X)
        if fit.get("ok", False):
            vals.append(fit["beta"][pred_idx])

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"boot_mean": np.nan, "boot_ci_low": np.nan, "boot_ci_high": np.nan, "boot_n": 0}

    return {
        "boot_mean": float(np.mean(vals)),
        "boot_ci_low": float(np.quantile(vals, 0.025)),
        "boot_ci_high": float(np.quantile(vals, 0.975)),
        "boot_n": int(len(vals)),
    }


def run_models(df: pd.DataFrame, n_perm: int, n_boot: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    loo_rows = []
    boot_rows = []

    for target in PRIMARY_TARGETS:
        if target not in df.columns:
            continue

        for predictor in PRIMARY_PREDICTORS:
            if predictor not in df.columns:
                continue

            for cov_name, covariates in COVARIATE_SETS.items():
                for sensitivity in ["standard", "winsorized"]:
                    use_winsor = sensitivity == "winsorized"

                    y = standardized_target(df[target], winsorize=use_winsor).values.astype(float)
                    X, _, names, Xstd = build_design(df, predictor, covariates, winsorize=use_winsor)

                    if predictor not in names:
                        continue
                    pred_idx = names.index(predictor) + 1

                    fit = ols_fit(y, X)
                    if not fit.get("ok", False):
                        continue

                    beta = float(fit["beta"][pred_idx])
                    se = float(fit["se"][pred_idx])
                    tval = float(fit["t"][pred_idx])
                    pval = float(fit["p"][pred_idx])

                    perm_p = permutation_p_for_predictor(
                        y=y,
                        Xstd=Xstd,
                        predictor=predictor,
                        observed_beta=beta,
                        n_perm=n_perm,
                        rng=rng,
                    )

                    boot = bootstrap_ci(
                        df=df,
                        target=target,
                        predictor=predictor,
                        covariates=covariates,
                        n_boot=n_boot,
                        rng=rng,
                        winsorize=use_winsor,
                    )
                    boot_row = {
                        "target": target,
                        "predictor": predictor,
                        "covariate_set": cov_name,
                        "sensitivity": sensitivity,
                        **boot,
                    }
                    boot_rows.append(boot_row)

                    rows.append({
                        "target": target,
                        "predictor": predictor,
                        "covariate_set": cov_name,
                        "sensitivity": sensitivity,
                        "n_subjects": int(fit["n"]),
                        "standardized_beta": beta,
                        "standard_error": se,
                        "t_value": tval,
                        "p_value": pval,
                        "permutation_p": perm_p,
                        "r2": float(fit["r2"]),
                        "boot_mean": boot["boot_mean"],
                        "boot_ci_low": boot["boot_ci_low"],
                        "boot_ci_high": boot["boot_ci_high"],
                        "boot_n": boot["boot_n"],
                        "covariates": ",".join([c for c in covariates if c in df.columns]),
                    })

                    # Leave-one-out stability.
                    loo_betas = []
                    for i in range(len(df)):
                        ldf = df.drop(df.index[i]).reset_index(drop=True)
                        yy = standardized_target(ldf[target], winsorize=use_winsor).values.astype(float)
                        XX, _, nn, _ = build_design(ldf, predictor, covariates, winsorize=use_winsor)
                        if predictor not in nn:
                            continue
                        pp_idx = nn.index(predictor) + 1
                        f2 = ols_fit(yy, XX)
                        if f2.get("ok", False):
                            b = float(f2["beta"][pp_idx])
                            loo_betas.append(b)
                            loo_rows.append({
                                "target": target,
                                "predictor": predictor,
                                "covariate_set": cov_name,
                                "sensitivity": sensitivity,
                                "left_out_subject": df.iloc[i]["subject_id"] if "subject_id" in df.columns else i,
                                "standardized_beta": b,
                            })

    results = pd.DataFrame(rows)
    if not results.empty:
        # FDR over standard sensitivity only, but store a global q too.
        results["q_value_fdr_bh"] = fdr_bh(results["p_value"].values)
        results["permutation_q_fdr_bh"] = fdr_bh(results["permutation_p"].values)
        results = results.sort_values(["permutation_q_fdr_bh", "q_value_fdr_bh", "p_value"]).reset_index(drop=True)

    loo = pd.DataFrame(loo_rows)
    boot_df = pd.DataFrame(boot_rows)

    return results, loo, boot_df


def summarize_loo(loo: pd.DataFrame) -> pd.DataFrame:
    if loo.empty:
        return pd.DataFrame()

    rows = []
    for keys, g in loo.groupby(["target", "predictor", "covariate_set", "sensitivity"]):
        vals = pd.to_numeric(g["standardized_beta"], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        rows.append({
            "target": keys[0],
            "predictor": keys[1],
            "covariate_set": keys[2],
            "sensitivity": keys[3],
            "n_leave_one_out": int(len(vals)),
            "loo_mean_beta": float(np.mean(vals)),
            "loo_min_beta": float(np.min(vals)),
            "loo_max_beta": float(np.max(vals)),
            "loo_same_direction_fraction": float(max(np.mean(vals > 0), np.mean(vals < 0))),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["loo_same_direction_fraction", "n_leave_one_out"], ascending=[False, False]).reset_index(drop=True)
    return out


def write_report(out_dir: Path, results: pd.DataFrame, loo_summary: pd.DataFrame, n_perm: int, n_boot: int) -> None:
    lines = []
    lines.append("# Robust RL–theta association report")
    lines.append("")
    lines.append("This report validates subject-level associations between RL/DDM latent summaries and neural error-monitoring signatures.")
    lines.append("")
    lines.append("## Analysis settings")
    lines.append("")
    lines.append(f"- Permutations per model: {n_perm}")
    lines.append(f"- Bootstrap samples per model: {n_boot}")
    lines.append("")
    lines.append("## Top regression results")
    lines.append("")
    lines.append(f"- Tested rows: {len(results)}")
    if not results.empty:
        lines.append(f"- OLS FDR q < 0.05 rows: {int((results['q_value_fdr_bh'] < 0.05).sum())}")
        lines.append(f"- Permutation FDR q < 0.05 rows: {int((results['permutation_q_fdr_bh'] < 0.05).sum())}")
        lines.append("")
        lines.append("| Target | Predictor | Covariates | Sensitivity | n | beta | 95% boot CI | p | perm p | perm q | R² |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in results.head(40).iterrows():
            lines.append(
                f"| {row['target']} | {row['predictor']} | {row['covariate_set']} | {row['sensitivity']} | "
                f"{int(row['n_subjects'])} | {row['standardized_beta']:.4f} | "
                f"[{row['boot_ci_low']:.4f}, {row['boot_ci_high']:.4f}] | "
                f"{row['p_value']:.3g} | {row['permutation_p']:.4g} | {row['permutation_q_fdr_bh']:.4g} | "
                f"{row['r2']:.3f} |"
            )

    lines.append("")
    lines.append("## Leave-one-subject-out stability")
    lines.append("")
    if loo_summary.empty:
        lines.append("- No leave-one-out rows were generated.")
    else:
        lines.append("| Target | Predictor | Covariates | Sensitivity | n | mean beta | min beta | max beta | same-direction fraction |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
        for _, row in loo_summary.head(40).iterrows():
            lines.append(
                f"| {row['target']} | {row['predictor']} | {row['covariate_set']} | {row['sensitivity']} | "
                f"{int(row['n_leave_one_out'])} | {row['loo_mean_beta']:.4f} | "
                f"{row['loo_min_beta']:.4f} | {row['loo_max_beta']:.4f} | "
                f"{row['loo_same_direction_fraction']:.2f} |"
            )

    lines.append("")
    lines.append("## Interpretation guardrail")
    lines.append("")
    lines.append("- This is a subject-level association analysis with about 38 usable subjects; it should be framed as supportive association evidence.")
    lines.append("- Effects that survive group/behavior covariates, permutation tests, bootstrap CIs, winsorization, and leave-one-out stability are the safest.")
    lines.append("- Because latent states may be derived from model fits rather than an entirely held-out latent-estimation pipeline, avoid causal or fully leakage-free prediction wording.")
    lines.append("")
    (out_dir / "ROBUST_RL_THETA_ASSOCIATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_table", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    subject_table_path = Path(args.subject_table).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(subject_table_path)

    results, loo, boot = run_models(
        df=df,
        n_perm=int(args.n_perm),
        n_boot=int(args.n_boot),
        seed=int(args.seed),
    )
    loo_summary = summarize_loo(loo)

    results.to_csv(out_dir / "robust_rl_theta_association_model_results.csv", index=False)
    loo.to_csv(out_dir / "robust_rl_theta_association_leave_one_out.csv", index=False)
    loo_summary.to_csv(out_dir / "robust_rl_theta_association_leave_one_out_summary.csv", index=False)
    boot.to_csv(out_dir / "robust_rl_theta_association_bootstrap.csv", index=False)

    write_report(out_dir, results, loo_summary, int(args.n_perm), int(args.n_boot))

    print(f"Wrote outputs to: {out_dir}")
    if not results.empty:
        print(results.head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
