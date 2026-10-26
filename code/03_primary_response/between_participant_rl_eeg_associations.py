#!/usr/bin/env python3
"""Freedman-Lane inference for the EEG-computational study canonical reinforcement-learning
to response-signature association analysis.

The module recomputes the fixed six neural-signature by three
reinforcement-learning-predictor family using residual permutation under the
reduced nuisance-covariate model. This preserves predictor-covariate structure
when nuisance covariates are present.

For each association, the analysis reports the standardized predictor
coefficient, participant-bootstrap interval, Freedman-Lane permutation p-value,
predictor VIF, and design-matrix condition number. The primary group-adjusted
family uses 10,000 permutations, 5,000 participant bootstraps, and
Benjamini-Hochberg FDR correction across 18 tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SIGNATURES = [
    "erp_ern",
    "erp_pe",
    "tf_theta",
    "tf_beta",
    "entropy_theta",
    "entropy_beta",
]

PREDICTORS = [
    "mean_p_press",
    "mean_actor_qdiff_current",
    "mean_control_current",
]


def ensure_dir(path):
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def fdr_bh(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    back = np.empty(m)
    back[order] = adj
    q[np.where(ok)[0]] = back
    return q


def winsorize(s, lo=0.05, hi=0.95):
    x = pd.to_numeric(s, errors="coerce").astype(float)
    a, b = x.quantile([lo, hi])
    return x.clip(a, b)


def zscore_array(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        raise ValueError("Cannot standardize constant/invalid vector.")
    return (x - mu) / sd


def prepare_design(df, ycol, predictor, covariates, winsor=False):
    cols = ["subject_id", ycol, predictor] + list(covariates)
    x = df[cols].copy().dropna().reset_index(drop=True)

    if winsor:
        for c in [ycol, predictor] + [z for z in covariates if z != "group_high"]:
            x[c] = winsorize(x[c])

    y = zscore_array(pd.to_numeric(x[ycol], errors="raise").to_numpy(float))
    pred = zscore_array(pd.to_numeric(x[predictor], errors="raise").to_numpy(float))

    nuisance_parts = [np.ones(len(x), dtype=float)]
    nuisance_names = ["intercept"]

    for cov in covariates:
        raw = pd.to_numeric(x[cov], errors="raise").to_numpy(float)
        if cov == "group_high":
            vec = raw
        else:
            vec = zscore_array(raw)
        nuisance_parts.append(vec)
        nuisance_names.append(cov)

    Z = np.column_stack(nuisance_parts)
    X = np.column_stack([Z, pred])
    predictor_index = X.shape[1] - 1

    ok = (
        np.isfinite(y)
        & np.all(np.isfinite(X), axis=1)
    )
    y = y[ok]
    X = X[ok]
    Z = Z[ok]
    x = x.loc[ok].reset_index(drop=True)

    if len(y) <= X.shape[1] + 1:
        raise ValueError("Insufficient subjects after complete-case filtering.")

    return x, y, X, Z, predictor_index, nuisance_names


def ols_from_arrays(y, X, predictor_index):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = float(np.sum(resid ** 2) / dof)
    covb = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(covb), 0))
    b = float(beta[predictor_index])
    seb = float(se[predictor_index])
    t = b / seb if seb > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), dof)) if np.isfinite(t) else np.nan
    denom = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - np.sum(resid ** 2) / denom) if denom > 0 else np.nan
    return {
        "beta": b,
        "se": seb,
        "t": float(t),
        "p": p,
        "r2": r2,
        "dof": int(dof),
    }


def predictor_vif(X, predictor_index):
    p = X[:, predictor_index]
    Z = np.delete(X, predictor_index, axis=1)
    beta, *_ = np.linalg.lstsq(Z, p, rcond=None)
    resid = p - Z @ beta
    denom = np.sum((p - p.mean()) ** 2)
    if denom <= 1e-12:
        return np.inf
    r2 = 1 - np.sum(resid ** 2) / denom
    if r2 >= 1 - 1e-12:
        return np.inf
    return float(1.0 / max(1e-12, 1.0 - r2))


def condition_number(X):
    # Standardize non-intercept columns only for scale-free diagnostic.
    XX = X.copy().astype(float)
    for j in range(1, XX.shape[1]):
        sd = np.std(XX[:, j], ddof=0)
        if sd > 1e-12:
            XX[:, j] = (XX[:, j] - np.mean(XX[:, j])) / sd
    return float(np.linalg.cond(XX))


def permute_residuals(resid, groups, rng):
    out = resid.copy()
    if groups is None:
        return rng.permutation(out)

    groups = np.asarray(groups)
    for g in pd.unique(groups):
        idx = np.where(groups == g)[0]
        out[idx] = rng.permutation(out[idx])
    return out


def freedman_lane_p(
    xused,
    y,
    X,
    Z,
    predictor_index,
    observed_t,
    n_perm,
    rng,
    group_restricted,
):
    # Reduced model y ~ nuisance
    b0, *_ = np.linalg.lstsq(Z, y, rcond=None)
    fitted0 = Z @ b0
    resid0 = y - fitted0

    groups = None
    if group_restricted and "group_high" in xused.columns:
        groups = pd.to_numeric(xused["group_high"], errors="raise").to_numpy()

    ge = 0
    valid = 0

    for _ in range(int(n_perm)):
        rp = permute_residuals(resid0, groups, rng)
        ystar = fitted0 + rp
        fit = ols_from_arrays(ystar, X, predictor_index)
        if np.isfinite(fit["t"]):
            ge += abs(fit["t"]) >= abs(observed_t)
            valid += 1

    if valid == 0:
        return np.nan, 0

    return float((ge + 1) / (valid + 1)), int(valid)


def bootstrap_ci(df, ycol, predictor, covariates, winsor, n_boot, rng):
    vals = []
    n = len(df)
    for _ in range(int(n_boot)):
        b = df.iloc[rng.integers(0, n, size=n)].reset_index(drop=True)
        try:
            _, y, X, _, pi, _ = prepare_design(
                b, ycol, predictor, covariates, winsor=winsor
            )
            vals.append(ols_from_arrays(y, X, pi)["beta"])
        except Exception:
            continue

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return np.nan, np.nan, 0

    return (
        float(np.quantile(vals, 0.025)),
        float(np.quantile(vals, 0.975)),
        int(len(vals)),
    )


def loo_summary(df, ycol, predictor, covariates, winsor):
    vals = []
    for i in range(len(df)):
        z = df.drop(df.index[i]).reset_index(drop=True)
        try:
            _, y, X, _, pi, _ = prepare_design(
                z, ycol, predictor, covariates, winsor=winsor
            )
            vals.append(ols_from_arrays(y, X, pi)["beta"])
        except Exception:
            continue

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return {
            "loo_n": 0,
            "loo_same_direction_fraction": np.nan,
            "loo_min_beta": np.nan,
            "loo_max_beta": np.nan,
        }

    return {
        "loo_n": int(len(vals)),
        "loo_same_direction_fraction": np.nan,  # filled after observed sign is known
        "loo_min_beta": float(np.min(vals)),
        "loo_max_beta": float(np.max(vals)),
        "_loo_vals": vals,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    out = ensure_dir(args.out)
    df = pd.read_csv(Path(args.analysis_table).expanduser().resolve())

    required = (
        ["subject_id", "group_high", "nogo_fa_rate", "go_miss_rate", "mean_rt_ms"]
        + SIGNATURES
        + PREDICTORS
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Analysis table missing required columns: {missing}")

    specs = [
        ("primary_group", ["group_high"], False, True),
        ("sensitivity_none", [], False, False),
        (
            "sensitivity_group_behavior",
            ["group_high", "nogo_fa_rate", "go_miss_rate", "mean_rt_ms"],
            False,
            True,
        ),
        ("sensitivity_group_winsor", ["group_high"], True, True),
    ]

    rng = np.random.default_rng(args.seed)
    rows = []

    for signature in SIGNATURES:
        for predictor in PREDICTORS:
            for analysis, covs, winsor, group_restricted in specs:
                xused, y, X, Z, pi, _ = prepare_design(
                    df,
                    signature,
                    predictor,
                    covs,
                    winsor=winsor,
                )

                fit = ols_from_arrays(y, X, pi)

                perm_p, n_valid_perm = freedman_lane_p(
                    xused=xused,
                    y=y,
                    X=X,
                    Z=Z,
                    predictor_index=pi,
                    observed_t=fit["t"],
                    n_perm=args.n_perm,
                    rng=rng,
                    group_restricted=group_restricted,
                )

                blo, bhi, n_valid_boot = bootstrap_ci(
                    df,
                    signature,
                    predictor,
                    covs,
                    winsor,
                    args.n_boot,
                    rng,
                )

                loo = loo_summary(
                    df,
                    signature,
                    predictor,
                    covs,
                    winsor,
                )
                loo_vals = loo.pop("_loo_vals", np.array([]))
                if len(loo_vals):
                    loo["loo_same_direction_fraction"] = float(
                        np.mean(np.sign(loo_vals) == np.sign(fit["beta"]))
                    )

                vif = predictor_vif(X, pi)
                cond = condition_number(X)

                rows.append({
                    "signature": signature,
                    "predictor": predictor,
                    "analysis": analysis,
                    "covariates": ",".join(covs),
                    "winsorized": bool(winsor),
                    "group_restricted_permutation": bool(group_restricted),
                    "n_subjects": int(len(y)),
                    "standardized_beta": fit["beta"],
                    "standard_error": fit["se"],
                    "t_value": fit["t"],
                    "ols_p": fit["p"],
                    "bootstrap_ci_low": blo,
                    "bootstrap_ci_high": bhi,
                    "n_valid_bootstrap": n_valid_boot,
                    "freedman_lane_p": perm_p,
                    "n_valid_permutations": n_valid_perm,
                    "r2": fit["r2"],
                    "predictor_vif": vif,
                    "design_condition_number": cond,
                    "collinearity_flag":
                        bool((np.isfinite(vif) and vif >= 5.0) or
                             (np.isfinite(cond) and cond >= 30.0)),
                    **loo,
                })

    res = pd.DataFrame(rows)
    res["primary_freedman_lane_q_fdr_bh"] = np.nan
    m = res["analysis"] == "primary_group"
    res.loc[m, "primary_freedman_lane_q_fdr_bh"] = fdr_bh(
        res.loc[m, "freedman_lane_p"].values
    )

    res.to_csv(
        out / "corrected_canonical_rl_eeg_associations.csv",
        index=False,
    )

    primary = res[res["analysis"] == "primary_group"].copy()
    primary = primary.sort_values(
        ["primary_freedman_lane_q_fdr_bh", "freedman_lane_p"]
    )
    primary.to_csv(
        out / "corrected_canonical_rl_eeg_associations_primary.csv",
        index=False,
    )

    diagnostic = (
        res.groupby("analysis", as_index=False)
        .agg(
            n_rows=("signature", "size"),
            max_predictor_vif=("predictor_vif", "max"),
            max_condition_number=("design_condition_number", "max"),
            n_collinearity_flags=("collinearity_flag", "sum"),
        )
    )
    diagnostic.to_csv(
        out / "association_collinearity_diagnostics.csv",
        index=False,
    )

    settings = {
        "source_analysis_table":
            str(Path(args.analysis_table).expanduser().resolve()),
        "seed": args.seed,
        "n_permutations": args.n_perm,
        "n_bootstrap": args.n_boot,
        "primary_family_size": 18,
        "primary_analysis": "primary_group",
        "permutation_method":
            "Freedman-Lane residual permutation; residuals restricted within group strata when group_high is a nuisance covariate",
        "primary_fdr":
            "BH across fixed 6 neural signatures x 3 canonical RL predictors",
        "diagnostics":
            "predictor VIF >=5 or design condition number >=30 flagged for collinearity",
        "interpretation":
            "supportive between-subject association, not independent confirmation or causal evidence",
    }
    (out / "analysis_settings.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )

    report = [
        "# Corrected between-participant RL–EEG association inference",
        "",
        "## Primary group-adjusted results",
        "",
        primary.to_markdown(index=False),
        "",
        "## Collinearity diagnostics",
        "",
        diagnostic.to_markdown(index=False),
        "",
        "## Interpretation constraints",
        "",
        "- Primary permutation inference uses Freedman-Lane residual permutation rather than direct predictor shuffling.",
        "- Primary FDR correction covers the fixed 18-test neural-signature × RL-predictor family.",
        "- Group+behavior sensitivity rows with high VIF/condition number should not be used as evidence even if a p-value is small.",
        "- Bootstrap CIs, permutation/FDR, and leave-one-out direction should be considered jointly.",
        "- These are supportive individual-differences associations with cross-fitted RL summaries, not causal neural-mechanism tests.",
    ]
    (out / "CANONICAL_RL_EEG_ASSOCIATIONS_REPORT.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\n=== Primary corrected RL–EEG associations ===")
    print(primary.to_string(index=False))
    print("\n=== Collinearity diagnostics ===")
    print(diagnostic.to_string(index=False))
    print("\nOutput:", out)


if __name__ == "__main__":
    main()
