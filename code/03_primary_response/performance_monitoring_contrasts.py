#!/usr/bin/env python3
"""Canonical response-locked performance-monitoring signatures and reinforcement-learning
association analysis for EEG-computational study.

The module separates response-locked false-alarm-minus-Go-hit contrasts for a
fixed ERP/time-frequency/TF-derived feature set from between-participant
associations between those response signatures and cross-fitted
reinforcement-learning summaries.

Reinforcement-learning variables are taken from the canonical nested latent
table rather than from embedded columns in EEG-derived tables. The association
analyses are cross-sectional and are not interpreted as causal or independent
confirmation.

The response-signature extraction workflow also computes a historical direct
predictor-permutation association analysis for provenance and diagnostic continuity. That
direct-permutation association analysis is not used for manuscript inference. Corrected
group-adjusted RL-response inference is implemented separately with
Freedman-Lane residual permutation in
between_participant_rl_eeg_associations.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PRIMARY_FEATURES = {
    "erp_ern": ("ERP", "resp_ern_fz_50_150_mean_uv"),
    "erp_pe": ("ERP", "resp_pe_cz_200_400_mean_uv"),
    "tf_theta": ("TF", "resp_theta_fz_0_200"),
    "tf_beta": ("TF", "resp_beta_cz_200_500"),
    "entropy_theta": ("ENTROPY", "resp_global_theta_fraction"),
    "entropy_beta": ("ENTROPY", "resp_global_beta_fraction"),
}

RL_PREDICTORS = [
    "mean_p_press",
    "mean_actor_qdiff_current",
    "mean_control_current",
]


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
    adj = np.clip(adj, 0, 1)
    back = np.empty(m)
    back[order] = adj
    q[np.where(ok)[0]] = back
    return q


def ensure_dir(path):
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def zscore(s):
    x = pd.to_numeric(s, errors="coerce").astype(float)
    mu = x.mean()
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.nan, index=x.index)
    return (x - mu) / sd


def winsorize(s, lo=0.05, hi=0.95):
    x = pd.to_numeric(s, errors="coerce").astype(float)
    a, b = x.quantile([lo, hi])
    return x.clip(a, b)


def group_high(s):
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        vals = sorted(x.dropna().unique())
        if len(vals) <= 2:
            return x.astype(float)
    t = s.astype(str).str.strip().str.lower()
    return t.str.contains("high").astype(float)


def load_source(path, label):
    d = pd.read_csv(Path(path).expanduser().resolve())
    required = ["subject_id", "trial_index"]
    miss = [c for c in required if c not in d.columns]
    if miss:
        raise ValueError(f"{label}: missing {miss}")
    if d.duplicated(["subject_id", "trial_index"]).any():
        raise ValueError(f"{label}: duplicate subject/trial rows")
    d = d.copy()
    d["_canonical_trial_index"] = (
        pd.to_numeric(d["trial_index"], errors="raise").astype(int) - 1
    )
    return d


def select_block(d, label):
    cols = ["subject_id", "_canonical_trial_index"]
    chosen = []
    for key, (src, col) in PRIMARY_FEATURES.items():
        if src == label:
            if col not in d.columns:
                raise ValueError(f"{label}: required feature missing: {col}")
            chosen.append(col)
    if label == "ERP":
        if "rt_response_from_target_ms" not in d.columns:
            raise ValueError("ERP: missing rt_response_from_target_ms")
        chosen.append("rt_response_from_target_ms")
    return d[cols + chosen].copy()


def build_trial_table(erp, tf, ent, lat):
    required_lat = [
        "subject_id", "group", "trial_index", "is_nogo", "action_press",
        "p_press", "actor_qdiff_current", "control_current",
        "nested_rl_seed", "nested_rl_variant", "nested_rl_source",
    ]
    miss = [c for c in required_lat if c not in lat.columns]
    if miss:
        raise ValueError(f"LATENTS missing {miss}")
    if lat.duplicated(["subject_id", "trial_index"]).any():
        raise ValueError("LATENTS: duplicate subject/trial rows")

    d = lat[required_lat].copy()
    d["trial_index"] = pd.to_numeric(d["trial_index"], errors="raise").astype(int)

    for label, src in [("ERP", erp), ("TF", tf), ("ENTROPY", ent)]:
        block = select_block(src, label).rename(
            columns={"_canonical_trial_index": "trial_index"}
        )
        d = d.merge(
            block,
            on=["subject_id", "trial_index"],
            how="left",
            validate="one_to_one",
        )

    # Outcome labels from canonical behavior, never from EEG embedded columns.
    nogo = pd.to_numeric(d["is_nogo"], errors="raise").astype(int)
    press = pd.to_numeric(d["action_press"], errors="raise").astype(int)
    d["outcome"] = np.select(
        [
            (nogo == 1) & (press == 1),
            (nogo == 0) & (press == 1),
            (nogo == 1) & (press == 0),
            (nogo == 0) & (press == 0),
        ],
        ["FA", "Hit", "CR", "Miss"],
        default="unknown",
    )

    # QC: selected response features should be available for FA/Hit rows.
    response = d["outcome"].isin(["FA", "Hit"])
    qc = []
    for key, (_, col) in PRIMARY_FEATURES.items():
        qc.append({
            "signature": key,
            "feature": col,
            "response_rows": int(response.sum()),
            "missing_fraction_response":
                float(d.loc[response, col].isna().mean()),
        })

    return d, pd.DataFrame(qc)


def subject_contrasts(d, min_trials):
    rows = []
    for sid, g in d.groupby("subject_id", sort=False):
        fa = g[g["outcome"] == "FA"]
        hit = g[g["outcome"] == "Hit"]
        for signature, (_, col) in PRIMARY_FEATURES.items():
            a = pd.to_numeric(fa[col], errors="coerce").dropna()
            b = pd.to_numeric(hit[col], errors="coerce").dropna()
            if len(a) < min_trials or len(b) < min_trials:
                continue
            rows.append({
                "subject_id": str(sid),
                "signature": signature,
                "feature": col,
                "n_fa": int(len(a)),
                "n_hit": int(len(b)),
                "fa_mean": float(a.mean()),
                "hit_mean": float(b.mean()),
                "fa_minus_hit": float(a.mean() - b.mean()),
            })
    return pd.DataFrame(rows)


def signflip_p(vals, n_perm, rng):
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    obs = abs(np.mean(x))
    ge = 0
    for _ in range(int(n_perm)):
        signs = rng.choice([-1.0, 1.0], size=len(x))
        ge += abs(np.mean(x * signs)) >= obs
    return float((ge + 1) / (n_perm + 1))


def boot_mean_ci(vals, n_boot, rng):
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    boot = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        boot[i] = np.mean(rng.choice(x, size=len(x), replace=True))
    return (
        float(np.mean(x)),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def outcome_summary(d, thresholds, n_perm, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows = []
    contrast_tables = {}
    for th in thresholds:
        c = subject_contrasts(d, th)
        contrast_tables[th] = c
        for signature, g in c.groupby("signature", sort=False):
            vals = g["fa_minus_hit"].to_numpy(float)
            mean, lo, hi = boot_mean_ci(vals, n_boot, rng)
            p = signflip_p(vals, n_perm, rng)
            rows.append({
                "min_trials": int(th),
                "signature": signature,
                "feature": g["feature"].iloc[0],
                "n_subjects": int(g["subject_id"].nunique()),
                "mean_fa_minus_hit": mean,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "signflip_p": p,
                "positive_subjects": int((vals > 0).sum()),
                "negative_subjects": int((vals < 0).sum()),
            })

    out = pd.DataFrame(rows)
    # Primary multiplicity correction is only the fixed six-feature family at min=10.
    out["primary_signflip_q_fdr_bh"] = np.nan
    m = out["min_trials"] == 10
    out.loc[m, "primary_signflip_q_fdr_bh"] = fdr_bh(
        out.loc[m, "signflip_p"].values
    )
    return out, contrast_tables


def subject_summaries(d):
    rows = []
    for sid, g in d.groupby("subject_id", sort=False):
        nogo = pd.to_numeric(g["is_nogo"]).astype(int) == 1
        go = ~nogo
        press = pd.to_numeric(g["action_press"]).astype(int) == 1
        rt = pd.to_numeric(g["rt_response_from_target_ms"], errors="coerce")

        rows.append({
            "subject_id": str(sid),
            "group": g["group"].iloc[0],
            "group_high": float(group_high(g["group"]).iloc[0]),
            "mean_p_press": float(pd.to_numeric(g["p_press"]).mean()),
            "mean_actor_qdiff_current":
                float(pd.to_numeric(g["actor_qdiff_current"]).mean()),
            "mean_control_current":
                float(pd.to_numeric(g["control_current"]).mean()),
            "nogo_fa_rate":
                float(press[nogo].mean()) if nogo.any() else np.nan,
            "go_miss_rate":
                float((~press[go]).mean()) if go.any() else np.nan,
            "press_rate": float(press.mean()),
            "mean_rt_ms": float(rt[press].mean()),
        })
    return pd.DataFrame(rows)


def fit_standardized_ols(df, ycol, predictor, covariates, winsor=False):
    cols = [ycol, predictor] + list(covariates)
    x = df[cols].copy().dropna()
    if winsor:
        for c in [ycol, predictor] + [
            z for z in covariates if z != "group_high"
        ]:
            x[c] = winsorize(x[c])

    y = zscore(x[ycol])
    parts = [np.ones(len(x)), zscore(x[predictor]).to_numpy(float)]
    names = ["intercept", predictor]

    for cov in covariates:
        if cov == "group_high":
            parts.append(pd.to_numeric(x[cov]).to_numpy(float))
        else:
            parts.append(zscore(x[cov]).to_numpy(float))
        names.append(cov)

    X = np.column_stack(parts)
    ok = np.isfinite(y.to_numpy()) & np.all(np.isfinite(X), axis=1)
    X = X[ok]
    yy = y.to_numpy(float)[ok]
    xused = x.loc[ok].reset_index(drop=True)

    if len(yy) <= X.shape[1] + 1:
        raise ValueError("Not enough subjects for OLS")

    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    dof = len(yy) - X.shape[1]
    sigma2 = float(np.sum(resid**2) / dof)
    covb = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(covb))
    tval = beta / se
    p = 2 * stats.t.sf(np.abs(tval), dof)
    r2 = 1 - np.sum(resid**2) / np.sum((yy - yy.mean())**2)
    return {
        "beta": float(beta[1]),
        "se": float(se[1]),
        "p": float(p[1]),
        "r2": float(r2),
        "n": int(len(yy)),
        "used": xused,
    }


def permutation_p(df, ycol, predictor, covariates, observed_beta, n_perm, rng, winsor=False):
    base = df.copy()
    vals = []
    pred = base[predictor].to_numpy(copy=True)
    for _ in range(int(n_perm)):
        z = base.copy()
        z[predictor] = rng.permutation(pred)
        try:
            fit = fit_standardized_ols(
                z, ycol, predictor, covariates, winsor=winsor
            )
            vals.append(fit["beta"])
        except Exception:
            pass
    vals = np.asarray(vals, dtype=float)
    if not len(vals):
        return np.nan
    return float(
        (np.sum(np.abs(vals) >= abs(observed_beta)) + 1)
        / (len(vals) + 1)
    )


def bootstrap_beta(df, ycol, predictor, covariates, n_boot, rng, winsor=False):
    vals = []
    n = len(df)
    for _ in range(int(n_boot)):
        b = df.iloc[rng.integers(0, n, size=n)].reset_index(drop=True)
        try:
            vals.append(
                fit_standardized_ols(
                    b, ycol, predictor, covariates, winsor=winsor
                )["beta"]
            )
        except Exception:
            pass
    vals = np.asarray(vals, dtype=float)
    if not len(vals):
        return np.nan, np.nan
    return (
        float(np.quantile(vals, 0.025)),
        float(np.quantile(vals, 0.975)),
    )


def loo_stability(df, ycol, predictor, covariates, winsor=False):
    vals = []
    ids = []
    for i in range(len(df)):
        z = df.drop(df.index[i]).reset_index(drop=True)
        try:
            vals.append(
                fit_standardized_ols(
                    z, ycol, predictor, covariates, winsor=winsor
                )["beta"]
            )
            ids.append(str(df.iloc[i]["subject_id"]))
        except Exception:
            pass
    return np.asarray(vals, dtype=float), ids


def association_analysis(primary_contrasts, subj, n_perm, n_boot, seed):
    wide = primary_contrasts.pivot(
        index="subject_id",
        columns="signature",
        values="fa_minus_hit",
    ).reset_index()

    dat = subj.merge(wide, on="subject_id", how="inner")
    rng = np.random.default_rng(seed)

    specs = [
        ("primary_group", ["group_high"], False),
        ("sensitivity_none", [], False),
        (
            "sensitivity_group_behavior",
            ["group_high", "nogo_fa_rate", "go_miss_rate", "mean_rt_ms"],
            False,
        ),
        ("sensitivity_group_winsor", ["group_high"], True),
    ]

    rows = []
    loo_rows = []

    for signature in PRIMARY_FEATURES:
        if signature not in dat.columns:
            continue

        for predictor in RL_PREDICTORS:
            for analysis, covs, use_winsor in specs:
                fit = fit_standardized_ols(
                    dat, signature, predictor, covs, winsor=use_winsor
                )
                pp = permutation_p(
                    dat, signature, predictor, covs,
                    fit["beta"], n_perm, rng, winsor=use_winsor
                )
                lo, hi = bootstrap_beta(
                    dat, signature, predictor, covs,
                    n_boot, rng, winsor=use_winsor
                )
                loo, left_ids = loo_stability(
                    dat, signature, predictor, covs, winsor=use_winsor
                )

                same_dir = np.nan
                if len(loo):
                    sgn = np.sign(fit["beta"])
                    same_dir = float(np.mean(np.sign(loo) == sgn))

                rows.append({
                    "signature": signature,
                    "predictor": predictor,
                    "analysis": analysis,
                    "covariates": ",".join(covs),
                    "winsorized": bool(use_winsor),
                    "n_subjects": fit["n"],
                    "standardized_beta": fit["beta"],
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "ols_p": fit["p"],
                    "permutation_p": pp,
                    "r2": fit["r2"],
                    "loo_same_direction_fraction": same_dir,
                    "loo_min_beta": float(np.min(loo)) if len(loo) else np.nan,
                    "loo_max_beta": float(np.max(loo)) if len(loo) else np.nan,
                })

                for sid, b in zip(left_ids, loo):
                    loo_rows.append({
                        "signature": signature,
                        "predictor": predictor,
                        "analysis": analysis,
                        "left_out_subject": sid,
                        "standardized_beta": float(b),
                    })

    out = pd.DataFrame(rows)
    out["primary_permutation_q_fdr_bh"] = np.nan
    m = out["analysis"] == "primary_group"
    out.loc[m, "primary_permutation_q_fdr_bh"] = fdr_bh(
        out.loc[m, "permutation_p"].values
    )
    return out, pd.DataFrame(loo_rows), dat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--erp", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--entropy", required=True)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min-trials", nargs="+", type=int, default=[5, 10, 15, 20])
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    out = ensure_dir(args.out)

    erp = load_source(args.erp, "ERP")
    tf = load_source(args.tf, "TF")
    ent = load_source(args.entropy, "ENTROPY")
    lat = pd.read_csv(Path(args.latents).expanduser().resolve())

    trials, qc = build_trial_table(erp, tf, ent, lat)
    outcome, contrast_tables = outcome_summary(
        trials, args.min_trials, args.n_perm, args.n_boot, args.seed
    )
    if 10 not in contrast_tables:
        raise ValueError("Primary min_trials=10 is required")

    subj = subject_summaries(trials)
    association_results, loo, association_table = association_analysis(
        contrast_tables[10],
        subj,
        args.n_perm,
        args.n_boot,
        args.seed + 1,
    )

    qc.to_csv(out / "response_signature_feature_qc.csv", index=False)
    outcome.to_csv(out / "performance_monitoring_outcome_contrasts.csv", index=False)
    contrast_tables[10].to_csv(
        out / "performance_monitoring_subject_contrasts_min10.csv", index=False
    )
    subj.to_csv(out / "canonical_subject_rl_behavior_summary.csv", index=False)
    association_results.to_csv(out / "canonical_rl_eeg_associations_legacy.csv", index=False)
    loo.to_csv(out / "canonical_rl_eeg_associations_legacy_loo.csv", index=False)
    association_table.to_csv(out / "canonical_association_analysis_table.csv", index=False)

    settings = {
        "seed": args.seed,
        "min_trial_thresholds": args.min_trials,
        "primary_min_trials": 10,
        "n_permutations": args.n_perm,
        "n_bootstrap": args.n_boot,
        "rl_source": str(Path(args.latents).expanduser().resolve()),
        "rl_provenance":
            "canonical nested/cross-fitted LATENTS only; embedded EEG-table RL columns ignored",
        "trial_mapping":
            "EEG trial_index - 1 = canonical RL trial_index",
        "primary_neural_signatures": PRIMARY_FEATURES,
        "primary_rl_predictors": RL_PREDICTORS,
        "legacy_direct_permutation_association_model":
            "standardized subject-level OLS adjusted for group; direct predictor permutation + bootstrap; retained for provenance only",
        "association_status":
            "direct predictor-permutation association analysis is superseded for manuscript inference; corrected Freedman-Lane inference is authoritative",
    }
    (out / "analysis_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    primary_outcome = outcome[outcome["min_trials"] == 10].copy()
    primary_association = association_results[association_results["analysis"] == "primary_group"].copy()

    report = [
        "# EEG-computational study canonical response-locked performance-monitoring signatures",
        "",
        "## Primary response-locked performance-monitoring contrasts (FA minus Go Hit; min 10 trials/cell)",
        "",
        primary_outcome.to_markdown(index=False),
        "",
        "## Historical direct-permutation RL–EEG association analysis (superseded for manuscript inference)",
        "",
        primary_association.to_markdown(index=False),
        "",
        "Corrected manuscript inference is produced by "
        "`between_participant_rl_eeg_associations.py`.",
        "",
        "## Interpretation constraints",
        "",
        "- Response-locked contrasts characterize post-response-locked performance-monitoring activity; they are not prospective prediction of the same action.",
        "- RL summaries come only from the canonical nested/cross-fitted LATENTS table.",
        "- Positive/negative neural contrasts retain the physical feature direction shown in the table; no mechanistic meaning should be inferred from sign alone without feature-specific interpretation.",
        "- The direct-permutation association analysis shown above is retained for provenance only; corrected Freedman-Lane inference is reported separately.",
        "- A RL–EEG association is not causal evidence that an RL variable is literally implemented by the measured neural signature.",
    ]
    (out / "CANONICAL_RESPONSE_MONITORING_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("\n=== Primary response-locked performance-monitoring contrasts ===")
    print(primary_outcome.to_string(index=False))
    print("\n=== Primary canonical RL–EEG association analysis ===")
    print(primary_association.to_string(index=False))
    print("\n=== Feature QC ===")
    print(qc.to_string(index=False))
    print("\nOutput:", out)


if __name__ == "__main__":
    import argparse
    main()
