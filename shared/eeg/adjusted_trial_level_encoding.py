#!/usr/bin/env python3
"""Adjusted trial-level computational-to-EEG encoding.

This replaces simple within-subject correlations with models that:
- isolate within-person fluctuations;
- control outcome, cue, target class, RT, prior error, and trial progress;
- use subject fixed effects and subject-clustered standard errors;
- use Freedman-Lane residual permutation within subject/outcome/cue blocks;
- distinguish genuine latent-state predictors (control, actor value) from deterministic
  transforms of p_press (policy entropy and action surprise).

Important: policy entropy and action surprise are deterministic functions of p_press and
the observed action. They are tested as alternative representations, not as independent
mechanisms after a flexible p_press spline.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common_utils import (  # noqa: E402
    action_surprise,
    add_trial_history,
    binary_entropy,
    ensure_dir,
    fdr_bh,
    normalize_subject_id,
    stratified_subject_folds,
    within_subject_z,
)



def stable_seed(base: int, *parts: object) -> int:
    material = "|".join(map(str, parts)).encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(material).digest()[:4], "little")
    return int((int(base) + offset) % (2**32 - 1))

DEFAULT_TARGETS = [
    "resp_ern_fz_50_150_mean_uv", "resp_ern_fz_50_150_min_uv",
    "resp_pe_fz_200_400_mean_uv", "resp_pe_cz_200_400_mean_uv",
    "resp_theta_fz_0_200", "resp_theta_fz_0_400", "resp_theta_cz_0_400",
    "resp_beta_cz_200_500", "resp_global_theta_fraction", "resp_global_beta_fraction",
    "resp_global_spectral_entropy", "resp_cz_allwindows_spectral_entropy",
]
PREDICTORS = ["control_current", "actor_qdiff_current", "p_press", "policy_entropy", "action_surprise"]


def load_table(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    scol = "subject_id_norm" if "subject_id_norm" in d.columns else ("subject_id" if "subject_id" in d.columns else "subj_id_ddm")
    d["subject_id"] = d[scol].map(normalize_subject_id)
    if "trial_index_numeric" not in d.columns:
        tcol = "trial_index" if "trial_index" in d.columns else "trial_ordinal"
        d["trial_index_numeric"] = pd.to_numeric(d[tcol], errors="coerce")
    return d


def prepare(df: pd.DataFrame, block_size: int) -> pd.DataFrame:
    d = add_trial_history(df, "subject_id", "trial_index_numeric", block_size=block_size)
    d["cue_type"] = d.get("cue_type", "").astype(str).str.lower()
    d["log_rt"] = np.log(pd.to_numeric(d.get("rt_response_from_target_ms"), errors="coerce").where(lambda x: x > 0))
    if "p_press" in d.columns:
        p = pd.to_numeric(d["p_press"], errors="coerce")
        d["policy_entropy"] = binary_entropy(p)
        d["action_surprise"] = action_surprise(p, d["response_present_numeric"])
    return d


def subset_data(d: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "response_all":
        return d[d["outcome"].isin(["nogo_fa", "go_hit"])].copy()
    if subset == "fa_only":
        return d[d["outcome"].eq("nogo_fa")].copy()
    if subset == "hit_only":
        return d[d["outcome"].eq("go_hit")].copy()
    raise ValueError(subset)


def design_formula(target_z: str, predictor_z: str, subset: str, add_p_spline: bool) -> tuple[str, str]:
    cov = ["C(cue_type)", "prev_error", "prev_press", "trial_progress", "C(block)", "C(subject_id)"]
    if subset == "response_all":
        cov += ["C(outcome)", "log_rt"]
    elif subset in {"fa_only", "hit_only"}:
        cov += ["log_rt"]
    if add_p_spline:
        cov += ["bs(p_press_wpz, df=4, degree=3, include_intercept=False)"]
    reduced = f"{target_z} ~ " + " + ".join(cov)
    full = reduced + f" + {predictor_z}"
    return reduced, full


def fit_clustered(formula: str, data: pd.DataFrame):
    y, X = patsy.dmatrices(formula, data=data, return_type="dataframe", NA_action="drop")
    used = data.loc[y.index]
    fit = sm.OLS(y.iloc[:, 0].to_numpy(), X.to_numpy()).fit(cov_type="cluster", cov_kwds={"groups": used["subject_id"].to_numpy()})
    return fit, y.iloc[:, 0].to_numpy(), X, used


def freedman_lane(
    reduced_formula: str,
    full_formula: str,
    data: pd.DataFrame,
    predictor_term: str,
    n_perm: int,
    seed: int,
) -> tuple[float, float, int]:
    # Build full and reduced designs on their common complete-case rows.
    y_f, X_f = patsy.dmatrices(full_formula, data=data, return_type="dataframe", NA_action="drop")
    y_r, X_r = patsy.dmatrices(reduced_formula, data=data.loc[y_f.index], return_type="dataframe", NA_action="drop")
    common = y_f.index.intersection(y_r.index)
    y = y_f.loc[common].iloc[:, 0].to_numpy()
    Xf = X_f.loc[common]
    Xr = X_r.loc[common]
    used = data.loc[common]
    if predictor_term not in Xf.columns:
        return np.nan, np.nan, len(common)
    j = list(Xf.columns).index(predictor_term)
    # Fast one-added-term Freedman-Lane via Frisch-Waugh-Lovell.
    # The full model differs from the reduced model by exactly predictor_term.
    Xr_np = Xr.to_numpy(dtype=float)
    z = Xf.iloc[:, j].to_numpy(dtype=float)
    red = sm.OLS(y, Xr_np).fit()
    z_resid = z - sm.OLS(z, Xr_np).fit().fittedvalues
    denom = float(np.dot(z_resid, z_resid))
    if denom <= 1e-12:
        return np.nan, np.nan, len(common)
    resid = np.asarray(red.resid, dtype=float)
    obs = float(np.dot(z_resid, resid) / denom)
    blocks = (used["subject_id"].astype(str) + "|" + used["outcome"].astype(str) + "|" + used["cue_type"].astype(str)).to_numpy()
    block_indices = [np.flatnonzero(blocks == key) for key in pd.unique(blocks)]
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        rp = resid.copy()
        for idx in block_indices:
            if len(idx) > 1:
                rp[idx] = rng.permutation(rp[idx])
        null[b] = float(np.dot(z_resid, rp) / denom)
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    return obs, p, len(common)


def run_inference(d: pd.DataFrame, targets: list[str], n_perm: int, seed: int) -> pd.DataFrame:
    rows = []
    for subset in ["response_all", "fa_only", "hit_only"]:
        ds = subset_data(d, subset)
        for target in targets:
            if target not in ds.columns:
                continue
            target_z = within_subject_z(ds, target, "subject_id", suffix="__wpz")
            for predictor in PREDICTORS:
                if predictor not in ds.columns:
                    continue
                pred_z = within_subject_z(ds, predictor, "subject_id", suffix="__wpz")
                # Only non-deterministic latent states are tested beyond a flexible p_press spline.
                add_p_spline = predictor in {"control_current", "actor_qdiff_current"} and "p_press" in ds.columns
                if add_p_spline:
                    within_subject_z(ds, "p_press", "subject_id", suffix="_wpz")
                reduced, full = design_formula(target_z, pred_z, subset, add_p_spline)
                try:
                    fit, _, X, used = fit_clustered(full, ds)
                    if pred_z not in X.columns:
                        continue
                    j = list(X.columns).index(pred_z)
                    obs_beta, perm_p, n = freedman_lane(
                        reduced, full, ds, pred_z, n_perm,
                        stable_seed(seed, subset, target, predictor),
                    )
                    rows.append({
                        "subset": subset, "target": target, "predictor": predictor,
                        "predictor_interpretation": "incremental_beyond_spline_p_press" if add_p_spline else "alternative_representation_or_descriptive",
                        "n_trials": n, "n_subjects": int(used["subject_id"].nunique()),
                        "beta_cluster": float(fit.params[j]), "se_cluster": float(fit.bse[j]),
                        "p_cluster": float(fit.pvalues[j]), "freedman_lane_p": perm_p,
                        "r_squared": float(fit.rsquared),
                    })
                except Exception as exc:
                    rows.append({"subset": subset, "target": target, "predictor": predictor, "error": str(exc)})
    out = pd.DataFrame(rows)
    if not out.empty and "freedman_lane_p" in out.columns:
        out["freedman_lane_q_fdr_bh"] = fdr_bh(out["freedman_lane_p"])
        out["cluster_q_fdr_bh"] = fdr_bh(out["p_cluster"])
        out = out.sort_values(["freedman_lane_q_fdr_bh", "cluster_q_fdr_bh"], na_position="last")
    return out


def make_ridge_pipeline(numeric: list[str], categorical: list[str], spline_col: str | None = None) -> Pipeline:
    transformers = []
    regular_num = [c for c in numeric if c != spline_col]
    if regular_num:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), regular_num))
    if spline_col:
        transformers.append(("spline", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("spl", SplineTransformer(n_knots=5, degree=3, include_bias=False)),
            ("sc", StandardScaler()),
        ]), [spline_col]))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return Pipeline([("prep", ColumnTransformer(transformers)), ("reg", Ridge(alpha=10.0))])


def run_grouped_prediction(d: pd.DataFrame, targets: list[str], k: int, seed: int) -> pd.DataFrame:
    rows = []
    ds = subset_data(d, "response_all")
    folds = stratified_subject_folds(ds, "subject_id", "group" if "group" in ds.columns else None, k, seed)
    base_num = ["is_nogo", "prev_error", "prev_press", "trial_progress", "log_rt"]
    base_cat = ["cue_type", "outcome", "block"]
    specs = {
        "covariates_only": (base_num, base_cat, None),
        "linear_p_press": (base_num + ["p_press"], base_cat, None),
        "spline_p_press": (base_num + ["p_press"], base_cat, "p_press"),
        "policy_entropy": (base_num + ["policy_entropy"], base_cat, None),
        "action_surprise": (base_num + ["action_surprise"], base_cat, None),
        "control_actor_plus_spline_p": (base_num + ["p_press", "control_current", "actor_qdiff_current"], base_cat, "p_press"),
    }
    for target in targets:
        if target not in ds.columns:
            continue
        for fi, test_ids in enumerate(folds):
            train = ds[~ds["subject_id"].isin(test_ids)].copy()
            test = ds[ds["subject_id"].isin(test_ids)].copy()
            train[target] = pd.to_numeric(train[target], errors="coerce")
            test[target] = pd.to_numeric(test[target], errors="coerce")
            train = train.dropna(subset=[target]); test = test.dropna(subset=[target])
            if len(train) < 200 or len(test) < 50:
                continue
            for name, (num, cat, spline) in specs.items():
                cols = [c for c in num + cat if c in train.columns]
                if any(c not in train.columns for c in num if c not in base_num):
                    continue
                model = make_ridge_pipeline([c for c in num if c in train.columns], [c for c in cat if c in train.columns], spline)
                model.fit(train[cols], train[target])
                pred = model.predict(test[cols])
                rows.append({
                    "target": target, "fold": fi, "model": name, "n_test": len(test),
                    "rmse": float(mean_squared_error(test[target], pred) ** 0.5),
                    "r2": float(r2_score(test[target], pred)),
                })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial-features", required=True, help="Merged trial-level ERP/TF/latent table; entropy_trial_features.csv is ideal")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-cues", nargs="*", default=None, help="Optional cue labels; use 'up down' for confirmatory analyses")
    ap.add_argument("--block-size", type=int, default=95)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out = ensure_dir(args.out)
    d = prepare(load_table(Path(args.trial_features)), args.block_size)
    if args.include_cues:
        allowed = {str(x).strip().lower() for x in args.include_cues}
        d = d[d["cue_type"].isin(allowed)].copy()
    targets = args.targets if args.targets else [c for c in DEFAULT_TARGETS if c in d.columns]
    if not targets:
        raise ValueError("None of the targeted EEG columns were found. Pass --targets explicitly.")

    inference = run_inference(d, targets, args.n_perm, args.seed)
    prediction = run_grouped_prediction(d, targets, args.k, args.seed)
    inference.to_csv(out / "adjusted_computational_eeg_inference.csv", index=False)
    prediction.to_csv(out / "computational_eeg_grouped_prediction_folds.csv", index=False)
    pred_summary = prediction.groupby(["target", "model"], as_index=False).agg(
        n_folds=("fold", "nunique"), mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean")) if not prediction.empty else pd.DataFrame()
    pred_summary.to_csv(out / "computational_eeg_grouped_prediction_summary.csv", index=False)

    lines = [
        "# Adjusted computational-to-EEG encoding", "", "## Inference", "",
        inference.head(80).to_markdown(index=False) if not inference.empty else "No eligible models.", "",
        "## Held-out-subject representation comparison", "",
        pred_summary.head(80).to_markdown(index=False) if not pred_summary.empty else "No eligible predictive models.", "",
        "## Interpretation rules", "",
        "- Treat control_current and actor_qdiff_current as mechanistic candidates only when they survive the flexible p_press spline, clustered inference, and Freedman-Lane permutation.",
        "- Treat policy_entropy and action_surprise as alternative mathematical representations of p_press/action probability, not independent mechanisms.",
        "- FA-only and Hit-only effects are more informative than associations driven solely by mixing outcomes.",
        "- Held-out-subject improvement is supportive; absence of improvement limits predictive claims but does not erase a well-controlled encoding effect.",
    ]
    (out / "ADJUSTED_COMPUTATIONAL_EEG_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(inference.head(40).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
