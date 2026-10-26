#!/usr/bin/env python3
# FORMALIZED HISTORICAL SOURCE SNAPSHOT
# Evidence status: STOPPED_BY_DESIGN; NOT_PART_OF_FINAL_EVIDENCE
# This file preserves a historical project stage with public-facing terminology standardized for release.
# The SHA-256 of the original executed/development source is recorded in the corresponding provenance registry.

"""Trial-level post-error adaptation analysis.

Primary question
----------------
Does the magnitude of response-locked EEG activity on trial t predict behavior
on trial t+1 beyond current behavior, cue, computational state, and recent
history?  Transitions across blocks are removed.  Lag 2 and lag 3 are included
as temporal-placebo/specificity analyses.

Inference
---------
* Subject fixed effects + participant-clustered standard errors.
* Freedman-Lane residual permutation, with residuals shuffled within
  participant x current-outcome x next-cue blocks.
* Binary outcomes additionally receive a population-average GEE sensitivity.
* Held-out-participant prediction tests whether EEG adds prospective value.

This is association, not causal mediation.
"""
from __future__ import annotations
import hashlib

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from common_utils import (  # noqa: E402
    add_trial_history,
    ensure_dir,
    fdr_bh,
    infer_outcome,
    stratified_subject_folds,
    within_subject_z,
)

DEFAULT_FEATURE_PATTERNS = [
    r"^resp_ern_fz_50_150_mean_uv$",
    r"^resp_pe_fz_200_400_mean_uv$",
    r"^resp_pe_cz_200_400_mean_uv$",
    r"^resp_theta_fz_0_200$",
    r"^resp_theta_fz_0_400$",
    r"^resp_theta_cz_0_400$",
    r"^resp_beta_cz_200_500$",
    r"^resp_beta_c[34z]_m400_0$",
    r"^resp_mu_c[34z]_m400_0$",
]


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def canonicalize(d: pd.DataFrame, block_size: int) -> pd.DataFrame:
    d = d.copy()
    rename = {}
    if "subject_id" not in d.columns and "subject_id_norm" in d.columns:
        rename["subject_id_norm"] = "subject_id"
    if "trial_index_numeric" not in d.columns:
        if "trial_index" in d.columns:
            rename["trial_index"] = "trial_index_numeric"
        elif "trial_ordinal" in d.columns:
            rename["trial_ordinal"] = "trial_index_numeric"
    d = d.rename(columns=rename)
    required = {"subject_id", "trial_index_numeric", "trial_class", "response_present"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    d["subject_id"] = d["subject_id"].astype(str)
    d["trial_index_numeric"] = pd.to_numeric(d["trial_index_numeric"], errors="coerce")
    d["response_present"] = pd.to_numeric(d["response_present"], errors="coerce")
    if "rt_response_from_target_ms" in d.columns:
        d["rt_ms"] = pd.to_numeric(d["rt_response_from_target_ms"], errors="coerce")
    elif "rt_ms" in d.columns:
        d["rt_ms"] = pd.to_numeric(d["rt_ms"], errors="coerce")
    else:
        d["rt_ms"] = np.nan
    d["log_rt"] = np.log(d["rt_ms"].where(d["rt_ms"] > 0))
    if "cue_type" not in d.columns and "cue_type_norm" in d.columns:
        d["cue_type"] = d["cue_type_norm"]
    if "cue_type" not in d.columns:
        d["cue_type"] = "unknown"
    d["cue_type"] = d["cue_type"].astype(str).str.lower().str.strip()
    d["outcome"] = infer_outcome(d)
    d = add_trial_history(d, "subject_id", "trial_index_numeric", block_size=block_size)
    if "p_press" in d.columns:
        d["p_press"] = pd.to_numeric(d["p_press"], errors="coerce")
    else:
        d["p_press"] = np.nan
    for c in ["control_current", "actor_qdiff_current", "action_surprise", "policy_entropy"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def merge_tables(paths: list[Path], block_size: int) -> pd.DataFrame:
    tables = [canonicalize(load_table(p), block_size) for p in paths]
    keys = ["subject_id", "trial_index_numeric"]
    out = tables[0]
    for i, t in enumerate(tables[1:], start=2):
        feature_cols = [c for c in t.columns if c not in out.columns or c in keys]
        t2 = t[feature_cols].drop_duplicates(keys)
        out = out.merge(t2, on=keys, how="left", validate="one_to_one", suffixes=("", f"__t{i}"))
    return canonicalize(out, block_size)


def select_features(d: pd.DataFrame, explicit: list[str] | None) -> list[str]:
    if explicit:
        missing = [c for c in explicit if c not in d.columns]
        if missing:
            raise ValueError(f"Requested EEG features not found: {missing}")
        return explicit
    found = []
    for c in d.columns:
        if any(re.match(p, c, flags=re.I) for p in DEFAULT_FEATURE_PATTERNS):
            found.append(c)
    return found


def make_transition_table(d: pd.DataFrame, lag: int) -> pd.DataFrame:
    if lag < 1:
        raise ValueError("lag must be >= 1")
    d = d.sort_values(["subject_id", "trial_index_numeric"]).copy()
    g = d.groupby("subject_id", sort=False)
    next_cols = {
        "response_present_numeric": "next_response",
        "error": "next_error",
        "is_nogo": "next_is_nogo",
        "cue_type": "next_cue",
        "log_rt": "next_log_rt",
        "block": "next_block_lag",
        "trial_index_numeric": "next_trial_index",
    }
    for src, dst in next_cols.items():
        d[dst] = g[src].shift(-lag)
    d["same_block_lag"] = d["block"].eq(d["next_block_lag"])
    d["consecutive_lag"] = np.isclose(
        pd.to_numeric(d["next_trial_index"], errors="coerce") - d["trial_index_numeric"], lag,
        equal_nan=False,
    )
    d["current_commission_error"] = d["outcome"].eq("nogo_fa").astype(int)
    # Response-locked features are defined only for responded trials.
    d = d[d["response_present_numeric"].eq(1) & d["same_block_lag"] & d["consecutive_lag"]].copy()
    return d


def formula_terms(d: pd.DataFrame) -> list[str]:
    terms = [
        "current_commission_error", "log_rt", "p_press", "prev_error", "prev_press",
        "trial_progress", "next_is_nogo", "C(cue_type)", "C(next_cue)", "C(block)",
        "C(subject_id)",
    ]
    for c in ["control_current", "actor_qdiff_current", "action_surprise"]:
        if c in d.columns and pd.to_numeric(d[c], errors="coerce").notna().sum() > 50:
            terms.insert(4, c)
    return terms


def fit_clustered_ols(formula: str, d: pd.DataFrame):
    y, X = patsy.dmatrices(formula, d, return_type="dataframe", NA_action="drop")
    used = d.loc[y.index]
    fit = sm.OLS(y.iloc[:, 0], X).fit(cov_type="cluster", cov_kwds={"groups": used["subject_id"]})
    return fit, X, used


def freedman_lane(
    reduced_formula: str,
    full_formula: str,
    d: pd.DataFrame,
    tested_term: str,
    n_perm: int,
    seed: int,
) -> tuple[float, float, int, np.ndarray]:
    yf, Xf = patsy.dmatrices(full_formula, d, return_type="dataframe", NA_action="drop")
    yr, Xr = patsy.dmatrices(reduced_formula, d.loc[yf.index], return_type="dataframe", NA_action="drop")
    idx = yf.index.intersection(yr.index)
    y = yf.loc[idx].iloc[:, 0].to_numpy()
    Xf = Xf.loc[idx]
    Xr = Xr.loc[idx]
    used = d.loc[idx]
    if tested_term not in Xf.columns:
        return np.nan, np.nan, len(idx), np.array([])
    j = list(Xf.columns).index(tested_term)
    # Fast one-added-term Freedman-Lane via Frisch-Waugh-Lovell.
    Xr_np = Xr.to_numpy(dtype=float)
    z = Xf.iloc[:, j].to_numpy(dtype=float)
    red = sm.OLS(y, Xr_np).fit()
    z_resid = z - sm.OLS(z, Xr_np).fit().fittedvalues
    denom = float(np.dot(z_resid, z_resid))
    if denom <= 1e-12:
        return np.nan, np.nan, len(idx), np.array([])
    resid = np.asarray(red.resid, dtype=float)
    obs = float(np.dot(z_resid, resid) / denom)
    blocks = (
        used["subject_id"].astype(str) + "|" + used["outcome"].astype(str)
        + "|" + used["next_cue"].astype(str)
    ).to_numpy()
    block_indices = [np.flatnonzero(blocks == key) for key in pd.unique(blocks)]
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        rp = resid.copy()
        for ii in block_indices:
            if len(ii) > 1:
                rp[ii] = rng.permutation(rp[ii])
        null[b] = float(np.dot(z_resid, rp) / denom)
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    return obs, p, len(idx), null


def run_inference(d: pd.DataFrame, features: list[str], lags: list[int], n_perm: int, seed: int) -> pd.DataFrame:
    rows = []
    outcomes = [("next_log_rt", "continuous"), ("next_response", "binary"), ("next_error", "binary")]
    for lag in lags:
        tr = make_transition_table(d, lag)
        for feature in features:
            tr[feature] = pd.to_numeric(tr[feature], errors="coerce")
            z = within_subject_z(tr, feature, "subject_id", suffix="__wpz")
            tested = f"current_commission_error:{z}"
            covars = formula_terms(tr)
            reduced_rhs = " + ".join(covars + [z])
            full_rhs = reduced_rhs + f" + current_commission_error:{z}"
            for outcome, kind in outcomes:
                if outcome not in tr.columns:
                    continue
                formula_full = f"{outcome} ~ {full_rhs}"
                formula_red = f"{outcome} ~ {reduced_rhs}"
                try:
                    fit, X, used = fit_clustered_ols(formula_full, tr)
                    if tested not in X.columns:
                        continue
                    j = list(X.columns).index(tested)
                    obs, perm_p, n, _ = freedman_lane(
                        formula_red, formula_full, tr, tested, n_perm,
                        stable_seed(seed, lag, feature, outcome),
                    )
                    row = {
                        "lag": lag, "outcome": outcome, "outcome_kind": kind, "feature": feature,
                        "n_trials": n, "n_subjects": int(used["subject_id"].nunique()),
                        "interaction_beta_cluster": float(fit.params.iloc[j]),
                        "interaction_se_cluster": float(fit.bse.iloc[j]),
                        "interaction_p_cluster": float(fit.pvalues.iloc[j]),
                        "freedman_lane_beta": obs, "freedman_lane_p": perm_p,
                        "r_squared_lpm_or_ols": float(fit.rsquared),
                    }
                    # Population-average logistic sensitivity. Fixed subject dummies are omitted.
                    if kind == "binary":
                        gee_rhs = " + ".join([x for x in covars if x != "C(subject_id)"] + [z, f"current_commission_error:{z}"])
                        try:
                            gee = sm.GEE.from_formula(
                                f"{outcome} ~ {gee_rhs}", groups="subject_id", data=tr,
                                family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable(),
                            ).fit(maxiter=200)
                            row.update({
                                "gee_interaction_beta": float(gee.params.get(tested, np.nan)),
                                "gee_interaction_se": float(gee.bse.get(tested, np.nan)),
                                "gee_interaction_p": float(gee.pvalues.get(tested, np.nan)),
                            })
                        except Exception as exc:
                            row["gee_error"] = str(exc)
                    rows.append(row)
                except Exception as exc:
                    rows.append({"lag": lag, "outcome": outcome, "feature": feature, "error": str(exc)})
    out = pd.DataFrame(rows)
    if not out.empty:
        if "freedman_lane_p" in out.columns:
            # Confirmatory family = lag 1 only. Lags 2/3 are specificity checks.
            out["freedman_lane_q_lag1"] = np.nan
            m = out["lag"].eq(1) & out["freedman_lane_p"].notna()
            out.loc[m, "freedman_lane_q_lag1"] = fdr_bh(out.loc[m, "freedman_lane_p"])
        if "interaction_p_cluster" in out.columns:
            out["cluster_q_lag1"] = np.nan
            m = out["lag"].eq(1) & out["interaction_p_cluster"].notna()
            out.loc[m, "cluster_q_lag1"] = fdr_bh(out.loc[m, "interaction_p_cluster"])
    return out


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def run_prediction(d: pd.DataFrame, features: list[str], k: int, seed: int) -> pd.DataFrame:
    tr = make_transition_table(d, 1)
    folds = stratified_subject_folds(tr, "subject_id", "group" if "group" in tr.columns else None, k, seed)
    base_num = [c for c in [
        "current_commission_error", "log_rt", "p_press", "prev_error", "prev_press",
        "trial_progress", "next_is_nogo", "control_current", "actor_qdiff_current", "action_surprise",
    ] if c in tr.columns]
    base_cat = [c for c in ["cue_type", "next_cue", "block", "outcome"] if c in tr.columns]
    rows = []
    for outcome, binary in [("next_log_rt", False), ("next_response", True), ("next_error", True)]:
        for fi, ids in enumerate(folds):
            train = tr[~tr["subject_id"].isin(ids)].copy()
            test = tr[tr["subject_id"].isin(ids)].copy()
            for name, extra in [("covariates", []), ("covariates_plus_eeg", features)]:
                nums = base_num + [c for c in extra if c in tr.columns]
                cols = nums + base_cat
                train2 = train.dropna(subset=[outcome]).copy()
                test2 = test.dropna(subset=[outcome]).copy()
                if len(train2) < 100 or len(test2) < 20:
                    continue
                prep = make_preprocessor(nums, base_cat)
                est = LogisticRegression(C=0.2, max_iter=5000, class_weight="balanced") if binary else Ridge(alpha=10.0)
                pipe = Pipeline([("prep", prep), ("model", est)])
                try:
                    pipe.fit(train2[cols], train2[outcome])
                    if binary:
                        pred = pipe.predict_proba(test2[cols])[:, 1]
                        auc = roc_auc_score(test2[outcome], pred) if test2[outcome].nunique() == 2 else np.nan
                        loss = log_loss(test2[outcome], pred, labels=[0, 1])
                        rows.append({"outcome": outcome, "fold": fi, "model": name, "n_test": len(test2), "auc": auc, "log_loss": loss})
                    else:
                        pred = pipe.predict(test2[cols])
                        rmse = mean_squared_error(test2[outcome], pred) ** 0.5
                        rows.append({"outcome": outcome, "fold": fi, "model": name, "n_test": len(test2), "rmse": rmse})
                except Exception as exc:
                    rows.append({"outcome": outcome, "fold": fi, "model": name, "error": str(exc)})
    return pd.DataFrame(rows)



def stable_seed(base: int, *parts: object) -> int:
    material = (str(base) + "|" + "|".join(map(str, parts))).encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(material).digest()[:4], "little")
    return int((base + offset) % (2**32 - 1))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial-features", nargs="+", required=True, help="One merged table, or several trial tables to merge by participant/trial")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-cues", nargs="*", default=None, help="Optional current-trial cue labels; use 'up down' for confirmatory analyses")
    ap.add_argument("--features", nargs="*", default=None)
    ap.add_argument("--lags", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--block-size", type=int, default=95)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out = ensure_dir(args.out)
    d = merge_tables([Path(x) for x in args.trial_features], args.block_size)
    if args.include_cues:
        allowed = {str(x).strip().lower() for x in args.include_cues}
        d = d[d["cue_type"].isin(allowed)].copy()
    features = select_features(d, args.features)
    if not features:
        raise ValueError("No response-locked EEG features found. Pass them explicitly with --features.")
    inference = run_inference(d, features, args.lags, args.n_perm, args.seed)
    prediction = run_prediction(d, features, args.k, args.seed)
    inference.to_csv(out / "post_error_adaptation_inference.csv", index=False)
    prediction.to_csv(out / "post_error_adaptation_prediction_folds.csv", index=False)
    if not prediction.empty:
        agg = prediction.groupby(["outcome", "model"], as_index=False).agg(
            n_folds=("fold", "nunique"), mean_auc=("auc", "mean") if "auc" in prediction.columns else ("fold", "size"),
            mean_log_loss=("log_loss", "mean") if "log_loss" in prediction.columns else ("fold", "size"),
            mean_rmse=("rmse", "mean") if "rmse" in prediction.columns else ("fold", "size"),
        )
    else:
        agg = pd.DataFrame()
    agg.to_csv(out / "post_error_adaptation_prediction_summary.csv", index=False)
    report = [
        "# Post-error adaptation report", "", "## Confirmatory inference (lag 1)", "",
        inference[inference.get("lag", pd.Series(dtype=float)).eq(1)].head(100).to_markdown(index=False) if not inference.empty else "No models.",
        "", "## Held-out-participant prediction", "", agg.to_markdown(index=False) if not agg.empty else "No predictions.",
        "", "## Decision rules", "",
        "- The primary evidence is lag 1, surviving participant-clustered inference and Freedman-Lane permutation after FDR correction.",
        "- An effect should be weaker or absent at lags 2 and 3 unless a longer-lived adaptation mechanism is explicitly hypothesized.",
        "- For binary outcomes, GEE is a sensitivity analysis; the shared permutation inference uses a linear-probability model.",
        "- Do not describe these observational temporal associations as causal mediation.",
    ]
    (out / "POST_ERROR_ADAPTATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(inference.head(60).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
