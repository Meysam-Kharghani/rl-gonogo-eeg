#!/usr/bin/env python3
"""Strict outer-fold RL -> action benchmark.

For every outer RL fold, this script loads that fold's train-only RL posterior,
generates population-predictive latent trajectories for BOTH the outer-training
and outer-test subjects from the same posterior, fits downstream action models on the
outer-training subjects, and evaluates only on the untouched outer-test subjects.

This avoids the subtle second-stage leakage that can occur when one pooled CSV of
subject-wise cross-fitted latents is re-split for a new downstream CV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def import_py(path: Path, name: str):
    import importlib.util
    path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def stable_seed(base: int, *parts: object) -> int:
    text = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(text).digest()
    return int((base + int.from_bytes(digest[:4], "little")) % (2**32 - 1))


def safe_logit(x, eps: float = 1e-5):
    x = np.clip(np.asarray(x, dtype=float), eps, 1 - eps)
    return np.log(x / (1 - x))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def cue_map(path: Path) -> dict[str, float]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k).strip().lower(): float(v) for k, v in raw.items()}


def add_beliefs(df: pd.DataFrame, probs: dict[str, float], prior_strength: float) -> pd.DataFrame:
    out = df.copy().sort_values(["subj_id_ddm", "trial_index"]).reset_index(drop=True)
    cue = out["cue_label_ddm"].astype(str).str.strip().str.lower()
    out["instructed_nogo_prob"] = cue.map(probs)
    if out["instructed_nogo_prob"].isna().any():
        bad = sorted(cue[out["instructed_nogo_prob"].isna()].unique())
        raise ValueError(f"Missing instructed probability for cues: {bad}")
    out["instructed_nogo_logit"] = safe_logit(out["instructed_nogo_prob"])
    out["_is_nogo"] = out["is_nogo"].astype(float)
    grp = out.groupby(["subj_id_ddm", "cue_label_ddm"], sort=False)
    prior_n = grp.cumcount().astype(float)
    prior_ng = grp["_is_nogo"].transform(lambda s: s.cumsum().shift(1)).fillna(0.0)
    a0 = out["instructed_nogo_prob"] * prior_strength
    b0 = (1.0 - out["instructed_nogo_prob"]) * prior_strength
    out["empirical_nogo_prob"] = (a0 + prior_ng) / (a0 + b0 + prior_n)
    out["empirical_nogo_logit"] = safe_logit(out["empirical_nogo_prob"])
    return out.drop(columns=["_is_nogo"])


def feature_sets(include_group: bool) -> dict[str, list[str]]:
    base = ["is_nogo", "cue_up", "cue_bi", "prev_error"]
    if include_group:
        base.append("group_high")
    hist = ["prev_press", "prior_press_rate", "prior_fa_rate"]
    policy = ["logit_p_press"]
    ac = ["actor_qdiff_current", "control_current"]
    return {
        "task_design_only": base,
        "instructed_numeric": [c for c in base if c not in {"cue_up", "cue_bi"}] + ["instructed_nogo_logit"],
        "empirical_cue_belief": [c for c in base if c not in {"cue_up", "cue_bi"}] + ["empirical_nogo_logit"],
        "lagged_history_reference": base + hist,
        "policy_only": base + policy,
        "actor_control_no_policy": base + ac,
        "actor_control_only": base + policy + ac,
        "policy_plus_history": base + policy + hist,
        "actor_control_plus_history": base + ac + hist,
        "full_rl_plus_history": base + policy + ac + hist,
        "instructed_plus_actor": [c for c in base if c not in {"cue_up", "cue_bi"}] + ["instructed_nogo_logit"] + policy + ac,
    }


def make_model(C: float, seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=float(C), solver="lbfgs", max_iter=4000, random_state=seed)),
    ])


def subject_folds(df: pd.DataFrame, k: int, seed: int) -> list[list[str]]:
    subjects = df[["subj_id_ddm", "group_high"]].drop_duplicates("subj_id_ddm")
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for _, g in subjects.groupby("group_high", sort=True):
        ids = g["subj_id_ddm"].astype(str).to_numpy()
        rng.shuffle(ids)
        for i, sid in enumerate(ids):
            folds[i % k].append(str(sid))
    return folds


def tune_C(train: pd.DataFrame, cols: list[str], Cs: list[float], inner_k: int, seed: int) -> float:
    if len(Cs) == 1:
        return float(Cs[0])
    folds = subject_folds(train, inner_k, seed)
    scores = []
    for C in Cs:
        total = 0.0
        for fi, ids in enumerate(folds):
            tr = train[~train["subj_id_ddm"].astype(str).isin(ids)]
            va = train[train["subj_id_ddm"].astype(str).isin(ids)]
            if tr.empty or va.empty:
                continue
            model = make_model(C, seed + fi)
            model.fit(tr[cols], tr["press"].astype(int))
            p = np.clip(model.predict_proba(va[cols])[:, 1], 1e-6, 1 - 1e-6)
            y = va["press"].astype(int).to_numpy()
            total += float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        scores.append((total, float(C)))
    scores.sort(key=lambda x: (x[0], -abs(np.log10(x[1]))), reverse=True)
    return scores[0][1]


def ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ans = 0.0
    for b in range(n_bins):
        m = idx == b
        if np.any(m):
            ans += np.mean(m) * abs(np.mean(y[m]) - np.mean(p[m]))
    return float(ans)


def metrics(y, p) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "observed_press_rate": float(np.mean(y)),
        "predicted_press_rate": float(np.mean(p)),
        "elpd": float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "ece10": ece(y, p),
    }


def bootstrap_ci(vals, n_boot: int, seed: int):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(n_boot)])
    return float(np.mean(vals)), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def prepare_frame(ddm, rows: list[dict], data_root: Path, deadline_ms: float,
                  rt_filter_mode: str, rt_min_ms: float, rt_max_ms: float,
                  probs: dict[str, float], prior_strength: float) -> tuple[pd.DataFrame, dict]:
    with tempfile.TemporaryDirectory(prefix="rl_action_analysis_") as td:
        tmp = Path(td) / "latents.csv"
        pd.DataFrame(rows).to_csv(tmp, index=False)
        raw = ddm.load_latents(tmp, deadline_ms=deadline_ms, data_root=data_root)
    if rt_filter_mode == "ddm":
        d, info = ddm.filter_ddm_trials(raw, rt_min_ms=rt_min_ms, rt_max_ms=rt_max_ms)
    else:
        d = raw.reset_index(drop=True)
        info = {
            "mode": "all_action_trials",
            "n_raw": int(len(raw)), "n_kept": int(len(raw)), "n_removed": 0,
            "note": "All action trials retained; RT is not used to define the action outcome sample.",
        }
    d = add_beliefs(d, probs, prior_strength)
    d["logit_p_press"] = safe_logit(pd.to_numeric(d["p_press"], errors="coerce").fillna(.5))
    return d, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nested-script", required=True)
    ap.add_argument("--rl-script", required=True)
    ap.add_argument("--ddm-script", required=True)
    ap.add_argument("--rl-out", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--cue-probabilities", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="M6")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-rl-draws", type=int, default=300)
    ap.add_argument("--prior-strength", type=float, default=10.0)
    ap.add_argument("--deadline-ms", type=float, default=700.0)
    ap.add_argument("--rt-filter-mode", choices=["all", "ddm"], default="all")
    ap.add_argument("--rt-min-ms", type=float, default=100.0)
    ap.add_argument("--rt-max-ms", type=float, default=1000.0)
    ap.add_argument("--include-group", action="store_true")
    ap.add_argument("--C-grid", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--inner-k", type=int, default=4)
    ap.add_argument("--horizons", nargs="+", type=int, default=[0, 20, 50, 100, 200])
    ap.add_argument("--trial-bin-edges", nargs="+", type=int, default=[0, 20, 50, 100, 200, 1000000])
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    args = ap.parse_args()

    out = ensure_dir(args.out)
    nested = import_py(Path(args.nested_script), "strict_nested_support")
    rlmod = import_py(Path(args.rl_script), "strict_rl_model")
    ddm = import_py(Path(args.ddm_script), "strict_ddm_loader")
    probs = cue_map(Path(args.cue_probabilities))
    sets = feature_sets(args.include_group)

    arr = rlmod.load_artifacts(Path(args.rl_out))
    meta = rlmod.get_meta(arr)
    n_subj = int(meta["n_subj"])
    all_idx = set(range(n_subj))
    cv_path = nested.cv_summary_path(Path(args.rl_out), args.seed, args.variant)
    cv_obj = nested.read_json(cv_path)
    folds = cv_obj.get("folds", [])
    if not folds:
        raise ValueError(f"No fold metadata in {cv_path}")
    if len(folds) != args.k:
        raise ValueError(f"Requested k={args.k}, but RL CV metadata contains {len(folds)} folds")

    fold_rows, subj_rows, pred_rows, C_rows, qc_rows = [], [], [], [], []
    for fr in folds:
        fi = int(fr["fold"])
        test_idx = [int(x) for x in fr.get("test_subject_indices", [])]
        if not test_idx:
            fr2 = nested.read_json(nested.fold_result_path(Path(args.rl_out), args.seed, args.variant, fi), default={})
            test_idx = [int(x) for x in fr2.get("test_subject_indices", [])]
        train_idx = sorted(all_idx.difference(test_idx))
        print(f"[outer fold {fi}] train={len(train_idx)} test={len(test_idx)}", flush=True)

        idata = nested.load_rl_fold_idata(Path(args.rl_out), args.seed, args.variant, fi)
        arr_train = rlmod.subset_subjects(arr, train_idx)
        arr_test = rlmod.subset_subjects(arr, test_idx)
        lat_train = nested.posterior_draws_mean_latents(
            rlmod, arr_train, args.variant, idata, args.max_rl_draws,
            stable_seed(args.seed, "train", fi),
        )
        lat_test = nested.posterior_draws_mean_latents(
            rlmod, arr_test, args.variant, idata, args.max_rl_draws,
            stable_seed(args.seed, "test", fi),
        )
        rows_train = nested.arr_to_nested_latent_rows(rlmod, arr_train, lat_train, fi, args.seed, args.variant)
        rows_test = nested.arr_to_nested_latent_rows(rlmod, arr_test, lat_test, fi, args.seed, args.variant)
        train, train_qc = prepare_frame(ddm, rows_train, Path(args.data_root), args.deadline_ms,
                                        args.rt_filter_mode, args.rt_min_ms, args.rt_max_ms,
                                        probs, args.prior_strength)
        test, test_qc = prepare_frame(ddm, rows_test, Path(args.data_root), args.deadline_ms,
                                      args.rt_filter_mode, args.rt_min_ms, args.rt_max_ms,
                                      probs, args.prior_strength)
        qc_rows.append({"fold": fi, "split": "train", **train_qc})
        qc_rows.append({"fold": fi, "split": "test", **test_qc})

        required = sorted({c for cols in sets.values() for c in cols})
        missing = [c for c in required if c not in train.columns or c not in test.columns]
        if missing:
            raise ValueError(f"Fold {fi}: missing columns {missing}")

        predictions = {"m6_direct_probability": np.clip(test["p_press"].astype(float).to_numpy(), 1e-6, 1 - 1e-6)}
        for model_name, cols in sets.items():
            chosen_C = tune_C(train, cols, args.C_grid, min(args.inner_k, max(2, len(train["subj_id_ddm"].unique()) // 4)),
                              stable_seed(args.seed, "tune", fi, model_name))
            model = make_model(chosen_C, stable_seed(args.seed, "fit", fi, model_name))
            model.fit(train[cols], train["press"].astype(int))
            predictions[model_name] = model.predict_proba(test[cols])[:, 1]
            C_rows.append({"fold": fi, "model": model_name, "chosen_C": chosen_C})

        pf = test[["subj_id_ddm", "group_high", "trial_index", "trial_n_prior", "press", "is_nogo", "cue_label_ddm"]].copy()
        pf["fold"] = fi
        for name, p in predictions.items():
            pf[f"p__{name}"] = p
        pred_rows.append(pf)

        for horizon in sorted(set(args.horizons)):
            mask = test["trial_n_prior"].astype(float).to_numpy() >= horizon
            if not np.any(mask):
                continue
            y = test.loc[mask, "press"].astype(int).to_numpy()
            for name, p in predictions.items():
                row = {"fold": fi, "horizon_min_prior_trials": horizon, "model": name}
                row.update(metrics(y, p[mask]))
                fold_rows.append(row)

        edges = sorted(set(args.trial_bin_edges))
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = ((test["trial_n_prior"] >= lo) & (test["trial_n_prior"] < hi)).to_numpy()
            if not np.any(mask):
                continue
            label = f"{lo}-{hi-1}" if hi < 1000000 else f"{lo}+"
            y = test.loc[mask, "press"].astype(int).to_numpy()
            for name, p in predictions.items():
                row = {"fold": fi, "trial_bin": label, "model": name}
                row.update(metrics(y, p[mask]))
                fold_rows.append({"metric_scope": "trial_bin", **row})

        for sid, g in test.groupby("subj_id_ddm"):
            pos = test.index.get_indexer(g.index)
            y = g["press"].astype(int).to_numpy()
            for name, p in predictions.items():
                row = {"fold": fi, "subject_id": str(sid), "group_high": int(g["group_high"].iloc[0]), "model": name}
                row.update(metrics(y, p[pos]))
                subj_rows.append(row)

    folds_df = pd.DataFrame(fold_rows)
    subjects_df = pd.DataFrame(subj_rows)
    preds_df = pd.concat(pred_rows, ignore_index=True)
    Cs_df = pd.DataFrame(C_rows)
    qc_df = pd.DataFrame(qc_rows)
    folds_df.to_csv(out / "strict_action_fold_metrics.csv", index=False)
    subjects_df.to_csv(out / "strict_action_subject_metrics.csv", index=False)
    preds_df.to_csv(out / "strict_action_predictions.csv.gz", index=False, compression="gzip")
    Cs_df.to_csv(out / "strict_action_chosen_C.csv", index=False)
    qc_df.to_csv(out / "strict_action_filter_qc.csv", index=False)

    horizon_df = folds_df[folds_df.get("metric_scope", pd.Series(index=folds_df.index, dtype=object)).isna()].copy()
    summary = horizon_df.groupby(["horizon_min_prior_trials", "model"], as_index=False).agg(
        n_folds=("fold", "nunique"), total_elpd=("elpd", "sum"), mean_log_loss=("log_loss", "mean"),
        mean_auc=("auc", "mean"), mean_brier=("brier", "mean"), mean_ece10=("ece10", "mean"),
    ).sort_values(["horizon_min_prior_trials", "total_elpd"], ascending=[True, False])
    summary.to_csv(out / "strict_action_summary.csv", index=False)

    bin_df = folds_df[folds_df.get("metric_scope", pd.Series(index=folds_df.index, dtype=object)).eq("trial_bin")].copy()
    if not bin_df.empty:
        bin_summary = bin_df.groupby(["trial_bin", "model"], as_index=False).agg(
            n_folds=("fold", "nunique"), total_elpd=("elpd", "sum"), mean_log_loss=("log_loss", "mean"),
            mean_auc=("auc", "mean"), mean_brier=("brier", "mean"), mean_ece10=("ece10", "mean"),
        ).sort_values(["trial_bin", "total_elpd"], ascending=[True, False])
    else:
        bin_summary = pd.DataFrame()
    bin_summary.to_csv(out / "strict_action_trial_bin_summary.csv", index=False)

    wide = subjects_df.pivot_table(index="subject_id", columns="model", values="elpd", aggfunc="sum")
    refs = ["task_design_only", "lagged_history_reference", "policy_only", "actor_control_only", "policy_plus_history", "actor_control_plus_history"]
    pair_rows = []
    for model in wide.columns:
        for ref in refs:
            if model == ref or ref not in wide.columns:
                continue
            vals = (wide[model] - wide[ref]).dropna().to_numpy()
            mean, lo, hi = bootstrap_ci(vals, args.n_bootstrap, stable_seed(args.seed, "boot", model, ref))
            pair_rows.append({
                "model": model, "reference": ref, "n_subjects": len(vals),
                "mean_subject_delta_elpd": mean, "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
                "positive_subjects": int(np.sum(vals > 0)),
            })
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(out / "strict_action_pairwise_subject_deltas.csv", index=False)

    planned_pairs = [
        ("full_rl_plus_history", "lagged_history_reference"),
        ("full_rl_plus_history", "policy_plus_history"),
        ("actor_control_plus_history", "lagged_history_reference"),
        ("full_rl_plus_history", "actor_control_plus_history"),
        ("policy_plus_history", "lagged_history_reference"),
        ("actor_control_only", "policy_only"),
        ("actor_control_no_policy", "task_design_only"),
    ]
    planned_rows = []
    for i, (model, ref) in enumerate(planned_pairs):
        if model not in wide.columns or ref not in wide.columns:
            continue
        vals = (wide[model] - wide[ref]).dropna().to_numpy()
        mean, lo, hi = bootstrap_ci(vals, args.n_bootstrap, stable_seed(args.seed, "planned", i, model, ref))
        planned_rows.append({
            "model": model, "reference": ref, "n_subjects": len(vals),
            "mean_subject_delta_elpd": mean, "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
            "positive_subjects": int(np.sum(vals > 0)), "total_delta_elpd": float(np.sum(vals)),
        })
    planned = pd.DataFrame(planned_rows)
    planned.to_csv(out / "strict_action_planned_contrasts.csv", index=False)

    settings = vars(args).copy()
    settings.update({
        "n_subjects": n_subj,
        "strict_outer_fold_definition": (
            "For each RL outer fold, both train and test latent trajectories were generated from that fold's "
            "train-only RL posterior using population-predictive new-subject parameters. The downstream action model was "
            "fit on outer-train subjects and scored only on outer-test subjects."
        ),
        "feature_sets": sets,
    })
    (out / "analysis_settings.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    report = [
        "# Strict stacked RL-to-action benchmark", "",
        "## Main cumulative-horizon summary", "", summary.to_markdown(index=False), "",
        "Cumulative horizons overlap and should not be interpreted as independent time periods.", "",
    ]
    if not bin_summary.empty:
        report += ["## Disjoint trial-bin summary", "", bin_summary.to_markdown(index=False), ""]
    report += [
        "## Planned subject-clustered contrasts", "", planned.to_markdown(index=False), "",
        "## Full subject-clustered pairwise comparisons", "", pair.to_markdown(index=False), "",
        "## Interpretation rules", "",
        "- full_rl_plus_history > lagged_history_reference tests incremental total RL information beyond strictly lagged behavioral history.",
        "- full_rl_plus_history > policy_plus_history tests actor/control information beyond policy probability plus history.",
        "- actor_control_plus_history > lagged_history_reference tests actor/control information beyond history without policy probability.",
        "- full_rl_plus_history > actor_control_plus_history tests policy probability beyond actor/control plus history.",
        "- policy_plus_history > lagged_history_reference tests incremental policy-probability information.",
        "- actor_control_only > policy_only tests actor/control information beyond policy probability alone.",
        "- Use all action trials as primary for action occurrence; use the DDM RT filter only as sensitivity.",
        "- Treat alternate-seed analyses as sensitivity analyses, not independent replications.",
    ]
    (out / "STRICT_ACTION_BENCHMARK_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
