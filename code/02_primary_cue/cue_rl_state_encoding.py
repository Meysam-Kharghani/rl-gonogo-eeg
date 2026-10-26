#!/usr/bin/env python3
"""Strict fold-aligned cue-EEG to reinforcement-learning-state encoding benchmark.

For each reinforcement-learning outer fold, the analysis loads the posterior
estimated on outer-training participants, regenerates reinforcement-learning
trajectories for outer-training and outer-test participants from that same
train-only posterior, attaches the predefined nominal cue-locked EEG
predictors, tunes and fits ridge regression on outer-training participants, and
evaluates model-derived targets on untouched outer-test participants.

Targets are policy logit, actor Q-value difference, and current control state.
Positive mean participant MSE gain indicates lower held-out standardized MSE
than the strictly lagged behavioral-history reference model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def import_py(path, name):
    path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def stable_seed(base, *parts):
    payload = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int((base + int.from_bytes(digest[:4], "little")) % (2**32 - 1))


def ensure_dir(path):
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path, obj):
    def conv(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        return x

    Path(path).write_text(
        json.dumps(conv(obj), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def safe_logit(x, eps=1e-5):
    z = np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)
    return np.log(z / (1.0 - z))


def make_model(alpha):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=float(alpha))),
    ])


def predictor_specs(blocks, include_group=False):
    base = ["is_nogo", "cue_up", "cue_bi", "prev_error"]
    if include_group:
        base.append("group_high")

    history = base + [
        "prev_press",
        "prior_press_rate",
        "prior_fa_rate",
    ]

    return {
        "N0": ("history", history),
        "N1": ("history_plus_cue_erp", history + blocks["erp"]),
        "N2": ("history_plus_cue_tf", history + blocks["tf"]),
        "N3": ("history_plus_cue_entropy", history + blocks["entropy"]),
        "N4": ("history_plus_combined_cue_eeg", history + blocks["combined"]),
    }


def target_values(df, target):
    if target == "logit_p_press":
        if "p_press" not in df.columns:
            raise ValueError("Missing p_press.")
        return safe_logit(pd.to_numeric(df["p_press"], errors="coerce"))

    if target not in df.columns:
        raise ValueError(f"Missing target: {target}")

    return pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)


def subject_folds(df, k, seed):
    cols = ["subj_id_ddm"]
    if "group_high" in df.columns:
        cols.append("group_high")

    subjects = df[cols].drop_duplicates("subj_id_ddm").copy()
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(int(k))]

    if "group_high" in subjects.columns:
        grouped = subjects.groupby("group_high", sort=True)
    else:
        subjects["_group"] = 0
        grouped = subjects.groupby("_group", sort=True)

    for _, g in grouped:
        ids = g["subj_id_ddm"].astype(str).to_numpy()
        rng.shuffle(ids)
        for i, sid in enumerate(ids):
            folds[i % int(k)].append(str(sid))

    return folds


def mean_subject_mse(df, y, pred):
    tmp = pd.DataFrame({
        "subject_id": df["subj_id_ddm"].astype(str).values,
        "y": np.asarray(y, dtype=float),
        "pred": np.asarray(pred, dtype=float),
    })

    vals = []
    for _, g in tmp.groupby("subject_id", sort=False):
        ok = np.isfinite(g["y"]) & np.isfinite(g["pred"])
        if ok.any():
            vals.append(
                mean_squared_error(
                    g.loc[ok, "y"],
                    g.loc[ok, "pred"],
                )
            )
    return float(np.mean(vals)) if vals else np.inf


def tune_alpha(train, cols, target, alphas, inner_k, seed):
    folds = subject_folds(train, inner_k, seed)
    rows = []

    for alpha in alphas:
        scores = []

        for val_ids in folds:
            val_ids = set(map(str, val_ids))
            is_val = train["subj_id_ddm"].astype(str).isin(val_ids)

            tr = train.loc[~is_val].reset_index(drop=True)
            va = train.loc[is_val].reset_index(drop=True)

            ytr = target_values(tr, target)
            yva = target_values(va, target)

            ok_tr = np.isfinite(ytr)
            ok_va = np.isfinite(yva)
            if ok_tr.sum() < 2 or ok_va.sum() < 1:
                continue

            mu = float(np.mean(ytr[ok_tr]))
            sd = float(np.std(ytr[ok_tr], ddof=0))
            if not np.isfinite(sd) or sd <= 1e-12:
                continue

            ytr_z = (ytr[ok_tr] - mu) / sd
            yva_z = (yva[ok_va] - mu) / sd

            model = make_model(alpha)
            model.fit(tr.loc[ok_tr, cols], ytr_z)
            pred_z = model.predict(va.loc[ok_va, cols])

            scores.append(
                mean_subject_mse(
                    va.loc[ok_va].reset_index(drop=True),
                    yva_z,
                    pred_z,
                )
            )

        rows.append({
            "alpha": float(alpha),
            "mean_inner_subject_mse_z": (
                float(np.mean(scores)) if scores else np.inf
            ),
            "n_inner_folds": int(len(scores)),
        })

    table = pd.DataFrame(rows).sort_values(
        ["mean_inner_subject_mse_z", "alpha"],
        ascending=[True, True],
    ).reset_index(drop=True)

    if table.empty or not np.isfinite(table.iloc[0]["mean_inner_subject_mse_z"]):
        raise RuntimeError(
            f"Could not tune alpha for target={target}, p={len(cols)}"
        )

    return float(table.iloc[0]["alpha"]), table


def bootstrap_ci(vals, n_boot, seed):
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        boot[b] = np.mean(
            rng.choice(x, size=len(x), replace=True)
        )

    return (
        float(np.mean(x)),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def planned_contrasts(subj, n_boot, seed):
    planned = [
        ("N1", "N0", "cue ERP beyond strictly lagged behavioral history"),
        ("N2", "N0", "cue TF beyond strictly lagged behavioral history"),
        ("N3", "N0", "cue entropy beyond strictly lagged behavioral history"),
        ("N4", "N0", "combined cue EEG beyond strictly lagged behavioral history"),
    ]

    out = []

    for target in sorted(subj["target"].unique()):
        sd = subj[subj["target"] == target]
        available = set(sd["model"])

        for model, reference, label in planned:
            if model not in available or reference not in available:
                continue

            a = sd[sd["model"] == model][
                ["subject_id", "fold", "mse_z"]
            ].rename(columns={"mse_z": "mse_model"})

            b = sd[sd["model"] == reference][
                ["subject_id", "fold", "mse_z"]
            ].rename(columns={"mse_z": "mse_reference"})

            m = a.merge(
                b,
                on=["subject_id", "fold"],
                validate="one_to_one",
            )

            # Positive = EEG model lowers held-out error.
            m["gain"] = m["mse_reference"] - m["mse_model"]

            mean, lo, hi = bootstrap_ci(
                m["gain"].values,
                n_boot,
                stable_seed(seed, "contrast", target, model, reference),
            )

            fold_gain = m.groupby("fold")["gain"].mean()

            out.append({
                "target": target,
                "model": model,
                "reference": reference,
                "label": label,
                "n_subjects": int(m["subject_id"].nunique()),
                "mean_subject_mse_gain_z": mean,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "positive_subjects": int((m["gain"] > 0).sum()),
                "positive_folds": int((fold_gain > 0).sum()),
                "n_folds": int(fold_gain.size),
            })

    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()

    for x in [
        "nested-script",
        "rl-script",
        "ddm-script",
        "eeg-action-script",
        "rl-out",
        "data-root",
        "eeg-table",
        "out",
    ]:
        ap.add_argument("--" + x, required=True)

    ap.add_argument("--rl-variant", default="M6")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--folds", nargs="+", type=int, default=None)
    ap.add_argument(
        "--targets",
        nargs="+",
        default=[
            "logit_p_press",
            "actor_qdiff_current",
            "control_current",
        ],
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=["N0", "N1", "N2", "N3", "N4"],
    )
    ap.add_argument("--max-rl-draws", type=int, default=300)
    ap.add_argument("--deadline-ms", type=float, default=700.0)
    ap.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[
            0.0001, 0.001, 0.01, 0.1, 1.0,
            10.0, 100.0, 1000.0, 10000.0,
        ],
    )
    ap.add_argument("--inner-k", type=int, default=4)
    ap.add_argument("--include-group", action="store_true")
    ap.add_argument("--n-bootstrap", type=int, default=10000)

    a = ap.parse_args()
    out = ensure_dir(a.out)

    nested = import_py(a.nested_script, "p2enc_nested")
    rl = import_py(a.rl_script, "p2enc_rl")
    ddm = import_py(a.ddm_script, "p2enc_ddm")
    helper = import_py(a.eeg_action_script, "p2enc_helper")

    for fn in [
        "load_eeg_table",
        "load_fold_frame",
        "merge_eeg",
        "prepare_action_frame",
    ]:
        if not hasattr(helper, fn):
            raise AttributeError(
                f"EEG action script does not expose required helper: {fn}"
            )

    eeg, blocks = helper.load_eeg_table(a.eeg_table)
    specs = predictor_specs(blocks, a.include_group)

    bad_models = [m for m in a.models if m not in specs]
    if bad_models:
        raise ValueError(f"Unknown models: {bad_models}")

    valid_targets = {
        "logit_p_press",
        "actor_qdiff_current",
        "control_current",
    }
    bad_targets = [t for t in a.targets if t not in valid_targets]
    if bad_targets:
        raise ValueError(f"Unknown targets: {bad_targets}")

    arr = rl.load_artifacts(Path(a.rl_out))
    meta = rl.get_meta(arr)
    all_idx = set(range(int(meta["n_subj"])))

    cv = nested.read_json(
        nested.cv_summary_path(
            Path(a.rl_out),
            a.seed,
            a.rl_variant,
        )
    )
    folds = cv.get("folds", [])
    if len(folds) != a.k:
        raise ValueError(
            f"Expected {a.k} RL folds; found {len(folds)}"
        )

    selected = (
        {int(fr["fold"]) for fr in folds}
        if a.folds is None
        else set(a.folds)
    )

    fold_rows = []
    subj_rows = []
    fit_rows = []
    tune_rows = []
    align_rows = []

    for fr in folds:
        fi = int(fr["fold"])
        if fi not in selected:
            continue

        test_idx = [
            int(x)
            for x in fr.get("test_subject_indices", [])
        ]
        if not test_idx:
            fr2 = nested.read_json(
                nested.fold_result_path(
                    Path(a.rl_out),
                    a.seed,
                    a.rl_variant,
                    fi,
                ),
                default={},
            )
            test_idx = [
                int(x)
                for x in fr2.get("test_subject_indices", [])
            ]

        train_idx = sorted(all_idx - set(test_idx))

        print(
            f"[outer fold {fi}] train={len(train_idx)} test={len(test_idx)}",
            flush=True,
        )

        idata = nested.load_rl_fold_idata(
            Path(a.rl_out),
            a.seed,
            a.rl_variant,
            fi,
        )

        atr = rl.subset_subjects(arr, train_idx)
        ate = rl.subset_subjects(arr, test_idx)

        ltr = nested.posterior_draws_mean_latents(
            rl,
            atr,
            a.rl_variant,
            idata,
            a.max_rl_draws,
            stable_seed(a.seed, "p2enc_train", fi),
        )
        lte = nested.posterior_draws_mean_latents(
            rl,
            ate,
            a.rl_variant,
            idata,
            a.max_rl_draws,
            stable_seed(a.seed, "p2enc_test", fi),
        )

        rtr = nested.arr_to_nested_latent_rows(
            rl, atr, ltr, fi, a.seed, a.rl_variant
        )
        rte = nested.arr_to_nested_latent_rows(
            rl, ate, lte, fi, a.seed, a.rl_variant
        )

        tr = helper.load_fold_frame(
            ddm, rtr, a.data_root, a.deadline_ms
        )
        te = helper.load_fold_frame(
            ddm, rte, a.data_root, a.deadline_ms
        )

        tr, qa = helper.merge_eeg(
            tr, eeg, blocks["combined"]
        )
        te, qb = helper.merge_eeg(
            te, eeg, blocks["combined"]
        )

        tr = helper.prepare_action_frame(
            tr, a.include_group
        )
        te = helper.prepare_action_frame(
            te, a.include_group
        )

        align_rows += [
            {"fold": fi, "split": "train", **qa},
            {"fold": fi, "split": "test", **qb},
        ]

        for target in a.targets:
            ytr_all = target_values(tr, target)
            yte_all = target_values(te, target)

            ok_tr = np.isfinite(ytr_all)
            ok_te = np.isfinite(yte_all)

            trt = tr.loc[ok_tr].reset_index(drop=True)
            tet = te.loc[ok_te].reset_index(drop=True)
            ytr = ytr_all[ok_tr]
            yte = yte_all[ok_te]

            mu = float(np.mean(ytr))
            sd = float(np.std(ytr, ddof=0))
            if not np.isfinite(sd) or sd <= 1e-12:
                raise ValueError(
                    f"Fold {fi}, target {target}: invalid train SD={sd}"
                )

            ytr_z = (ytr - mu) / sd
            yte_z = (yte - mu) / sd

            for mk in a.models:
                name, cols = specs[mk]
                missing = [
                    c for c in cols
                    if c not in trt.columns or c not in tet.columns
                ]
                if missing:
                    raise ValueError(
                        f"{target}/{mk}: missing predictors {missing}"
                    )

                print(
                    f"[outer fold {fi}] target={target} "
                    f"fit {mk}: {name} p={len(cols)}",
                    flush=True,
                )

                alpha, tuning = tune_alpha(
                    trt,
                    cols,
                    target,
                    [float(x) for x in a.alphas],
                    a.inner_k,
                    stable_seed(
                        a.seed, "alpha", fi, target, mk
                    ),
                )

                for _, rr in tuning.iterrows():
                    tune_rows.append({
                        "fold": fi,
                        "target": target,
                        "model": mk,
                        "alpha": float(rr["alpha"]),
                        "mean_inner_subject_mse_z":
                            float(rr["mean_inner_subject_mse_z"]),
                        "n_inner_folds":
                            int(rr["n_inner_folds"]),
                        "selected":
                            bool(float(rr["alpha"]) == alpha),
                    })

                model = make_model(alpha)
                model.fit(trt[cols], ytr_z)
                pred_z = model.predict(tet[cols])

                mse = float(
                    mean_squared_error(yte_z, pred_z)
                )
                r2 = float(
                    r2_score(yte_z, pred_z)
                )

                corr = np.nan
                if (
                    len(yte_z) >= 2
                    and np.std(yte_z) > 0
                    and np.std(pred_z) > 0
                ):
                    corr = float(
                        np.corrcoef(yte_z, pred_z)[0, 1]
                    )

                fold_rows.append({
                    "fold": fi,
                    "target": target,
                    "model": mk,
                    "model_name": name,
                    "chosen_alpha": alpha,
                    "n_features": len(cols),
                    "n_test_trials": len(yte_z),
                    "mse_z": mse,
                    "r2": r2,
                    "pearson_r": corr,
                })

                tmp = pd.DataFrame({
                    "subject_id":
                        tet["subj_id_ddm"].astype(str).values,
                    "y_z": yte_z,
                    "pred_z": pred_z,
                })

                for sid, g in tmp.groupby(
                    "subject_id", sort=False
                ):
                    sr = np.nan
                    if (
                        len(g) >= 2
                        and g["y_z"].std(ddof=0) > 0
                        and g["pred_z"].std(ddof=0) > 0
                    ):
                        sr = float(
                            np.corrcoef(
                                g["y_z"], g["pred_z"]
                            )[0, 1]
                        )

                    subj_rows.append({
                        "fold": fi,
                        "subject_id": str(sid),
                        "target": target,
                        "model": mk,
                        "model_name": name,
                        "n_trials": len(g),
                        "mse_z": float(
                            mean_squared_error(
                                g["y_z"], g["pred_z"]
                            )
                        ),
                        "pearson_r": sr,
                    })

                fit_rows.append({
                    "fold": fi,
                    "target": target,
                    "model": mk,
                    "model_name": name,
                    "chosen_alpha": alpha,
                    "n_features": len(cols),
                    "train_n_trials": len(trt),
                    "train_n_subjects":
                        trt["subj_id_ddm"].nunique(),
                    "test_n_trials": len(tet),
                    "test_n_subjects":
                        tet["subj_id_ddm"].nunique(),
                    "target_train_mean": mu,
                    "target_train_sd": sd,
                })

    f = pd.DataFrame(fold_rows)
    s = pd.DataFrame(subj_rows)
    ft = pd.DataFrame(fit_rows)
    tu = pd.DataFrame(tune_rows)
    q = pd.DataFrame(align_rows)

    f.to_csv(
        out / "cue_rl_state_encoding_fold_scores.csv",
        index=False,
    )
    s.to_csv(
        out / "cue_rl_state_encoding_subject_metrics.csv",
        index=False,
    )
    ft.to_csv(
        out / "cue_rl_state_encoding_fit_summary.csv",
        index=False,
    )
    tu.to_csv(
        out / "cue_rl_state_encoding_alpha_tuning.csv",
        index=False,
    )
    q.to_csv(
        out / "cue_rl_state_encoding_alignment_qc.csv",
        index=False,
    )

    summary = (
        f.groupby(
            ["target", "model", "model_name"],
            as_index=False,
        )
        .agg(
            n_folds=("fold", "nunique"),
            mean_mse_z=("mse_z", "mean"),
            mean_r2=("r2", "mean"),
            mean_pearson_r=("pearson_r", "mean"),
            mean_n_features=("n_features", "mean"),
        )
        .sort_values(
            ["target", "mean_mse_z"],
            ascending=[True, True],
        )
    )
    summary.to_csv(
        out / "cue_rl_state_encoding_model_summary.csv",
        index=False,
    )

    c = planned_contrasts(
        s,
        a.n_bootstrap,
        a.seed,
    )
    c.to_csv(
        out / "cue_rl_state_encoding_planned_contrasts.csv",
        index=False,
    )

    write_json(
        out / "analysis_settings.json",
        {
            **vars(a),
            "strict_outer_fold_definition":
                "Same existing RL outer folds. RL targets for both train and "
                "test subjects are regenerated from that fold's train-only RL posterior.",
            "primary_predictor_definition":
                "Nominal cue-window EEG predictors only; no target-locked or response-locked EEG.",
            "baseline_definition":
                "Task design + strictly lagged behavioral history from companion behavioral-computational study.",
            "contrast_sign":
                "Positive MSE gain = EEG lowers held-out standardized target MSE.",
        },
    )

    report = (
        "# Cue-window EEG encoding of cross-fitted RL states\n\n"
        "## Model summary\n\n"
        + summary.to_markdown(index=False)
        + "\n\n## Planned contrasts\n\n"
        + c.to_markdown(index=False)
        + "\n\n## Interpretation constraints\n\n"
        "- Positive MSE gain means the EEG-augmented model reduced held-out target error relative to strictly lagged behavioral history.\n"
        "- `logit_p_press` is policy probability on the log-odds scale.\n"
        "- Both train and test RL targets are generated from the same outer-train-only RL posterior within each fold.\n"
        "- Only nominal cue-window EEG enters this primary encoding benchmark.\n"
        "- Predictive encoding does not establish a literal or unique neural representation.\n"
    )

    (out / "CUE_RL_STATE_ENCODING_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )

    print("\n=== Model summary ===")
    print(summary.to_string(index=False))

    print("\n=== Planned contrasts ===")
    print(c.to_string(index=False))

    print("\n=== Selected alpha counts ===")
    print(
        ft.groupby(
            ["target", "model", "chosen_alpha"]
        )
        .size()
        .rename("n_folds")
        .reset_index()
        .to_string(index=False)
    )

    print("\nOutput:", out)


if __name__ == "__main__":
    main()
