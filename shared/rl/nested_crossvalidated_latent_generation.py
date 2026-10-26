#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict nested reinforcement-learning to diffusion-model cross-validation.

This module generates held-out reinforcement-learning latent trajectories from
outer-training-fold posteriors, runs diffusion-model cross-validation using
those fold-specific latents, evaluates response-time and nondecision-time
sensitivity specifications, and collects cross-validation outputs.

Held-out participants do not contribute to the reinforcement-learning posterior
used to generate their downstream predictors. New-participant latent
trajectories are generated from population- and group-level posterior
quantities estimated on the corresponding outer-training fold."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import arviz as az


# =============================================================================
# Small utilities
# =============================================================================


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(cmd: Sequence[str], cwd: Optional[Path] = None, dry_run: bool = False) -> None:
    print("\n" + "=" * 100)
    print("[cmd]", " ".join(map(str, cmd)))
    print("=" * 100)
    if dry_run:
        return
    subprocess.run(list(map(str, cmd)), cwd=str(cwd) if cwd else None, check=True)


def import_py(path: Path, name: str):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def as_float(x: Any, default: float = 0.0) -> float:
    try:
        a = np.asarray(x)
        if a.size == 0:
            return default
        v = float(a.reshape(-1)[0])
        return v if np.isfinite(v) else default
    except Exception:
        return default


def get_manifest_subject_ids(rl_out: Path) -> List[str]:
    manifest_path = Path(rl_out) / "artifacts" / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing RL manifest: {manifest_path}")
    man = pd.read_csv(manifest_path)
    if "subj_id" not in man.columns:
        raise ValueError(f"manifest.csv must contain subj_id column: {manifest_path}")
    return man["subj_id"].astype(str).tolist()


def _first_existing(*paths: Path) -> Optional[Path]:
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_seed_results_root(rl_out: Path, seed: int) -> Path:
    rl_out = Path(rl_out)
    candidates = [
        rl_out / f"seed_{seed}" / "results",
        rl_out / f"seed_{seed}",
        rl_out / "results" / f"seed_{seed}" / "results",
        rl_out / "results" / f"seed_{seed}",
        rl_out,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not resolve seed results root for seed={seed} under rl_out={rl_out}")


def _resolve_variant_dir(seed_root: Path, variant_key: str) -> Path:
    seed_root = Path(seed_root)
    candidates = [
        seed_root / f"{variant_key}_posterior",
        seed_root / f"{variant_key}_nuts",
        seed_root / variant_key,
    ]
    for c in candidates:
        if c.exists():
            return c

    # Historical controlled archives may use implementation-specific directory names.
    # Resolve them structurally without exposing retired naming conventions.
    ranked = []
    for cand in seed_root.rglob("*"):
        if not cand.is_dir() or not cand.name.startswith(variant_key):
            continue
        has_cv = (cand / "cv_folds").exists() or any(cand.rglob("fold_result.json"))
        has_posterior = any(cand.rglob("idata.nc"))
        if has_cv or has_posterior:
            ranked.append(cand)
    if ranked:
        ranked.sort(key=lambda x: (len(x.parts), x.as_posix()))
        return ranked[0]
    raise FileNotFoundError(f"Could not resolve variant dir for variant={variant_key} under {seed_root}")


def _resolve_fold_dir(seed_root: Path, variant_key: str, fold_idx: int) -> Path:
    seed_root = Path(seed_root)
    variant_dir = _resolve_variant_dir(seed_root, variant_key)
    fold_names = [f"fold_{fold_idx:02d}", f"fold_{fold_idx}"]
    for fn in fold_names:
        for c in [variant_dir / "cv_folds" / fn, variant_dir / fn, variant_dir / "cv" / fn]:
            if c.exists():
                return c
    for fr in variant_dir.rglob("fold_result.json"):
        if fr.parent.name in fold_names:
            return fr.parent
        if fr.parent.parent.name in fold_names:
            return fr.parent.parent
    raise FileNotFoundError(f"Could not resolve fold dir for variant={variant_key}, fold={fold_idx} under {seed_root}")


def _resolve_fold_artifact(fold_dir: Path) -> Path:
    fold_dir = Path(fold_dir)
    nc = _first_existing(*list(fold_dir.rglob("idata.nc")))
    if nc is not None:
        return nc
    tgz = _first_existing(*list(fold_dir.rglob("*.tar.gz")))
    if tgz is not None:
        return tgz
    fr = _first_existing(*list(fold_dir.rglob("fold_result.json")))
    if fr is not None:
        return fr
    raise FileNotFoundError(f"No idata.nc, tar.gz, or fold_result.json found under {fold_dir}")


def _candidate_related_tarballs(fold_result_json: Path,
                               rl_out: Optional[Path] = None,
                               seed: Optional[int] = None,
                               variant_key: Optional[str] = None,
                               fold_idx: Optional[int] = None) -> List[Path]:
    """Find likely tar.gz archives when idata.nc is not unpacked locally."""
    fold_result_json = Path(fold_result_json)
    candidates: List[Path] = []

    roots: List[Path] = [fold_result_json.parent, fold_result_json.parent.parent]
    if rl_out is not None:
        roots.append(Path(rl_out))
    if rl_out is not None and seed is not None:
        try:
            seed_root = _resolve_seed_results_root(rl_out, seed)
            roots.extend([seed_root, _resolve_variant_dir(seed_root, variant_key or "")])
        except Exception:
            pass

    for base in roots:
        if base.exists():
            try:
                candidates.extend(sorted(base.rglob("*.tar.gz")))
            except Exception:
                pass
            try:
                candidates.extend(sorted(base.glob("*.tar.gz")))
            except Exception:
                pass

    if rl_out is not None and seed is not None and variant_key is not None:
        try:
            seed_root = _resolve_seed_results_root(rl_out, seed)
            variant_dir = _resolve_variant_dir(seed_root, variant_key)
            candidates.extend(sorted(seed_root.rglob(f"*{variant_key}*.tar.gz")))
            candidates.extend(sorted(variant_dir.rglob(f"*{variant_key}*.tar.gz")))
            if fold_idx is not None:
                fold_tag = f"fold_{fold_idx:02d}"
                candidates.extend(sorted(seed_root.rglob(f"*{fold_tag}*.tar.gz")))
                candidates.extend(sorted(variant_dir.rglob(f"*{fold_tag}*.tar.gz")))
        except Exception:
            pass

    seen = set()
    out = []
    for c in candidates:
        c = Path(c)
        if c.exists() and c not in seen:
            seen.add(c)
            out.append(c)
    return out

def _load_idata_from_tarball(tar_path: Path,
                             prefer_tokens: Optional[Sequence[str]] = None) -> az.InferenceData:
    """Load the first compatible InferenceData object from a .tar.gz archive."""
    tar_path = Path(tar_path)
    prefer_tokens = [t for t in (prefer_tokens or []) if t]

    with tarfile.open(tar_path, mode="r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        if not members:
            raise FileNotFoundError(f"Archive has no files: {tar_path}")

        nc_members = [m for m in members if m.name.endswith(".nc") or m.name.endswith(".nc4") or m.name.endswith(".netcdf")]
        if not nc_members:
            nc_members = [m for m in members if "idata" in m.name.lower() and ".nc" in m.name.lower()]
        if not nc_members:
            raise FileNotFoundError(
                f"No compatible NetCDF members found in archive: {tar_path}. "
                f"Sample members: {[m.name for m in members[:10]]}"
            )

        def score(m):
            token_hit = any(tok and tok in m.name for tok in prefer_tokens)
            return (0 if token_hit else 1, len(m.name))

        nc_members = sorted(nc_members, key=score)
        last_err = None
        for member in nc_members:
            with tempfile.TemporaryDirectory(prefix="rl_posterior_archive_") as td:
                tf.extract(member, path=td)
                extracted = Path(td) / member.name
                if not extracted.exists():
                    extracted = next(Path(td).rglob(Path(member.name).name), None)
                    if extracted is None:
                        last_err = FileNotFoundError(f"Could not locate extracted member {member.name} from {tar_path}")
                        continue
                try:
                    return az.from_netcdf(extracted)
                except Exception as e:
                    last_err = e
                    continue

        if last_err is not None:
            raise last_err
        raise FileNotFoundError(f"Failed to load any netcdf member from archive: {tar_path}")

def _load_idata_from_fold_result_json(fold_result_json: Path,
                                     rl_out: Optional[Path] = None,
                                     seed: Optional[int] = None,
                                     variant_key: Optional[str] = None,
                                     fold_idx: Optional[int] = None) -> az.InferenceData:
    fold_result_json = Path(fold_result_json)
    meta = _load_json(fold_result_json)

    pointer_keys = [
        "idata_path", "idata_nc", "posterior_path", "result_path",
        "results_path", "artifact_path", "nc_path", "path"
    ]
    for k in pointer_keys:
        p = meta.get(k, None)
        if not p:
            continue
        p = Path(p)
        if not p.is_absolute():
            p = (fold_result_json.parent / p).resolve()
        if p.exists():
            if p.suffix == ".nc":
                return az.from_netcdf(p)
            if p.suffix == ".json":
                return _load_idata_from_fold_result_json(
                    p, rl_out=rl_out, seed=seed, variant_key=variant_key, fold_idx=fold_idx
                )
            if p.name.endswith(".tar.gz"):
                return _load_idata_from_tarball(
                    p,
                    prefer_tokens=[
                        f"fold_{fold_idx:02d}" if fold_idx is not None else None,
                        f"fold_{fold_idx}" if fold_idx is not None else None,
                        variant_key or "",
                    ],
                )

    search_roots: List[Path] = [fold_result_json.parent, fold_result_json.parent.parent]
    if rl_out is not None:
        search_roots.append(Path(rl_out))
    if rl_out is not None and seed is not None:
        try:
            seed_root = _resolve_seed_results_root(rl_out, seed)
            search_roots.extend([seed_root, _resolve_variant_dir(seed_root, variant_key or "")])
        except Exception:
            pass

    nc_candidates: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        try:
            nc_candidates.extend(sorted(root.rglob("idata.nc")))
        except Exception:
            pass
    if nc_candidates:
        tokens = [fold_result_json.parent.name, variant_key or ""]
        if fold_idx is not None:
            tokens.extend([f"fold_{fold_idx:02d}", f"fold_{fold_idx}"])

        def _score_nc(p: Path):
            s = str(p)
            hit = sum(tok in s for tok in tokens if tok)
            return (-hit, len(s))

        nc_candidates = sorted(set(map(Path, nc_candidates)), key=_score_nc)
        return az.from_netcdf(nc_candidates[0])

    related_tars = _candidate_related_tarballs(
        fold_result_json, rl_out=rl_out, seed=seed, variant_key=variant_key, fold_idx=fold_idx
    )
    prefer_tokens = [fold_result_json.parent.name, variant_key or ""]
    if fold_idx is not None:
        prefer_tokens.append(f"fold_{fold_idx:02d}")
        prefer_tokens.append(f"fold_{fold_idx}")

    tried: List[Path] = []
    for tgz in related_tars:
        tried.append(tgz)
        try:
            return _load_idata_from_tarball(tgz, prefer_tokens=prefer_tokens)
        except Exception:
            continue

    raise FileNotFoundError(
        f"fold_result.json exists but does not point to an idata.nc and no usable tarball was found. "
        f"tried_tarballs={tried[:10]} "
        f"Inspect keys={sorted(meta.keys())} in {fold_result_json}"
    )

def load_rl_fold_idata(rl_out: Path, seed: int, variant_key: str, fold_idx: int) -> az.InferenceData:
    seed_root = _resolve_seed_results_root(rl_out, seed)
    variant_dir = _resolve_variant_dir(seed_root, variant_key)
    fold_dir = _resolve_fold_dir(seed_root, variant_key, fold_idx)

    # 1) Preferred case: unpacked inference data under fold, variant, seed, or rl_out.
    for root in [fold_dir, variant_dir, seed_root, Path(rl_out)]:
        if not root.exists():
            continue
        try:
            nc = _first_existing(*list(root.rglob("idata.nc")))
            if nc is not None:
                return az.from_netcdf(nc)
        except Exception:
            pass

    # 2) Search archived inference data under fold, variant, seed, or rl_out.
    for root in [fold_dir, variant_dir, seed_root, Path(rl_out)]:
        if not root.exists():
            continue
        try:
            tgz = _first_existing(*list(root.rglob("*.tar.gz")))
            if tgz is not None:
                try:
                    return _load_idata_from_tarball(
                        tgz,
                        prefer_tokens=[variant_key, f"fold_{fold_idx:02d}", f"fold_{fold_idx}"],
                    )
                except Exception:
                    pass
        except Exception:
            pass

    # 3) Final search strategy: inspect fold_result sidecars.
    artifact = _resolve_fold_artifact(fold_dir)
    if artifact.name == "fold_result.json":
        return _load_idata_from_fold_result_json(
            artifact,
            rl_out=rl_out,
            seed=seed,
            variant_key=variant_key,
            fold_idx=fold_idx,
        )
    if artifact.suffix == ".nc":
        return az.from_netcdf(artifact)
    if artifact.name.endswith(".tar.gz"):
        return _load_idata_from_tarball(
            artifact,
            prefer_tokens=[variant_key, f"fold_{fold_idx:02d}", f"fold_{fold_idx}"],
        )

    raise RuntimeError(f"Unsupported fold artifact: {artifact}")

def fold_result_path(rl_out: Path, seed: int, variant: str, fold: int) -> Path:

    # Kept for backward compatibility.
    seed_root = _resolve_seed_results_root(rl_out, seed)
    fold_dir = _resolve_fold_dir(seed_root, variant, fold)
    return fold_dir / "fold_result.json"


def cv_summary_path(rl_out: Path, seed: int, variant: str) -> Path:
    seed_root = _resolve_seed_results_root(rl_out, seed)
    variant_dir = _resolve_variant_dir(seed_root, variant)
    return variant_dir / "cv_kfold_subject.json"


def posterior_draws_mean_latents(rlmod, arr_test: Dict[str, np.ndarray], variant_key: str,
                                 idata, max_draws: int, seed: int) -> Dict[str, np.ndarray]:
    """Posterior-predictive mean latent states for held-out subjects.

    The RL train-fold model has subject-specific random effects only for training
    subjects. For held-out subjects, subject-level parameters are drawn from the
    train-fold posterior hyperparameters using rlmod.params_for_new_subjects.
    The RL forward pass is then evaluated on held-out behavior and cues, and
    latent states are averaged over posterior draws. This procedure prevents use
    of full-data subject-specific RL estimates during DDM cross-validation.
    """
    variant = rlmod.VARIANTS[variant_key]
    meta = rlmod.get_meta(arr_test)
    group = arr_test["group"].astype(int)
    n_cue = len(meta["cue_levels"])
    draws = rlmod.posterior_draw_dicts(idata, max_draws=int(max_draws))
    if not draws:
        raise RuntimeError("No posterior draws found in RL fold idata.")

    accum = None
    count = 0
    for i, dd in enumerate(draws):
        params = rlmod.params_for_new_subjects(dd, variant, group, n_cue, int(seed) + i * 1009)
        fwd = rlmod.forward_numpy(arr_test, variant, params)
        keep = {
            "p_press": fwd["p_press"],
            "belief_nogo_current": fwd["belief"],
            "associability_current": fwd["associability"],
            "control_current": fwd["control"],
            "actor_qdiff_current": fwd["actor_qdiff"],
        }
        if accum is None:
            accum = {k: np.zeros_like(v, dtype=float) for k, v in keep.items()}
        for k, v in keep.items():
            accum[k] += np.asarray(v, dtype=float)
        count += 1

    assert accum is not None
    return {k: v / max(count, 1) for k, v in accum.items()}


def arr_to_nested_latent_rows(rlmod, arr_test: Dict[str, np.ndarray], lat: Dict[str, np.ndarray],
                              fold: int, seed: int, variant_key: str) -> List[Dict[str, Any]]:
    meta = rlmod.get_meta(arr_test)
    cue_levels = meta["cue_levels"]
    subj_ids = meta["subj_ids"]
    orig_indices = meta.get("orig_subject_indices", list(range(int(meta["n_subj"]))))
    S = int(meta["n_subj"])
    T = int(meta["n_trials_per_subj"])
    rows: List[Dict[str, Any]] = []
    for s in range(S):
        sid = str(subj_ids[s])
        orig_s = int(orig_indices[s]) if len(orig_indices) > s else int(s)
        for t in range(T):
            ci = int(arr_test["cue_idx"][s, t])
            rows.append({
                "subject_index": orig_s,
                "subject_id": sid,
                "group": int(arr_test["group"][s]),
                "trial_index": int(t),
                "cue_idx": ci,
                "cue_type": str(cue_levels[ci]),
                "is_nogo": int(arr_test["is_nogo"][s, t]),
                "action_press": int(arr_test["action"][s, t]),
                "correct": int(arr_test["correct"][s, t]),
                "error": int(arr_test["error"][s, t]),
                "fa": int(arr_test["fa"][s, t]),
                "p_press": float(lat["p_press"][s, t]),
                "belief_nogo_current": float(lat["belief_nogo_current"][s, t]),
                "associability_current": float(lat["associability_current"][s, t]),
                "control_current": float(lat["control_current"][s, t]),
                "actor_qdiff_current": float(lat["actor_qdiff_current"][s, t]),
                "cnv_z": float(arr_test["cnv_z"][s, t]),
                "p300_z": float(arr_test["p300_z"][s, t]),
                "ern_strength_z": float(arr_test["ern_strength_z"][s, t]),
                "pe_strength_z": float(arr_test["pe_strength_z"][s, t]),
                "nested_rl_fold": int(fold),
                "nested_rl_seed": int(seed),
                "nested_rl_variant": str(variant_key),
                "nested_rl_source": "train_fold_posterior_predictive_mean",
            })
    return rows


def make_nested_latents(args) -> None:
    rl_script = Path(args.rl_script)
    rl_out = Path(args.rl_out)
    out = Path(args.out)
    nested_dir = out / "nested_latents"
    ensure_dir(nested_dir)

    rlmod = import_py(rl_script, "actor_control_rl_2_imported_for_nested")

    cv_path = cv_summary_path(rl_out, args.seed, args.variant)
    if not cv_path.exists():
        if args.run_rl_cv_if_missing:
            cmd = [
                sys.executable, str(rl_script), "cv",
                "--out", str(rl_out),
                "--seed", str(args.seed),
                "--variants", str(args.variant),
                "--k", str(args.k),
                "--draws", str(args.rl_draws),
                "--tune", str(args.rl_tune),
                "--chains", str(args.rl_chains),
                "--cores", str(args.rl_cores),
                "--target_accept", str(args.rl_target_accept),
                "--max_post_draws", str(args.rl_score_draws),
            ]
            if args.force_refit_rl:
                cmd.append("--force_refit")
            run_cmd(cmd, dry_run=args.dry_run)
        else:
            raise FileNotFoundError(
                f"RL CV summary not found: {cv_path}\n"
                "Run 01_rl_actor_control_model.py cv first, or pass --run_rl_cv_if_missing."
            )

    arr = rlmod.load_artifacts(rl_out)
    cv_obj = read_json(cv_path)
    folds = cv_obj.get("folds", [])
    if not folds:
        # Inspect fold_result.json files directly when the summary is unavailable.
        seed_root = _resolve_seed_results_root(rl_out, args.seed)
        variant_dir = _resolve_variant_dir(seed_root, args.variant)
        fold_jsons = sorted(variant_dir.rglob("fold_result.json"))
        if not fold_jsons:
            raise ValueError(f"No folds found in {cv_path} and no fold_result.json files under {variant_dir}")
        folds = []
        for frp in fold_jsons:
            fr = read_json(frp, default={})
            if "fold" in fr:
                folds.append(fr)
            else:
                fold_num = None
                for part in frp.parts:
                    if part.startswith("fold_"):
                        try:
                            fold_num = int(part.split("_")[1])
                        except Exception:
                            pass
                if fold_num is None:
                    fold_num = len(folds)
                folds.append({"fold": fold_num, **fr})

    all_rows: List[Dict[str, Any]] = []
    leakage_rows = []
    for fr in folds:
        fi = int(fr["fold"])
        test_idx = [int(x) for x in fr.get("test_subject_indices", [])]
        if not test_idx:
            # Read fold_result.json when the cross-validation summary omits these fields.
            frp = fold_result_path(rl_out, args.seed, args.variant, fi)
            fr2 = read_json(frp, default={})
            test_idx = [int(x) for x in fr2.get("test_subject_indices", [])]
        if not test_idx:
            raise ValueError(f"Cannot determine test_subject_indices for fold {fi}.")

        seed_root = _resolve_seed_results_root(rl_out, args.seed)
        fold_dir = _resolve_fold_dir(seed_root, args.variant, fi)
        artifact = _resolve_fold_artifact(fold_dir)

        print(f"[nested_latents] fold={fi} test_subject_indices={test_idx} artifact={artifact}")
        arr_test = rlmod.subset_subjects(arr, test_idx)
        idata = load_rl_fold_idata(rl_out, args.seed, args.variant, fi)
        lat = posterior_draws_mean_latents(
            rlmod, arr_test, args.variant, idata,
            max_draws=args.max_rl_draws,
            seed=args.seed + 91001 * (fi + 1),
        )
        all_rows.extend(arr_to_nested_latent_rows(rlmod, arr_test, lat, fi, args.seed, args.variant))
        for s in test_idx:
            leakage_rows.append({"subject_index": int(s), "fold": fi, "count_as_test": 1})

    df = pd.DataFrame(all_rows)
    # Check every subject appears in exactly one test fold and all have full trials.
    subj_test_counts = pd.DataFrame(leakage_rows).groupby("subject_index")["count_as_test"].sum().to_dict()
    n_subjects = int(rlmod.get_meta(arr)["n_subj"])
    bad_counts = {int(k): int(v) for k, v in subj_test_counts.items() if int(v) != 1}
    missing_subj = [i for i in range(n_subjects) if i not in subj_test_counts]
    rows_per_subject = df.groupby("subject_index").size().to_dict()
    T = int(rlmod.get_meta(arr)["n_trials_per_subj"])
    bad_rows = {int(k): int(v) for k, v in rows_per_subject.items() if int(v) != T}

    if bad_counts or missing_subj or bad_rows:
        raise RuntimeError(
            "Nested latent leakage/coverage check failed: "
            f"bad_counts={bad_counts}, missing_subjects={missing_subj}, bad_rows={bad_rows}"
        )

    df = df.sort_values(["subject_index", "trial_index"]).reset_index(drop=True)
    out_csv = nested_dir / f"{args.variant}_seed{args.seed}_nested_trial_latents_for_ddm.csv"
    df.to_csv(out_csv, index=False)

    manifest = {
        "analysis": "nested_rl_to_ddm_latent_export",
        "rl_script": str(rl_script),
        "rl_out": str(rl_out),
        "variant": args.variant,
        "seed": int(args.seed),
        "k": int(args.k),
        "max_rl_draws": int(args.max_rl_draws),
        "n_subjects": n_subjects,
        "n_trials_per_subject": T,
        "n_rows": int(len(df)),
        "output_csv": str(out_csv),
        "leakage_control": (
            "Each held-out subject's latent states were generated from the corresponding "
            "RL train-fold posterior hyperparameters using posterior-predictive population/group "
            "parameters; no full-data subject-specific RL random effects were used."
        ),
        "folds": [
            {
                "fold": int(fr["fold"]),
                "test_subject_indices": [int(x) for x in fr.get("test_subject_indices", [])],
                "test_subject_ids": [str(x) for x in fr.get("test_subject_ids", [])],
            }
            for fr in folds
        ]
    }
    write_json(nested_dir / f"{args.variant}_seed{args.seed}_nested_latents_manifest.json", manifest)
    print("[nested_latents] wrote", out_csv)
    print("[nested_latents] wrote", nested_dir / f"{args.variant}_seed{args.seed}_nested_latents_manifest.json")


# =============================================================================
# DDM CV runners
# =============================================================================


def run_ddm_cv(args) -> None:
    out_dir = Path(args.out)
    if (out_dir / "cv_comparison.csv").exists():
        print(f"[skip] DDM CV already exists: {out_dir / 'cv_comparison.csv'}")
        return

    cmd = [
        sys.executable, str(args.ddm_script), "cv",
        "--latents", str(args.latents),
        "--data_root", str(args.data_root),
        "--out", str(args.out),
        "--variants", *list(args.variants),
        "--deadline_ms", str(args.deadline_ms),
        "--rt_min_ms", str(args.rt_min_ms),
        "--rt_max_ms", str(args.rt_max_ms),
        "--t0_min", str(args.t0_min),
        "--t0_max", str(args.t0_max),
        "--k", str(args.k),
        "--n_starts", str(args.n_starts),
        "--seed", str(args.seed),
    ]
    # Optional arguments retained for compatibility with the historical implementation.
    if args.l2 is not None:
        cmd.extend(["--l2", str(args.l2)])
    if args.dist_min is not None:
        cmd.extend(["--dist_min", str(args.dist_min)])
    run_cmd(cmd, dry_run=args.dry_run)


def sensitivity_grid(args) -> List[Dict[str, Any]]:
    if args.grid == "minimal":
        return [
            {"name": "higher_rtmin_150", "deadline_ms": 700, "rt_min_ms": 150, "rt_max_ms": 1000, "t0_min": 0.03, "t0_max": 0.08},
            {"name": "conservative_t0_050_150", "deadline_ms": 700, "rt_min_ms": 100, "rt_max_ms": 1000, "t0_min": 0.05, "t0_max": 0.15},
            {"name": "wider_t0_020_200", "deadline_ms": 700, "rt_min_ms": 100, "rt_max_ms": 1000, "t0_min": 0.02, "t0_max": 0.20},
        ]
    if args.grid == "strong":
        return [
            {"name": "higher_rtmin_150", "deadline_ms": 700, "rt_min_ms": 150, "rt_max_ms": 1000, "t0_min": 0.03, "t0_max": 0.08},
            {"name": "higher_rtmin_200", "deadline_ms": 700, "rt_min_ms": 200, "rt_max_ms": 1000, "t0_min": 0.03, "t0_max": 0.08},
            {"name": "conservative_t0_050_150", "deadline_ms": 700, "rt_min_ms": 100, "rt_max_ms": 1000, "t0_min": 0.05, "t0_max": 0.15},
            {"name": "conservative_t0_080_200", "deadline_ms": 700, "rt_min_ms": 100, "rt_max_ms": 1000, "t0_min": 0.08, "t0_max": 0.20},
            {"name": "wider_t0_020_200", "deadline_ms": 700, "rt_min_ms": 100, "rt_max_ms": 1000, "t0_min": 0.02, "t0_max": 0.20},
            {"name": "deadline_650", "deadline_ms": 650, "rt_min_ms": 100, "rt_max_ms": 1000, "t0_min": 0.03, "t0_max": 0.08},
            {"name": "deadline_750", "deadline_ms": 750, "rt_min_ms": 100, "rt_max_ms": 1000, "t0_min": 0.03, "t0_max": 0.08},
        ]
    raise ValueError(f"Unknown grid: {args.grid}")


def run_sensitivity(args) -> None:
    root = Path(args.out)
    ensure_dir(root)
    configs = sensitivity_grid(args)
    manifest_rows = []
    for cfg in configs:
        cfg_out = root / cfg["name"]
        if (cfg_out / "cv_comparison.csv").exists():
            print(f"[skip] sensitivity config already exists: {cfg_out / 'cv_comparison.csv'}")
            manifest_rows.append({"config": cfg["name"], "out": str(cfg_out), **cfg})
            continue
        cmd = [
            sys.executable, str(args.ddm_script), "cv",
            "--latents", str(args.latents),
            "--data_root", str(args.data_root),
            "--out", str(cfg_out),
            "--variants", *list(args.variants),
            "--deadline_ms", str(cfg["deadline_ms"]),
            "--rt_min_ms", str(cfg["rt_min_ms"]),
            "--rt_max_ms", str(cfg["rt_max_ms"]),
            "--t0_min", str(cfg["t0_min"]),
            "--t0_max", str(cfg["t0_max"]),
            "--k", str(args.k),
            "--n_starts", str(args.n_starts),
            "--seed", str(args.seed),
        ]
        if args.l2 is not None:
            cmd.extend(["--l2", str(args.l2)])
        if args.dist_min is not None:
            cmd.extend(["--dist_min", str(args.dist_min)])
        run_cmd(cmd, dry_run=args.dry_run)
        manifest_rows.append({"config": cfg["name"], "out": str(cfg_out), **cfg})

    pd.DataFrame(manifest_rows).to_csv(root / "sensitivity_manifest.csv", index=False)
    print("[sensitivity] wrote", root / "sensitivity_manifest.csv")

    # Collect immediately when not dry-run.
    if not args.dry_run:
        collect_roots([root], root / "summary")


# =============================================================================
# Collection / reporting
# =============================================================================


def collect_roots(roots: Sequence[Path], out: Path) -> None:
    ensure_dir(out)
    rows = []
    fold_rows = []
    for root in map(Path, roots):
        for comp in root.rglob("cv_comparison.csv"):
            try:
                df = pd.read_csv(comp)
            except Exception:
                continue
            rel = comp.parent.relative_to(root) if comp.parent != root else Path(".")
            for _, r in df.iterrows():
                rows.append({
                    "root": str(root),
                    "analysis_dir": str(comp.parent),
                    "config": str(rel),
                    **r.to_dict(),
                })
        for js in root.rglob("cv_kfold_subject.json"):
            try:
                obj = read_json(js)
            except Exception:
                continue
            variant = obj.get("variant")
            variant_name = obj.get("variant_name")
            for fr in obj.get("folds", []):
                fold_rows.append({
                    "root": str(root),
                    "analysis_dir": str(js.parent),
                    "variant": variant,
                    "variant_name": variant_name,
                    "fold": fr.get("fold"),
                    "elpd": fr.get("elpd"),
                    "nll": fr.get("nll"),
                })

    if rows:
        allcv = pd.DataFrame(rows)
        allcv.to_csv(out / "all_cv_comparisons.csv", index=False)
        # Rank by config and compute model deltas where possible.
        delta_rows = []
        for (analysis_dir, config), g in allcv.groupby(["analysis_dir", "config"], dropna=False):
            gg = g.copy().sort_values("total_elpd", ascending=False)
            best = float(gg["total_elpd"].iloc[0])
            for _, r in gg.iterrows():
                rr = r.to_dict()
                rr["rank_in_config"] = int(np.where(gg.index == r.name)[0][0] + 1)
                rr["delta_vs_best"] = float(r["total_elpd"] - best)
                # pairwise deltas are computed separately
                delta_rows.append(rr)
        pd.DataFrame(delta_rows).to_csv(out / "all_cv_ranked.csv", index=False)
        print("[collect] wrote", out / "all_cv_comparisons.csv")
        print("[collect] wrote", out / "all_cv_ranked.csv")
    else:
        print("[collect] no cv_comparison.csv files found")

    if fold_rows:
        pd.DataFrame(fold_rows).to_csv(out / "all_cv_folds.csv", index=False)
        print("[collect] wrote", out / "all_cv_folds.csv")

    write_json(out / "collect_manifest.json", {
        "roots": [str(Path(r)) for r in roots],
        "n_cv_tables": int(len(rows)),
        "n_fold_rows": int(len(fold_rows)),
        "notes": "Use all_cv_ranked.csv to verify whether D2 remains top across nested/sensitivity configurations.",
    })


def collect_cmd(args) -> None:
    collect_roots([Path(x) for x in args.roots], Path(args.out))


# =============================================================================
# Combined workflow helper
# =============================================================================


def all_cmd(args) -> None:
    # 1) nested latents
    nested_args = argparse.Namespace(**vars(args))
    make_nested_latents(nested_args)
    nested_lat = Path(args.out) / "nested_latents" / f"{args.variant}_seed{args.seed}_nested_trial_latents_for_ddm.csv"

    # 2) nested DDM CV, main settings
    ddm_out = Path(args.out) / f"ddm_nested_{args.variant}_seed{args.seed}_main"
    ddm_args = argparse.Namespace(
        ddm_script=args.ddm_script,
        latents=nested_lat,
        data_root=args.data_root,
        out=ddm_out,
        variants=args.variants,
        deadline_ms=args.deadline_ms,
        rt_min_ms=args.rt_min_ms,
        rt_max_ms=args.rt_max_ms,
        t0_min=args.t0_min,
        t0_max=args.t0_max,
        k=args.k,
        n_starts=args.n_starts,
        seed=args.seed,
        l2=args.l2,
        dist_min=args.dist_min,
        dry_run=args.dry_run,
    )
    run_ddm_cv(ddm_args)

    # 3) sensitivity on nested latents
    sens_args = argparse.Namespace(
        ddm_script=args.ddm_script,
        latents=nested_lat,
        data_root=args.data_root,
        out=Path(args.out) / f"ddm_nested_{args.variant}_seed{args.seed}_sensitivity",
        variants=args.variants,
        grid=args.grid,
        k=args.k,
        n_starts=args.n_starts,
        seed=args.seed,
        l2=args.l2,
        dist_min=args.dist_min,
        dry_run=args.dry_run,
    )
    run_sensitivity(sens_args)

    if not args.dry_run:
        collect_roots([Path(args.out)], Path(args.out) / "summary")


# =============================================================================
# CLI
# =============================================================================


def add_common_rl(p):
    p.add_argument("--rl_script", default="code/01_rl_actor_control_model.py")
    p.add_argument("--rl_out", default="outputs/actor_control_rl")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--variant", default="M6")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--max_rl_draws", type=int, default=300,
                   help="Posterior draws used to compute held-out posterior-predictive mean latents.")
    p.add_argument("--run_rl_cv_if_missing", action="store_true")
    p.add_argument("--force_refit_rl", action="store_true")
    p.add_argument("--rl_draws", type=int, default=600)
    p.add_argument("--rl_tune", type=int, default=600)
    p.add_argument("--rl_chains", type=int, default=4)
    p.add_argument("--rl_cores", type=int, default=4)
    p.add_argument("--rl_target_accept", type=float, default=0.9)
    p.add_argument("--rl_score_draws", type=int, default=400)
    p.add_argument("--dry_run", action="store_true")


def add_common_ddm(p):
    p.add_argument("--ddm_script", default="shared/ddm/one_boundary_diffusion.py")
    p.add_argument("--latents")
    p.add_argument("--data_root", default="data/raw/csv")
    p.add_argument("--out", required=True)
    p.add_argument("--variants", nargs="+", default=["D0", "D1", "D2", "D3", "D5"])
    p.add_argument("--deadline_ms", type=float, default=700)
    p.add_argument("--rt_min_ms", type=float, default=100)
    p.add_argument("--rt_max_ms", type=float, default=1000)
    p.add_argument("--t0_min", type=float, default=0.03)
    p.add_argument("--t0_max", type=float, default=0.08)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n_starts", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--l2", type=float, default=None)
    p.add_argument("--dist_min", type=float, default=None)
    p.add_argument("--dry_run", action="store_true")


def build_argparser():
    ap = argparse.ArgumentParser(description="Nested RL→DDM CV and DDM sensitivity checks")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("nested_latents", help="Export leakage-controlled held-out RL latents from RL CV folds")
    add_common_rl(p)
    p.set_defaults(func=make_nested_latents)

    p = sub.add_parser("ddm_cv", help="Run DDM subject-level CV on a latent CSV")
    add_common_ddm(p)
    p.set_defaults(func=run_ddm_cv)

    p = sub.add_parser("sensitivity", help="Run DDM CV over a predefined sensitivity grid")
    p.add_argument("--grid", choices=["minimal", "strong"], default="minimal")
    add_common_ddm(p)
    p.set_defaults(func=run_sensitivity)

    p = sub.add_parser("collect", help="Collect CV comparisons from nested/sensitivity output roots")
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=collect_cmd)

    p = sub.add_parser("all", help="Generate nested latents, run nested DDM CV, run sensitivity, collect")
    add_common_rl(p)
    # Manually add DDM args but avoid duplicate --out/--seed/--k.
    p.add_argument("--ddm_script", default="shared/ddm/one_boundary_diffusion.py")
    p.add_argument("--data_root", default="data/raw/csv")
    p.add_argument("--variants", nargs="+", default=["D0", "D1", "D2", "D3", "D5"],
                   help="DDM variants to run")
    p.add_argument("--deadline_ms", type=float, default=700)
    p.add_argument("--rt_min_ms", type=float, default=100)
    p.add_argument("--rt_max_ms", type=float, default=1000)
    p.add_argument("--t0_min", type=float, default=0.03)
    p.add_argument("--t0_max", type=float, default=0.08)
    p.add_argument("--n_starts", type=int, default=8)
    p.add_argument("--l2", type=float, default=None)
    p.add_argument("--dist_min", type=float, default=None)
    p.add_argument("--grid", choices=["minimal", "strong"], default="minimal")
    p.set_defaults(func=all_cmd)
    return ap


def main():
    args = build_argparser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
