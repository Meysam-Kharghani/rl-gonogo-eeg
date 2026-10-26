#!/usr/bin/env python3
"""Repeated-partition stability for the frozen EEG-computational study cue action and RL-state encoding benchmarks.

Portability and correctness notes:
- computational subprocesses run with a Python that has PyMC/ArviZ, even when
  the wrapper itself is launched from the EEG conda environment;
- --analysis-python and --rl-out are explicit/overrideable;
- all manifest seeds are preflighted for complete M6 5-fold CV before any
  EEG-computational study benchmark begins;
- fails early with actionable commands rather than halfway through a long run.

Seed-2026 remains the historical primary analysis. This is stability evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


REQUIRED_CHILD_MODULES = ("numpy", "pandas", "sklearn", "arviz", "pymc")


def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def python_ok(py: Path) -> tuple[bool, str]:
    code = "import " + ", ".join(REQUIRED_CHILD_MODULES) + "; print('OK')"
    q = subprocess.run([str(py), "-c", code], text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return q.returncode == 0, q.stdout.strip()


def choose_python(explicit: str | None) -> Path:
    cands = []
    if explicit:
        cands.append(Path(explicit).expanduser())
    env_python = os.environ.get("RL_ANALYSIS_PYTHON")
    if env_python:
        cands.append(Path(env_python).expanduser())
    cands.append(Path(sys.executable))
    seen = set()
    diagnostics = []
    for p in cands:
        p = p.resolve()
        if p in seen or not p.exists():
            continue
        seen.add(p)
        ok, msg = python_ok(p)
        diagnostics.append(f"{p}: {'OK' if ok else msg[-500:]}")
        if ok:
            return p
    raise RuntimeError(
        "No Python interpreter with numpy/pandas/sklearn/arviz/pymc was found.\n"
        + "\n".join(diagnostics)
        + "\nUse --analysis-python /path/to/python, set RL_ANALYSIS_PYTHON, or activate the documented analysis environment."
    )


def read_cv_objects(rl_out: Path):
    out = []
    for p in rl_out.rglob("cv_kfold_subject.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append((p, obj))
    return out


def cv_ready(cv_objects, seed: int, variant="M6", k=5):
    hits = []
    for p, obj in cv_objects:
        try:
            same = int(obj.get("seed", -1)) == int(seed)
        except Exception:
            same = False
        if not same or str(obj.get("variant", "")) != variant:
            continue
        folds = obj.get("folds", [])
        status = str(obj.get("status", "done")).lower()
        if len(folds) == k and status in {"done", "complete", "completed"}:
            # Older summaries may omit fold status; if present, require done.
            ok = all(
                str(fr.get("status", "done")).lower() in {"done", "complete", "completed"}
                for fr in folds
            )
            if ok:
                hits.append(p)
    return sorted(hits, key=lambda p: (len(str(p)), str(p)))


def infer_rl_out_from_cv_path(p: Path) -> Path:
    parts = p.parts
    for i, part in enumerate(parts):
        if part.startswith("seed_"):
            return Path(*parts[:i])
    # Fallback: a direct variant directory layout.
    return p.parent.parent


def find_rl_out(root: Path, explicit: str | None, seeds: list[int]) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--rl-out does not exist: {p}")
        return p

    known = [
        root / "controlled_inputs/rl_crossvalidated_posteriors",
        root / "controlled_inputs/rl_crossvalidated_posteriors",
    ]
    for p in known:
        if p.exists():
            objs = read_cv_objects(p)
            if cv_ready(objs, 2026):
                return p.resolve()

    candidates = []
    for cv in root.rglob("cv_kfold_subject.json"):
        try:
            obj = json.loads(cv.read_text(encoding="utf-8"))
        except Exception:
            continue
        if int(obj.get("seed", -1)) == 2026 and str(obj.get("variant", "")) == "M6":
            candidates.append(infer_rl_out_from_cv_path(cv))
    if not candidates:
        raise FileNotFoundError(
            "Could not locate an RL-CV output root containing seed-2026 M6 CV.\n"
            f"Search with:\n  find {root} -type f -name cv_kfold_subject.json -print | grep M6\n"
            "Then pass --rl-out /PATH/TO/RL_CV_ROOT."
        )
    return sorted(set(candidates), key=lambda p: (len(str(p)), str(p)))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--seed-manifest", default=None)
    ap.add_argument("--rl-out", default=None,
                    help="Root containing M6 CV folders for every repeated seed.")
    ap.add_argument("--analysis-python", default=None,
                    help="Python used for the primary cue analyses; should provide the documented RL dependencies.")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--max-rl-draws", type=int, default=100)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    a = ap.parse_args()

    root = Path(a.project_root).expanduser().resolve()
    man = (Path(a.seed_manifest).expanduser().resolve() if a.seed_manifest
           else root / "configs/repeated_partition_seeds.csv")
    if not man.exists():
        raise FileNotFoundError(f"Missing repeated-seed manifest: {man}")
    seeds = pd.read_csv(man).seed.astype(int).tolist()

    py = choose_python(a.analysis_python)
    rl_out = find_rl_out(root, a.rl_out, seeds)

    # Fail before any expensive run if a repeated M6 posterior family is absent.
    objs = read_cv_objects(rl_out)
    missing = [seed for seed in seeds if not cv_ready(objs, seed)]
    print("Resolved dependencies:")
    print("  analysis_python:", py)
    print("  seed_manifest:  ", man)
    print("  rl_out:         ", rl_out)
    print("  seeds:          ", " ".join(map(str, seeds)))
    if missing:
        raise RuntimeError(
            "Repeated EEG-computational study stability requires completed M6 5-fold CV posteriors "
            f"for every manifest seed. Missing seeds: {missing}\n\n"
            "Missing cross-validated posterior sets must be generated in the authorized controlled workspace before this robustness analysis is run."
        )

    outroot = (Path(a.out_root).expanduser().resolve() if a.out_root else
               root / "reproduced/controlled_rerun/robustness/repeated_partition_cue_stability")
    outroot.mkdir(parents=True, exist_ok=True)

    action = root / "code/02_primary_cue/cue_action_prediction.py"
    enc = root / "code/02_primary_cue/cue_rl_state_encoding.py"
    acthelper = root / "shared/action/action_occurrence_benchmark.py"
    nested = root / "shared/rl/nested_crossvalidated_latent_generation.py"
    rl = root / "shared/rl/actor_control_model_family.py"
    ddm = root / "shared/ddm/one_boundary_diffusion.py"
    data_root = root / "controlled_inputs/behavioral_trial_data"
    eeg_table = root / "controlled_inputs/cue_eeg/canonical_cue_feature_table.csv"

    required = [action, enc, acthelper, nested, rl, ddm, eeg_table]
    bad = [p for p in required if not p.exists()]
    if bad:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(map(str, bad)))

    common = [
        "--nested-script", str(nested),
        "--rl-script", str(rl),
        "--ddm-script", str(ddm),
        "--rl-out", str(rl_out),
        "--data-root", str(data_root),
        "--eeg-table", str(eeg_table),
        "--rl-variant", "M6",
        "--k", "5",
        "--max-rl-draws", str(a.max_rl_draws),
        "--deadline-ms", "700",
        "--n-bootstrap", str(a.n_bootstrap),
    ]

    if a.preflight_only:
        print("PRECHECK PASS: interpreter, sources, manifest and all M6 repeated-CV seeds are available.")
        return 0

    arows, erows = [], []
    for seed in seeds:
        ao = outroot / f"seed_{seed}/action"
        eo = outroot / f"seed_{seed}/encoding"

        if a.force or not (ao / "cue_action_planned_contrasts.csv").exists():
            run([
                str(py), str(action), *common,
                "--action-script", str(acthelper),
                "--out", str(ao),
                "--seed", str(seed),
                "--Cs", ".0003", ".001", ".003", ".01", ".03", ".1", ".3", "1", "3", "10",
                "--inner-k", "4",
            ])
        if a.force or not (eo / "cue_rl_state_encoding_planned_contrasts.csv").exists():
            run([
                str(py), str(enc), *common,
                "--eeg-action-script", str(action),
                "--out", str(eo),
                "--seed", str(seed),
                "--alphas", ".0001", ".001", ".01", ".1", "1", "10", "100", "1000", "10000",
                "--inner-k", "4",
            ])

        x = pd.read_csv(ao / "cue_action_planned_contrasts.csv")
        x["outer_seed"] = seed
        arows.append(x)
        y = pd.read_csv(eo / "cue_rl_state_encoding_planned_contrasts.csv")
        y["outer_seed"] = seed
        erows.append(y)

    A = pd.concat(arows, ignore_index=True)
    E = pd.concat(erows, ignore_index=True)
    A.to_csv(outroot / "action_contrasts_all_seeds.csv", index=False)
    E.to_csv(outroot / "encoding_contrasts_all_seeds.csv", index=False)

    A.groupby(["model", "reference", "label"], as_index=False).agg(
        n_seeds=("outer_seed", "nunique"),
        mean_delta=("mean_subject_delta_elpd", "mean"),
        median_delta=("mean_subject_delta_elpd", "median"),
        min_delta=("mean_subject_delta_elpd", "min"),
        max_delta=("mean_subject_delta_elpd", "max"),
        positive_seeds=("mean_subject_delta_elpd", lambda x: int((x > 0).sum())),
    ).to_csv(outroot / "action_seed_stability_summary.csv", index=False)

    E.groupby(["target", "model", "reference", "label"], as_index=False).agg(
        n_seeds=("outer_seed", "nunique"),
        mean_gain=("mean_subject_mse_gain_z", "mean"),
        median_gain=("mean_subject_mse_gain_z", "median"),
        min_gain=("mean_subject_mse_gain_z", "min"),
        max_gain=("mean_subject_mse_gain_z", "max"),
        positive_seeds=("mean_subject_mse_gain_z", lambda x: int((x > 0).sum())),
    ).to_csv(outroot / "encoding_seed_stability_summary.csv", index=False)

    (outroot / "analysis_settings.json").write_text(json.dumps({
        "seeds": seeds,
        "analysis_python": str(py),
        "rl_out": str(rl_out),
        "max_rl_draws": a.max_rl_draws,
        "n_bootstrap": a.n_bootstrap,
        "status": "stability extension; seed2026 remains historical primary",
    }, indent=2), encoding="utf-8")
    print("Wrote", outroot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
