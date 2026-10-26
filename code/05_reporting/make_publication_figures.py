#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RELEASE_LABEL = "rl-eeg-prediction-performance-monitoring reproducibility release"
RELEASE_DATE = "2026-10-25"
plt.rcParams["svg.hashsalt"] = "rl-eeg-prediction-performance-monitoring-release"


def save(fig, out: Path, stem: str, dpi: int = 300) -> None:
    fixed_dt = datetime(2026, 10, 25, tzinfo=timezone.utc)
    for ext in ("png", "pdf", "svg"):
        if ext == "png":
            metadata = {"Software": RELEASE_LABEL}
        elif ext == "pdf":
            metadata = {
                "Creator": RELEASE_LABEL,
                "Producer": "matplotlib",
                "CreationDate": fixed_dt,
                "ModDate": fixed_dt,
            }
        else:
            metadata = {"Date": RELEASE_DATE, "Creator": RELEASE_LABEL}
        fig.savefig(
            out / f"{stem}.{ext}",
            dpi=(dpi if ext == "png" else None),
            bbox_inches="tight",
            metadata=metadata,
        )
    plt.close(fig)


def forest(d, est, lo, hi, label, xlabel, title, out, stem):
    d = d.reset_index(drop=True).copy()
    y = np.arange(len(d))[::-1]
    x = pd.to_numeric(d[est], errors="coerce").to_numpy(float)
    l = pd.to_numeric(d[lo], errors="coerce").to_numpy(float)
    h = pd.to_numeric(d[hi], errors="coerce").to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.8, max(4.0, 0.55 * len(d) + 1.6)))
    ax.errorbar(x, y, xerr=np.vstack([x - l, h - x]), fmt="o", capsize=3, linewidth=1.2)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d[label].astype(str))
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.22)
    save(fig, out, stem)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out", default="results/reporting/figures")
    args = ap.parse_args()
    root = Path(args.results_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Main Figure 1: fixed response-locked performance-monitoring signature family.
    d = pd.read_csv(root / "response_monitoring/primary_standardized_signature_summary.csv")
    labels = {
        "erp_ern": "ERN, Fz, 50–150 ms",
        "erp_pe": "Pe, Cz, 200–400 ms",
        "tf_theta": "Theta, Fz, 0–200 ms",
        "tf_beta": "Beta, Cz, 200–500 ms",
        "entropy_theta": "Theta fraction",
        "entropy_beta": "Beta fraction",
    }
    d = d[d.signature.isin(labels)].copy()
    d["plot_label"] = d.signature.map(labels)
    d["_o"] = pd.Categorical(d.signature, categories=list(labels), ordered=True)
    d = d.sort_values("_o")
    forest(
        d,
        "cohens_dz",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "plot_label",
        "Standardized false-alarm − Go-hit contrast (Cohen's dz)",
        "Response-locked performance-monitoring contrasts",
        out,
        "figure_01_response_monitoring",
    )

    # Main Figure 2: corrected fixed 18-test between-participant association family.
    b = pd.read_csv(root / "between_participant_rl_eeg_associations/primary_18_tests.csv")
    sig_order = ["erp_ern", "erp_pe", "tf_theta", "tf_beta", "entropy_theta", "entropy_beta"]
    pred_order = ["mean_p_press", "mean_actor_qdiff_current", "mean_control_current"]
    sig_names = {
        "erp_ern": "ERN",
        "erp_pe": "Pe",
        "tf_theta": "TF theta",
        "tf_beta": "TF beta",
        "entropy_theta": "Theta fraction",
        "entropy_beta": "Beta fraction",
    }
    pred_names = {
        "mean_p_press": "Policy probability",
        "mean_actor_qdiff_current": "Actor Q-difference",
        "mean_control_current": "Control",
    }
    mat = np.full((6, 3), np.nan)
    qmat = np.full((6, 3), np.nan)
    for i, s in enumerate(sig_order):
        for j, p in enumerate(pred_order):
            q = b[(b.signature == s) & (b.predictor == p)]
            if len(q):
                mat[i, j] = float(q.iloc[0].standardized_beta)
                qmat[i, j] = float(q.iloc[0].primary_freedman_lane_q_fdr_bh)
    lim = np.nanmax(np.abs(mat))
    lim = 1.0 if not np.isfinite(lim) or lim == 0 else lim
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    im = ax.imshow(mat, aspect="auto", vmin=-lim, vmax=lim, cmap="RdBu_r")
    ax.set_xticks(range(3))
    ax.set_xticklabels([pred_names[x] for x in pred_order], rotation=22, ha="right")
    ax.set_yticks(range(6))
    ax.set_yticklabels([sig_names[x] for x in sig_order])
    for i in range(6):
        for j in range(3):
            if np.isfinite(mat[i, j]):
                star = "*" if np.isfinite(qmat[i, j]) and qmat[i, j] < 0.05 else ""
                ax.text(j, i, f"{mat[i, j]:.2f}{star}", ha="center", va="center")
    ax.set_title("Between-participant RL–EEG associations")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Standardized beta")
    save(fig, out, "figure_02_between_participant_rl_eeg_associations")

    # Supplementary Figure 1: cue-window action-prediction contrasts.
    a = pd.read_csv(root / "cue_action/key_contrasts.csv")
    pairs = [
        ("E6", "E0", "Combined cue EEG beyond lagged history"),
        ("E8", "E2", "Combined cue EEG beyond history + actor/control"),
        ("E9", "E7", "Actor/control beyond history + policy + EEG"),
        ("E9", "E8", "Policy beyond history + actor/control + EEG"),
        ("E9", "E6", "Total RL block beyond history + EEG"),
    ]
    rows = []
    for m, r, lbl in pairs:
        q = a[(a.model == m) & (a.reference == r)]
        if len(q):
            x = q.iloc[0].copy()
            x["plot_label"] = lbl
            rows.append(x)
    forest(
        pd.DataFrame(rows),
        "mean_subject_delta_elpd",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "plot_label",
        "Held-out ΔELPD per participant",
        "Nominal cue-window EEG: action prediction",
        out,
        "supplementary_figure_01_cue_action_prediction",
    )

    # Supplementary Figure 2: combined cue-window EEG encoding gains.
    e = pd.read_csv(root / "cue_encoding/key_contrasts.csv")
    names = {
        "logit_p_press": "Policy logit",
        "actor_qdiff_current": "Actor Q-difference",
        "control_current": "Control",
    }
    q = e[(e.model == "N4") & (e.reference == "N0")].copy()
    q["plot_label"] = q.target.map(names)
    forest(
        q,
        "mean_subject_mse_gain_z",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "plot_label",
        "Held-out standardized MSE gain (positive = lower error)",
        "Nominal cue-window EEG: cross-fitted RL-state encoding",
        out,
        "supplementary_figure_02_cue_rl_state_encoding",
    )

    # Supplementary Figure 3: specification robustness for associations supported in the fixed family.
    primary = pd.read_csv(root / "between_participant_rl_eeg_associations/primary_18_tests.csv")
    rb = pd.read_csv(root / "between_participant_rl_eeg_associations/robustness_all.csv")
    supported = primary[primary["primary_freedman_lane_q_fdr_bh"] < 0.05].copy()
    supported = supported.sort_values(["primary_freedman_lane_q_fdr_bh", "signature", "predictor"])
    sig_names = {
        "erp_ern": "ERN",
        "erp_pe": "Pe",
        "tf_theta": "TF theta",
        "tf_beta": "TF beta",
        "entropy_theta": "Theta fraction",
        "entropy_beta": "Beta fraction",
    }
    pred_names = {
        "mean_p_press": "policy probability",
        "mean_actor_qdiff_current": "actor Q-difference",
        "mean_control_current": "control",
    }
    assoc = [
        (r.signature, r.predictor, f"{sig_names.get(r.signature, r.signature)} × {pred_names.get(r.predictor, r.predictor)}")
        for r in supported.itertuples(index=False)
    ]
    analyses = [
        ("primary_group", "Group-adjusted", "o"),
        ("sensitivity_none", "Unadjusted", "s"),
        ("sensitivity_group_winsor", "Group-adjusted, winsorized", "^"),
    ]
    fig, ax = plt.subplots(figsize=(10.2, max(7.0, 0.56 * len(assoc) + 2.2)))
    base_y = np.arange(len(assoc))[::-1]
    offsets = np.linspace(-0.19, 0.19, len(analyses))
    for off, (analysis, label, marker) in zip(offsets, analyses):
        xs, ys, xlo, xhi = [], [], [], []
        for y, (sig, pred, _) in zip(base_y, assoc):
            q = rb[(rb.signature == sig) & (rb.predictor == pred) & (rb.analysis == analysis)]
            if q.empty:
                continue
            r = q.iloc[0]
            beta = float(r.standardized_beta)
            lo = float(r.bootstrap_ci_low)
            hi = float(r.bootstrap_ci_high)
            xs.append(beta)
            ys.append(y + off)
            xlo.append(beta - lo)
            xhi.append(hi - beta)
        ax.errorbar(xs, ys, xerr=np.vstack([xlo, xhi]), fmt=marker, capsize=3, linewidth=1.0, label=label)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(base_y)
    ax.set_yticklabels([x[2] for x in assoc])
    ax.set_xlabel("Standardized beta")
    ax.set_title("Robustness of FDR-supported between-participant associations")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    save(fig, out, "supplementary_figure_03_association_robustness")

    index = """# EEG-computational study figure index

Main figures
- `figure_01_response_monitoring`: response-locked false-alarm-minus-Go-hit performance-monitoring contrasts.
- `figure_02_between_participant_rl_eeg_associations`: corrected group-adjusted 6 × 3 RL–EEG association family; asterisks denote BH-FDR q < .05.

Supplementary figures
- `supplementary_figure_01_cue_action_prediction`: nominal cue-window EEG action-prediction contrasts.
- `supplementary_figure_02_cue_rl_state_encoding`: combined nominal cue-window EEG encoding gains for cross-fitted RL targets.
- `supplementary_figure_03_association_robustness`: specification robustness for associations supported in the fixed 18-test family.

Interpretive limits
- Cue-window features are nominally pre-target but do not establish strictly prospective temporal support after non-causal filtering.
- Response-locked analyses characterize performance-monitoring dynamics and do not prospectively predict the same response.
- False-alarm-minus-Go-hit contrasts differ in both error status and target context and are not pure error-specific effects.
- Participant-level RL–EEG associations are not evidence for trial-wise or causal neural implementation.
"""
    (out / "FIGURE_INDEX.md").write_text(index, encoding="utf-8")


if __name__ == "__main__":
    main()
