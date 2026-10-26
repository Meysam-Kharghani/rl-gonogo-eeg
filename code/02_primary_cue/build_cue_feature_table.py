#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


ERP_PREFIXES = (
    "cue_cnv_early_cz_100_200_",
    "cue_cnv_late_cz_200_500_",
    "cue_n2_fz_200_350_",
    "cue_n2_cz_200_350_",
    "cue_p3_cz_250_450_",
)

ERP_SUFFIXES = (
    "_mean_uv",
    "_min_uv",
    "_max_uv",
    "_minlat_ms",
    "_maxlat_ms",
    "_slope_uv_per_s",
)

TF_FEATURES = [
    "cue_theta_fz_200_500",
    "cue_theta_cz_200_500",
    "cue_alpha_pz_200_500",
    "cue_beta_cz_200_500",
]

# Primary entropy features are restricted to the explicitly timed cue window.
# "allwindows" and "global" features remain excluded because their extraction
# intervals are not part of the predefined nominal cue-window specification.
ENTROPY_FEATURES = [
    "cue_cz_200_500_spectral_entropy",
    "cue_cz_200_500_theta_fraction",
    "cue_cz_200_500_beta_fraction",
]

LATENT_COLS = [
    "p_press",
    "belief_nogo_current",
    "associability_current",
    "control_current",
    "actor_qdiff_current",
]

CANONICAL_COLS = [
    "subject_index",
    "subject_id",
    "group",
    "trial_index",
    "cue_idx",
    "cue_type",
    "is_nogo",
    "action_press",
    "correct",
    "error",
    "fa",
    *LATENT_COLS,
    "nested_rl_fold",
    "nested_rl_seed",
    "nested_rl_variant",
    "nested_rl_source",
]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--erp", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--entropy", required=True)
    p.add_argument("--latents", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def assert_unique(df, name):
    ndup = df.duplicated(["subject_id", "trial_index"]).sum()
    if ndup:
        raise ValueError(
            f"{name}: {ndup} duplicate subject/trial rows"
        )


def normalize_text(x):
    return str(x).strip().lower()


def prepare_eeg(df, name):
    df = df.copy()

    assert_unique(df, name)

    df["eeg_trial_index"] = pd.to_numeric(
        df["trial_index"],
        errors="raise",
    ).astype(int)

    # EEG tables are one-based; canonical RL table is zero-based.
    df["canonical_trial_index"] = df["eeg_trial_index"] - 1

    return df


def verify_alignment(eeg, lat, name):
    audit = eeg[
        [
            "subject_id",
            "eeg_trial_index",
            "canonical_trial_index",
            "cue_type",
            "response_present",
        ]
    ].merge(
        lat[
            [
                "subject_id",
                "trial_index",
                "cue_type",
                "action_press",
            ]
        ].rename(
            columns={
                "trial_index": "canonical_trial_index",
                "cue_type": "canonical_cue_type",
                "action_press": "canonical_action_press",
            }
        ),
        on=["subject_id", "canonical_trial_index"],
        how="left",
        validate="one_to_one",
    )

    matched = audit["canonical_cue_type"].notna().mean()

    cue_match = (
        audit["cue_type"].map(normalize_text)
        ==
        audit["canonical_cue_type"].map(normalize_text)
    ).mean()

    action_match = (
        pd.to_numeric(
            audit["response_present"],
            errors="coerce",
        )
        ==
        pd.to_numeric(
            audit["canonical_action_press"],
            errors="coerce",
        )
    ).mean()

    result = {
        "source": name,
        "n_rows": len(audit),
        "canonical_match_fraction": float(matched),
        "cue_match_fraction": float(cue_match),
        "action_match_fraction": float(action_match),
    }

    if matched != 1.0:
        raise ValueError(
            f"{name}: incomplete canonical mapping: {matched}"
        )

    if cue_match != 1.0:
        raise ValueError(
            f"{name}: cue mismatch: {cue_match}"
        )

    if action_match != 1.0:
        raise ValueError(
            f"{name}: action mismatch: {action_match}"
        )

    return result


def main():
    a = args()

    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    erp = prepare_eeg(
        pd.read_csv(a.erp),
        "ERP",
    )

    tf = prepare_eeg(
        pd.read_csv(a.tf),
        "TF",
    )

    ent = prepare_eeg(
        pd.read_csv(a.entropy),
        "ENTROPY",
    )

    lat = pd.read_csv(a.latents)

    missing = [
        c for c in CANONICAL_COLS
        if c not in lat.columns
    ]

    if missing:
        raise ValueError(
            f"Canonical LATENTS missing columns: {missing}"
        )

    lat = lat[CANONICAL_COLS].copy()

    assert_unique(lat, "LATENTS")

    if len(lat) != 15200:
        print(
            "WARNING: canonical row count is",
            len(lat),
            "rather than 15200",
        )

    # Verify trial alignment before feature construction.
    audits = [
        verify_alignment(erp, lat, "ERP"),
        verify_alignment(tf, lat, "TF"),
        verify_alignment(ent, lat, "ENTROPY"),
    ]

    # Select ERP cue-only columns.
    erp_features = [
        c for c in erp.columns
        if (
            c.startswith(ERP_PREFIXES)
            and c.endswith(ERP_SUFFIXES)
        )
    ]

    missing_tf = [
        c for c in TF_FEATURES
        if c not in tf.columns
    ]

    missing_ent = [
        c for c in ENTROPY_FEATURES
        if c not in ent.columns
    ]

    if missing_tf:
        raise ValueError(
            f"Missing TF features: {missing_tf}"
        )

    if missing_ent:
        raise ValueError(
            f"Missing entropy features: {missing_ent}"
        )

    if not erp_features:
        raise ValueError(
            "No ERP cue-only features selected"
        )

    # Canonical table starts from the nested RL/behavior table.
    data = lat.copy()

    data["eeg_trial_index"] = (
        data["trial_index"].astype(int) + 1
    )

    # ERP block
    e = erp[
        [
            "subject_id",
            "canonical_trial_index",
            *erp_features,
        ]
    ].copy()

    e = e.rename(
        columns={
            "canonical_trial_index": "trial_index",
            **{
                c: f"erp__{c}"
                for c in erp_features
            },
        }
    )

    data = data.merge(
        e,
        on=["subject_id", "trial_index"],
        how="left",
        validate="one_to_one",
    )

    # TF block
    t = tf[
        [
            "subject_id",
            "canonical_trial_index",
            *TF_FEATURES,
        ]
    ].copy()

    t = t.rename(
        columns={
            "canonical_trial_index": "trial_index",
            **{
                c: f"tf__{c}"
                for c in TF_FEATURES
            },
        }
    )

    data = data.merge(
        t,
        on=["subject_id", "trial_index"],
        how="left",
        validate="one_to_one",
    )

    # Entropy block
    en = ent[
        [
            "subject_id",
            "canonical_trial_index",
            *ENTROPY_FEATURES,
        ]
    ].copy()

    en = en.rename(
        columns={
            "canonical_trial_index": "trial_index",
            **{
                c: f"entropy__{c}"
                for c in ENTROPY_FEATURES
            },
        }
    )

    data = data.merge(
        en,
        on=["subject_id", "trial_index"],
        how="left",
        validate="one_to_one",
    )

    if len(data) != len(lat):
        raise RuntimeError(
            "Row count changed during EEG merge"
        )

    eeg_features = [
        c for c in data.columns
        if c.startswith(
            ("erp__", "tf__", "entropy__")
        )
    ]

    missingness = (
        data[eeg_features]
        .isna()
        .mean()
        .sort_values(ascending=False)
    )

    # Verify by column name that excluded feature families are absent.
    forbidden_tokens = [
        "resp_",
        "response",
        "target_",
        "action_press",
        "outcome",
        "correct",
        "error",
        "fa",
        "p_press",
        "control_current",
        "actor_qdiff_current",
        "belief_nogo_current",
        "associability_current",
        "policy_entropy",
        "action_surprise",
        "trial_class_surprise",
    ]

    flagged = [
        c for c in eeg_features
        if any(
            token in c.lower()
            for token in forbidden_tokens
        )
    ]

    if flagged:
        raise RuntimeError(
            "Forbidden feature names entered EEG block: "
            + ", ".join(flagged)
        )

    csv_path = (
        out
        / "canonical_cue_eeg_seed2026.csv"
    )

    data.to_csv(csv_path, index=False)

    pd.DataFrame(audits).to_csv(
        out / "cue_eeg_alignment_audit.csv",
        index=False,
    )

    pd.DataFrame({
        "feature": eeg_features,
        "missing_fraction": [
            float(missingness[c])
            for c in eeg_features
        ],
        "family": [
            (
                "ERP"
                if c.startswith("erp__")
                else "TF"
                if c.startswith("tf__")
                else "ENTROPY"
            )
            for c in eeg_features
        ],
    }).to_csv(
        out / "cue_eeg_feature_manifest.csv",
        index=False,
    )

    spec = {
        "primary_eeg_timing":
            "nominal cue-locked only; windows nominally pre-target",
        "canonical_trial_mapping":
            "EEG trial_index - 1 = nested RL trial_index",
        "rl_source":
            str(Path(a.latents).resolve()),
        "n_rows":
            int(len(data)),
        "n_subjects":
            int(data["subject_id"].nunique()),
        "erp_features":
            [f"erp__{c}" for c in erp_features],
        "tf_features":
            [f"tf__{c}" for c in TF_FEATURES],
        "entropy_features":
            [
                f"entropy__{c}"
                for c in ENTROPY_FEATURES
            ],
        "primary_exclusions": [
            "all response-locked EEG",
            "all target-locked EEG",
            "embedded RL columns from EEG-derived tables",
            "computational entropy/surprise columns",
            "response/outcome metadata as EEG predictors",
        ],
        "secondary_only": [
            "target-locked EEG",
            "response-locked ERP/TF/entropy",
            "cue allwindows/global entropy excluded because extraction intervals are outside the predefined nominal cue-window specification",
        ],
    }

    (
        out / "CUE_EEG_FEATURE_SPEC.json"
    ).write_text(
        json.dumps(
            spec,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== Alignment audit ===")
    print(
        pd.DataFrame(audits)
        .to_markdown(index=False)
    )

    print("\n=== Primary feature counts ===")
    print("ERP:", len(erp_features))
    print("TF:", len(TF_FEATURES))
    print("Entropy:", len(ENTROPY_FEATURES))
    print("Total:", len(eeg_features))

    print("\n=== Highest missingness ===")
    print(
        missingness.head(15).to_string()
    )

    print("\nCanonical table:")
    print(csv_path)

    print("\nFeature specification:")
    print(
        out / "CUE_EEG_FEATURE_SPEC.json"
    )


if __name__ == "__main__":
    main()
