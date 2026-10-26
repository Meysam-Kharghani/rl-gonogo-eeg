# Quality assurance

`tools/validate_repository.py` is a standard-library release validator. It checks:

- release version, MIT license, and verified machine-readable author/citation metadata;
- required repository structure and stage-gated decision documentation;
- Python syntax compilation;
- JSON and delimited-table parsing;
- formal publication-facing naming and nonportable-path exclusion;
- participant-identifier leakage at the public boundary;
- SHA-256 and byte-size agreement with `provenance/release_manifest.tsv`.

Scientific QA is distinct from repository QA. The evidence hierarchy preserves candidate development, superseded branches, primary cue analyses, primary response analyses, and completed robustness work. Analyses stopped at an earlier decision gate are not assigned artificial downstream sensitivity analyses.

## Reporting QA

`make figures` regenerates the publication-facing figures into `reproduced/figures/` and compares the generated assets with the frozen references in `results/reporting/figures/`.

`make quality-check` runs repository validation followed by figure regeneration/verification. The GitHub Actions workflow performs the same public-boundary checks in a clean hosted environment.

## Controlled-input QA

`make controlled-preflight` verifies the presence of authorized participant-level inputs required for full reruns. These inputs are intentionally absent from a public clone.

## Deterministic permutation-seed refresh

The public implementations of response-confound triangulation and trial-level RL–EEG inference use deterministic SHA-256-derived sub-seeds. For release 1.0.1, the seed-sensitive aggregate outputs for these two robustness branches were refreshed with the deterministic seed contract (`base_seed = 2026`, `seed_derivation = sha256_stable_subseed_v1`). Non-seed-sensitive estimates, counts, model fits, and grouped-prediction quantities reproduced within `1e-10`. The inferential decisions were unchanged: all 33 response-matched tests remained FDR-supported, the within-NoGo same-target family retained zero FDR-supported tests, and the trial-level Freedman–Lane family retained zero FDR-supported tests.

Participant-level matching records and trial-level analysis tables used for the controlled refresh remain outside the public boundary. Only aggregate outputs are distributed. See `docs/deterministic_robustness_refresh.md`.
