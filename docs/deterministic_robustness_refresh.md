# Deterministic robustness refresh

Release 1.0.1 closes the deterministic sub-seed portability item documented in release 1.0.0. The affected public analyses are:

- `results/robustness/response_confound_triangulation/`
- `results/robustness/trial_level_crossfitted_rl_eeg/`

## Seed contract

Both branches use `base_seed = 2026` and SHA-256-derived stable sub-seeds (`sha256_stable_subseed_v1`) for seed-sensitive bootstrap/permutation operations.

## Reproduction invariants

When the refreshed outputs were compared with the previously frozen outputs, all non-seed-sensitive quantities reproduced within `1e-10`. This includes sample/count fields, paired-effect point estimates, factorial-model coefficients and fit statistics, cluster-robust trial-level coefficients, and grouped-prediction metrics.

## Seed-sensitive changes

Only the expected resampling-derived quantities changed: bootstrap confidence limits and sign-flip p/FDR values in response matching, and Freedman–Lane p/FDR values in the trial-level analysis. The scientific decisions were stable:

- response-matched family: 33 FDR-supported tests before and after refresh;
- within-NoGo same-target family: 0 FDR-supported tests before and after refresh;
- trial-level Freedman–Lane family: 0 FDR-supported tests before and after refresh;
- minimum trial-level Freedman–Lane FDR value changed from approximately `0.25735` to `0.55759`, remaining well above the inferential threshold.

The response target × response factorial table and grouped-prediction tables were numerically unchanged.

## Public/private boundary

The controlled refresh also produced participant-level matched-pair records and a participant/trial-level analysis table. Those artifacts contain participant identifiers and are intentionally excluded from the public repository. Release 1.0.1 distributes only aggregate outputs.

The three direct per-caliper aggregate response-matching effect tables are synchronized with the refreshed combined caliper table so that the public repository contains one internally consistent numerical record.
