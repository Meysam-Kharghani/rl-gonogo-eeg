# Controlled input schema

Full downstream reruns require controlled participant-level inputs that are not redistributed publicly. The publication-facing code uses the following formal materialization layout by default; explicit paths may be supplied where a script provides an override.

```text
controlled_inputs/
  eeg_source_metadata/
    participant_trial_metadata.xlsx       Controlled participant/trial metadata used for feature regeneration
  processed_eeg/                           Controlled continuous EEGLAB .set/.fdt resources
  behavioral_trial_data/                   Participant trial-level behavioral source files
  rl_crossvalidated_posteriors/            Fold-specific train-only RL posterior/CV artifacts
  cue_eeg/
    canonical_cue_feature_table.csv         One row per participant × trial, fixed cue feature block
  eeg_features/
    erp_trial_features.csv
    time_frequency_trial_features.csv
    entropy_trial_features.csv
  crossfitted_rl/
    crossfitted_trial_latents.csv
    participant_rl_behavior_summary.csv
```

## Behavioral / RL information

Required conceptual fields include participant identifier, trial index, risk-group indicator, cue condition, Go/NoGo target class, response presence, response time, and outcome/history variables.

For every outer fold, the RL posterior must have been fit on outer-training participants only. Cross-fitted trial quantities include policy probability, actor Q-difference, and control.

## Canonical cue EEG table

The cue table contains the fixed 37-feature nominal cue-window block defined by `results/cue_feature_spec/`. Response-locked and target-locked EEG, response/outcome metadata, and embedded computational columns are not permitted as cue EEG predictors.

## Response-monitoring inputs

The response branch requires participant-level response trials/features sufficient to derive the fixed performance-monitoring signatures (ERN, Pe, response theta, response beta, and the two derived band-fraction summaries). Event codes and units must follow `docs/eeg_preprocessing_provenance.md` and `docs/temporal_provenance.md`.

## Optional processed-EEG feature-regeneration inputs

To rerun `code/01_feature_generation/` from processed continuous EEG rather than starting from distributed-schema trial feature tables, authorized users additionally need:

- the processed continuous EEGLAB `.set/.fdt` resources;
- the participant behavioral/metadata table used to align event streams and response times;
- event-code mapping and preprocessing provenance consistent with `docs/eeg_preprocessing_provenance.md`.

These resources are controlled and are not represented as empty public files. The repository intentionally distinguishes this processed-EEG starting point from unavailable raw-acquisition data.
