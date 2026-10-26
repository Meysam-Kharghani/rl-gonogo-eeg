# Manuscript–repository alignment

This document maps the publication-facing scientific narrative to the frozen public evidence in this repository.

## Main evidential structure

### Incremental cue-window prediction

- **Action prediction:** `results/cue_action/key_contrasts.csv`
- **RL-state encoding:** `results/cue_encoding/key_contrasts.csv`
- **Repeated-partition stability:** `results/robustness/repeated_partition_cue_stability/`
- **Richer lagged-history comparison:** `results/robustness/richer_history_consistency/`

Publication-facing interpretation: the fixed nominal cue-window EEG representation provides little or no held-out incremental information beyond behavioral/computational references. The conclusion remains stable across repeated participant partitions and against a richer lagged-history comparator.

### Response-locked performance monitoring

- **Primary response-signature family:** `results/response_monitoring/`
- **Response-time matching and contextual controls:** `results/robustness/response_confound_triangulation/`
- **Reliability:** `results/robustness/response_reliability/`

Publication-facing interpretation: false alarms and Go hits differ strongly in the fixed response-locked signatures. The contrasts are described as performance-monitoring differences rather than pure error-specific effects because false alarm versus Go hit also changes target context. Reliability is heterogeneous and constrains individual-difference interpretation.

### Computational–neural association

- **Corrected fixed 18-test between-participant family:** `results/between_participant_rl_eeg_associations/`
- **Expanded 33-test family:** `results/robustness/expanded_between_participant_associations/`
- **Trial-level cross-fitted analysis:** `results/robustness/trial_level_crossfitted_rl_eeg/`

Publication-facing interpretation: participant-level RL summaries show a limited set of nuisance-adjusted associations with response-locked neural contrasts. A broader multiplicity family retains a smaller robust core, while the trial-level analysis yields no multiplicity-corrected evidence and near-zero grouped predictive performance. The repository therefore supports between-participant covariation, not trial-wise neural implementation or causal mapping.

## Temporal-support boundary

The cue-window label refers to nominal event timing. The recoverable signal-processing chain includes non-causal filtering, so these analyses are not presented as prospective real-time decoding. Subject-level cross-fitting addresses computational/model-fitting leakage but cannot reverse temporal mixing introduced by prior signal processing.

## Technical identifiers

Technical IDs such as `M6`, `E0–E9`, and `N0–N4` are retained only for computational traceability. Manuscript prose should use the conceptual labels defined in `docs/model_nomenclature.md`.

## Historical development evidence

Development and superseded analyses under `results/model_development/` document how the final pipeline was reached. They are not interchangeable with final inferential evidence. Stage advancement and stopping decisions are documented in `docs/analysis_lineage_and_decision_log.md` and `docs/analysis_stage_registry.tsv`.
