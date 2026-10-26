# EEG-computational study analysis-development evidence

The public package separates historical/supportive development from the final cue and response analyses and from post-development robustness.

`docs/eeg_analysis_completeness.csv` enumerates the audited analysis families and assigns their current role: canonical, supportive, exploratory, superseded, robustness, or not advanced.

Compact aggregate outputs are distributed when they can be shared without participant identifiers or trial-level participant data. Participant-level EEG/behavior tables, matched-pair rows, fold-specific posterior objects, and processed participant EEG remain outside the public package.

## Historical/superseded analyses

Older decoding and association analyses remain in `results/model_development/` as an audit trail. They are not the inferential basis of the manuscript when they relied on broader feature sets, older alignment, or superseded permutation strategies.

## Final primary analyses

The final cue analyses are the fold-aligned action-prediction and RL-state encoding families under `results/cue_action/` and `results/cue_encoding/`. The final response branch uses the fixed performance-monitoring signature family and the corrected group-adjusted Freedman–Lane association analysis.

## Completed post-development robustness

The package now includes completed output for:

- repeated cue partitions;
- richer-history cue consistency;
- response-confound triangulation/matching;
- response-signature reliability;
- expanded Freedman–Lane association inference;
- trial-level cross-fitted RL–EEG analysis.

These results live under `results/robustness/` and are integrated into `docs/evidence_hierarchy.md`.

## Non-advanced ideas

Post-error and alternative response-marker extraction ideas were not advanced after the principal response/cue questions were resolved and the available timing QC was already essentially exact. Their status is recorded in `docs/eeg_code_only_extensions.csv`; non-execution is a decision outcome, not an unreported completed analysis.

Original raw EEG acquisition files are unavailable. The feature-regeneration boundary begins from the processed continuous EEGLAB inputs described in the preprocessing documentation.
