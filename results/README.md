# Results index — EEG-computational study

## Cue-window action prediction
`cue_action/` contains the frozen held-out action-prediction contrasts. The combined cue-EEG block does not add detectable held-out information beyond the selected behavioral/computational reference.

## Cue-window RL-state encoding
`cue_encoding/` contains the frozen encoding contrasts for cross-fitted policy probability, actor-value difference, and control. The combined cue-EEG block does not improve held-out target prediction.

## Cue-null robustness
`robustness/repeated_partition_cue_stability/` and `robustness/richer_history_consistency/` show that the cue-null interpretation is stable across repeated outer partitions and against a richer lagged behavioral-history comparator.

## Response-locked performance monitoring
`response_monitoring/` contains the fixed false-alarm-minus-Go-hit response signatures. Response-time matching preserves the effects; same-target and factorial robustness analyses do not support a pure error-specific interpretation after multiplicity correction.

## Between-participant RL–EEG associations
`between_participant_rl_eeg_associations/` contains the corrected fixed 18-test nuisance-adjusted association family. `robustness/expanded_between_participant_associations/` contains the broader 33-test family, which retains a smaller robust core. `robustness/trial_level_crossfitted_rl_eeg/` contains the trial-level analysis, which yields no multiplicity-corrected association and near-zero grouped predictive performance.

The final interpretation therefore distinguishes **between-participant covariation** from **trial-level neural implementation**.

## Reliability
`robustness/response_reliability/` contains response-signature reliability analyses. Reliability is heterogeneous and is an interpretive guardrail for individual-difference associations.
