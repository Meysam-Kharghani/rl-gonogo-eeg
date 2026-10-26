# Completed robustness analyses

| Analysis | Public result directory | Role |
|---|---|---|
| Repeated cue partitions | `results/robustness/repeated_partition_cue_stability/` | Partition stability of cue action and RL-state encoding results |
| Richer-history cue consistency | `results/robustness/richer_history_consistency/` | Cue result under a richer lagged behavioral-history comparator |
| Response-confound triangulation | `results/robustness/response_confound_triangulation/` | Response-time matching, within-NoGo comparison, and target × response tests |
| Response reliability | `results/robustness/response_reliability/` | Split-half and temporal-stability diagnostics |
| Expanded between-participant association family | `results/robustness/expanded_between_participant_associations/` | Broader 33-test multiplicity family |
| Trial-level cross-fitted RL–EEG | `results/robustness/trial_level_crossfitted_rl_eeg/` | Trial-level association and grouped-prediction stress test |

Deterministic seed refresh: the response-confound and trial-level branches use the release 1.0.1 SHA-256-derived sub-seed outputs; see `docs/deterministic_robustness_refresh.md`.
