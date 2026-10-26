# Controlled rerun commands

These commands are templates for an **authorized controlled-input rerun**. They write to `reproduced/controlled_rerun/` and do not overwrite the frozen public results under `results/`. Run from the repository root after materializing the inputs described in `docs/controlled_input_schema.md`.

## 1. Canonical cue table

If the fixed trial-level EEG feature tables and cross-fitted RL trial table are available, rebuild the canonical cue table as follows:

```bash
python code/02_primary_cue/build_cue_feature_table.py \
  --erp controlled_inputs/eeg_features/erp_trial_features.csv \
  --tf controlled_inputs/eeg_features/time_frequency_trial_features.csv \
  --entropy controlled_inputs/eeg_features/entropy_trial_features.csv \
  --latents controlled_inputs/crossfitted_rl/crossfitted_trial_latents.csv \
  --out reproduced/controlled_rerun/cue_feature_table
```

For the frozen downstream benchmarks, the controlled canonical table may instead be materialized directly as `controlled_inputs/cue_eeg/canonical_cue_feature_table.csv`.

## 2. Cue-window action prediction

```bash
python code/02_primary_cue/cue_action_prediction.py \
  --nested-script shared/rl/nested_crossvalidated_latent_generation.py \
  --rl-script shared/rl/actor_control_model_family.py \
  --ddm-script shared/ddm/one_boundary_diffusion.py \
  --action-script shared/action/action_occurrence_benchmark.py \
  --rl-out controlled_inputs/rl_crossvalidated_posteriors \
  --data-root controlled_inputs/behavioral_trial_data \
  --eeg-table controlled_inputs/cue_eeg/canonical_cue_feature_table.csv \
  --out reproduced/controlled_rerun/cue_action_prediction
```

## 3. Cue-window RL-state encoding

```bash
python code/02_primary_cue/cue_rl_state_encoding.py \
  --nested-script shared/rl/nested_crossvalidated_latent_generation.py \
  --rl-script shared/rl/actor_control_model_family.py \
  --ddm-script shared/ddm/one_boundary_diffusion.py \
  --eeg-action-script code/02_primary_cue/cue_action_prediction.py \
  --rl-out controlled_inputs/rl_crossvalidated_posteriors \
  --data-root controlled_inputs/behavioral_trial_data \
  --eeg-table controlled_inputs/cue_eeg/canonical_cue_feature_table.csv \
  --out reproduced/controlled_rerun/cue_rl_state_encoding
```

## 4. Response-locked performance-monitoring contrasts

```bash
python code/03_primary_response/performance_monitoring_contrasts.py \
  --erp controlled_inputs/eeg_features/erp_trial_features.csv \
  --tf controlled_inputs/eeg_features/time_frequency_trial_features.csv \
  --entropy controlled_inputs/eeg_features/entropy_trial_features.csv \
  --latents controlled_inputs/crossfitted_rl/crossfitted_trial_latents.csv \
  --out reproduced/controlled_rerun/response_monitoring
```

## 5. Corrected between-participant RL–EEG association analysis

The response-monitoring command produces the analysis table required by the association analysis.

```bash
python code/03_primary_response/between_participant_rl_eeg_associations.py \
  --analysis-table reproduced/controlled_rerun/response_monitoring/canonical_association_analysis_table.csv \
  --out reproduced/controlled_rerun/between_participant_rl_eeg_associations
```

## 6. Completed robustness analyses

The following scripts use formal controlled-input defaults and write below `reproduced/controlled_rerun/robustness/`:

```bash
python code/04_robustness/repeated_partition_cue_benchmarks.py
python code/04_robustness/richer_history_consistency.py
python code/04_robustness/response_confound_triangulation.py
python code/04_robustness/response_reliability.py
python code/04_robustness/expanded_between_participant_associations.py
python code/04_robustness/trial_level_crossfitted_rl_eeg.py
```

`response_confound_triangulation.py` and `trial_level_crossfitted_rl_eeg.py` use deterministic SHA-256-derived permutation/bootstrap sub-seeds. Release 1.0.1 aligns the frozen public aggregate outputs for those branches with that deterministic seed contract; see `docs/quality_assurance.md` and `docs/deterministic_robustness_refresh.md`.

## 7. EEG feature regeneration from processed continuous EEG

The feature-generation scripts under `code/01_feature_generation/` require the controlled processed continuous EEG/event resources and participant metadata. Example interfaces are available via `--help`. Raw acquisition files are not distributed, so the repository does not claim raw-acquisition-to-report reproducibility.

## 8. Historical and stopped development branches

`code/00_model_development/` preserves available historical source snapshots. `deferred_extensions/` contains code-only branches stopped by design and excluded from the final evidence hierarchy. See the decision log before interpreting or rerunning any historical branch.
