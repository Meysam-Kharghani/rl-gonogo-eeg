# Analysis lineage and decision log — EEG-computational study

## Reading rule

EEG-computational study also followed staged development. Analyses that failed a preceding gate, were contaminated by non-EEG/design variables, relied on superseded alignment, or were replaced by stricter inference were not automatically propagated through every later sensitivity analysis. The final package keeps their aggregate evidence under `results/model_development/` with explicit status labels.

## Stage 1 — EEG processing provenance and event alignment

The available project boundary begins from processed continuous EEGLAB data. Event/trial alignment and response-marker consistency were audited before the final feature tables were frozen. The present single-trial pipeline did not inherit an undocumented ICA step.

**Decision:** feature extraction proceeded from the documented processed-data boundary.

## Stage 2 — ERP, time-frequency, and derived feature development

ERP-like response/cue features and time-frequency features were generated and validated. Entropy/fraction features were derived from the time-frequency representation and are not treated as an independent neural modality.

Supportive waveform, trial-count, spatial/topographic, and targeted validations were retained where informative.

## Stage 3 — superseded exploratory decoding and association analyses

Earlier multivariate decoding and incremental association analyses were useful development evidence but were not advanced as primary evidence when feature contamination, alignment, broad multiplicity, or inferential limitations were identified. Corrected/sanitized analyses remain in the model-development archive for transparency.

**Decision:** `SUPERSEDED`; do not use these historical outputs for final cue or association analysis claims.

## Stage 4 — canonical cue table and leakage exclusions

The final cue table aligned EEG trials one-to-one with the fold-specific RL trial table. Response-locked/target-locked features, outcome metadata, and any embedded computational columns were excluded from the cue predictor block. Cross-fitted RL quantities were generated from train-fold-only posteriors.

**Decision:** freeze the 37-feature nominal cue-EEG block and the fold-aligned computational backbone.

## Stage 5 — primary cue action benchmark

The primary action comparison tested combined cue EEG beyond the selected behavioral/computational action baseline. No detectable held-out increment was obtained.

**Decision:** cue EEG did not advance as an incremental action predictor; the null itself became the result to stress-test across partitions and stronger baselines.

## Stage 6 — primary cue RL-state encoding benchmark

The same cue feature set was tested for held-out encoding of policy probability, actor Q-difference, and control beyond task design and lagged behavioral history. Combined cue EEG did not improve held-out target error.

**Decision:** no positive cue-encoding branch was advanced for post-hoc feature mining.

## Stage 7 — response-locked performance monitoring

A fixed six-signature response family compared false alarms with Go hits. Strong response-locked performance-monitoring contrasts motivated focused robustness work.

**Decision:** response signatures were `ADVANCED`, with the explicit interpretation that false-alarm versus Go-hit contrasts jointly vary error status and target context.

## Stage 8 — corrected between-participant RL–EEG association analysis

The canonical association analysis used participant-level response-signature contrasts and cross-fitted RL summaries with group-adjusted Freedman–Lane inference across a fixed 18-test family.

**Decision:** retain as supportive between-participant covariation; do not interpret as trial-wise or causal neural encoding.

## Stage 9 — cue stability robustness

Repeated outer partitions and a richer behavioral-history comparator were applied to the final cue analyses. The cue-EEG null remained stable across all repeated partitions and against the richer history comparator.

**Decision:** stability evidence strengthens the cue-null conclusion; no new cue-feature search was justified.

## Stage 10 — response confound and reliability robustness

RT-matched false-alarm versus Go-hit effects remained strong, but within-NoGo correct-rejection versus false-alarm effects and target × response interactions did not survive multiplicity correction. Split-half reliability varied substantially across response signatures.

**Decision:** retain “performance-monitoring contrast” terminology and explicitly report heterogeneous reliability; do not claim pure error specificity.

## Stage 11 — expanded association family

A broader 33-test Freedman–Lane family retained a smaller core of FDR-supported associations than the original fixed 18-test family.

**Decision:** emphasize the small robust core and avoid presenting every original RL–EEG association as equally stable.

## Stage 12 — trial-level cross-fitted RL–EEG analysis

A trial-level analysis using cross-fitted RL predictors and response EEG outcomes yielded no association surviving multiplicity correction and near-zero grouped-prediction R².

**Decision:** the positive association analysis is specifically a between-participant covariation result; no trial-wise neural implementation claim is supported.

## Stage 13 — non-advanced planned extensions

Additional code-only ideas such as post-error extensions or alternative marker-lock workflows were not required to support the final paper after the main gates and robustness analyses were resolved.

**Decision:** not part of the active scientific evidence package. Their non-execution is recorded in the historical project archive rather than presented as missing final analyses.
