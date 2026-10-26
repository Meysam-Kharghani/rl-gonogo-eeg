# Separating incremental cue-window EEG prediction from response-locked performance monitoring in cued Go/NoGo

Public analysis repository for the EEG–computational study **“Separating incremental cue-window EEG prediction from response-locked performance monitoring in cued Go/NoGo.”**

**Release:** 1.0.1 (2026-10-25)  
**Repository name:** `rl-gonogo-eeg`  
**License:** MIT  
**Cohort:** 40 participants (20 high-risk-propensity, 20 low-risk-propensity), 380 task trials per participant  
**Public-data boundary:** participant-level EEG, behavioral/RL trial tables, participant identifiers, matched-pair records, and fold-specific RL posterior objects are not redistributed.

## Scientific question

This repository separates three evidential questions that are often conflated in computational EEG studies:

1. **incremental cue-window prediction:** whether a fixed nominal cue-window EEG representation adds held-out information beyond behavioral history and cross-fitted reinforcement-learning (RL) states;
2. **response-locked physiology:** whether robust performance-monitoring differences are present around emitted responses and how those effects behave under response-time matching, target-context controls, and reliability analysis;
3. **computational–neural association:** whether participant-level RL summaries covary with response-locked neural contrasts, and whether those associations extend to trial-level within-participant evidence.

The final evidence is stage-gated. Candidate feature/decoding branches that failed an earlier gate, were superseded by corrected alignment or inference, or became irrelevant after a null result were **stopped by design** rather than being propagated through every later sensitivity analysis. The complete path is documented in `docs/analysis_lineage_and_decision_log.md` and `docs/analysis_stage_registry.tsv`.

## Main conclusions represented by this release

- Combined nominal cue-window EEG did not improve held-out action prediction beyond the selected behavioral/computational reference. Across 10 repeated outer partitions, the combined cue-EEG increment was negative in every partition.
- Combined cue-window EEG did not improve held-out encoding of cross-fitted policy probability, actor Q-difference, or control. The repeated-partition conclusion remained negative for all three targets.
- A richer strictly lagged behavioral-history comparator preserved the cue-EEG null.
- False alarms and Go hits showed strong response-locked differences in ERN, Pe, theta, beta, and derived band-fraction summaries. Response-time matching preserved the response differences, whereas within-NoGo and target-by-response analyses did not establish pure error specificity after multiplicity correction.
- Response-signature reliability was heterogeneous, which constrains individual-difference interpretation.
- The corrected 18-test participant-level RL–EEG association analysis contained multiple FDR-supported associations; a broader 33-test family retained a smaller robust core.
- A trial-level cross-fitted RL–EEG analysis yielded no association surviving multiplicity correction and near-zero grouped predictive performance. The positive association analysis is therefore interpreted as **between-participant covariation**, not trial-wise neural implementation.

## Temporal and inferential limits

Cue features are **nominally cue-window / pre-target**, not strictly prospective neural measurements. The available processing chain contains non-causal filtering, so subject-level cross-validation prevents model-fitting leakage but cannot reverse temporal mixing introduced by signal processing.

Response-locked false-alarm versus Go-hit contrasts also differ in target context as well as error status. They are therefore described as **performance-monitoring contrasts**, not pure error-specific generators.

## Repository structure

```text
code/
  00_model_development/          Historical, supportive, superseded, and stopped development source
  01_feature_generation/         ERP, time-frequency, and derived-feature generation
  02_primary_cue/                Cue action prediction and RL-state encoding
  03_primary_response/           Response monitoring and participant-level RL–EEG association analysis
  04_robustness/                 Completed stability, confound, reliability, expanded association family, and trial-level tests
  05_reporting/                  Publication-facing figure generation

shared/
  rl/ action/ ddm/ eeg/          Reusable analysis support modules

results/
  model_development/              Historical/supportive/superseded aggregate evidence
  cue_action/                     Frozen cue action-prediction benchmark
  cue_encoding/                   Frozen cue RL-state encoding benchmark
  response_monitoring/            Frozen response-locked performance-monitoring contrasts
  between_participant_rl_eeg_associations/   Frozen corrected 18-test between-participant association analysis
  robustness/                     Completed post-development robustness results
  reporting/                      Publication-facing source tables and frozen figures

configs/                          Machine-readable analysis specifications
controlled_inputs/               Schema for non-redistributed authorized inputs
provenance/                       Historical source hashes and release manifest
docs/                             Lineage, nomenclature, temporal provenance, privacy, QA, and rerun guidance
```

## Reproduction levels

### 1. Public reporting reproduction

No participant-level data are required.

```bash
python -m pip install -r requirements_reporting.txt
make figures
```

Generated reporting assets are compared with the frozen references in `results/reporting/figures/`.

### 2. Repository validation

```bash
make validate
```

Validation checks release metadata, required paths, Python syntax, structured-file integrity, formal public naming, nonportable-path exclusion, participant-identifier leakage, and release-manifest integrity.

### 3. Full controlled rerun

A full rerun requires authorized processed EEG resources, participant-level behavioral/RL data, cue-feature tables, and cross-fitted train-fold RL posterior objects. See:

- `docs/controlled_input_schema.md`
- `docs/controlled_rerun_commands.md`
- `make controlled-preflight`

Controlled reruns write under `reproduced/controlled_rerun/` and do not overwrite frozen public results.

## Signal-level reproducibility boundary

Raw acquisition EEG is not part of the available project holdings and is not redistributed. The recoverable feature-generation boundary begins from controlled processed continuous EEGLAB `.set/.fdt` resources. The analyzed single-trial branch contains no ICA stage. See `docs/eeg_preprocessing_provenance.md` and `docs/temporal_provenance.md`.

## Technical model identifiers

Technical identifiers such as `M6`, `E0–E9`, and `N0–N4` are retained only where computational traceability requires them. Manuscript-facing interpretation uses conceptual model names defined in `docs/model_nomenclature.md`.

## Privacy and data availability

The public release excludes participant identifiers, source EEG filenames, participant-level neural/behavioral rows, participant-level association tables, matched-pair records, participant-specific fold assignments, and participant-specific RL posterior objects. Aggregate results required to inspect the reported evidence are included. See `docs/data_availability.md` and `docs/privacy_and_exclusion_policy.md`.

## License

Repository-authored software and documentation are released under the **MIT License**; see `LICENSE`. The license does not grant access to controlled participant-level data or override third-party data/software terms.

## Citation

See `CITATION.md` and `CITATION.cff` for the verified author and citation metadata.
