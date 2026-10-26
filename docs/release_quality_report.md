# Release quality report

**Assessment date:** 2026-10-25  
**Release:** 1.0.1  
**Public status:** publication-facing reproducibility release; deterministic robustness refresh completed

## Scientific/project-history completeness

- The repository preserves the stage-gated EEG/RL lineage from processing/alignment and feature development through canonical cue analyses, response-monitoring analyses, the corrected participant-level association analysis, and all completed robustness analyses.
- Superseded exploratory branches remain documented as development history rather than being mixed with the final inferential hierarchy.
- Available historical source is included under `code/00_model_development/` using formal filenames and explicit status documentation.
- Code-only extensions that did not pass into the final evidence hierarchy are marked as stopped by design; they are not missing final analyses.

## Public-boundary controls

- Publication-facing paths and text use conceptual scientific names.
- Participant-level EEG/behavior/RL data, participant identifiers, source EEG filenames, participant-level association tables, matching records, and fold-specific posterior objects are excluded.
- Historical source provenance is retained by conceptual role plus SHA-256 rather than by environment-specific source paths.
- Controlled reruns write under `reproduced/controlled_rerun/` and do not overwrite frozen public results.

## Reproducibility boundary

The public repository does not claim raw-acquisition-to-report reproducibility because raw acquisition EEG is unavailable/not redistributed. Feature regeneration begins from controlled processed continuous EEG/event resources documented in `docs/eeg_preprocessing_provenance.md`. Non-causal filtering limits cue-window temporal claims even though computational targets are cross-fitted across participants.

## Deterministic permutation-seed refresh

The response-confound triangulation and trial-level RL–EEG robustness branches were refreshed using the deterministic SHA-256-derived sub-seed implementation already present in the public source. Release 1.0.1 replaces the affected public aggregate tables with the refreshed values and synchronizes the direct per-caliper response-matching summaries with the combined caliper table. All non-seed-sensitive quantities reproduced within `1e-10`, and the multiplicity-based scientific decisions were unchanged. The controlled participant-level matching and trial-analysis tables are not redistributed.

## Author and archival metadata

Verified manuscript-author metadata are included in `CITATION.md` and `CITATION.cff`. Repository URL/DOI and remaining publication-specific administrative metadata should be added only after verified records exist.
