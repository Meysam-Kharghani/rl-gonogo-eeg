# Change log

## Release 1.0.1 — 2026-10-25

- added verified manuscript-author metadata and `CITATION.cff` for GitHub citation support;

- refreshed the seed-sensitive aggregate outputs for response-confound triangulation and trial-level cross-fitted RL–EEG inference using the deterministic SHA-256 sub-seed contract already present in the public source;
- synchronized the three direct response-matching caliper summary tables with the refreshed combined caliper table;
- confirmed that non-seed-sensitive estimates, counts, model fits, and grouped-prediction quantities reproduce within `1e-10`;
- confirmed that the multiplicity-based scientific decisions are unchanged by the deterministic refresh;
- closed the previously documented narrow deterministic-rerun caveat while preserving the controlled-data privacy boundary.

## Release 1.0.0 — 2026-10-24

- rebuilt the public package around the complete staged EEG/computational analysis lineage;
- integrated repeated-partition cue stability, richer-history consistency, response-confound triangulation, response reliability, expanded association family, and trial-level cross-fitted RL–EEG robustness;
- formally classified historical/superseded decoding and association analyses instead of mixing them with the final evidence hierarchy;
- removed participant identifiers and participant-coded EEG filenames from the public results boundary;
- replaced participant-level matching-balance files with aggregate balance summaries;
- replaced environment-specific development naming with publication-facing terminology;
- patched publication-facing permutation helpers to use deterministic SHA256-derived seeds;
- added release validation for development naming, absolute paths, participant-identifier leakage, syntax, structured files, and checksums;
- changed figure reproduction to a non-destructive `reproduced/figures/` target.
