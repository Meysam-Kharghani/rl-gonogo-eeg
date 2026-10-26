# Public-data boundary and exclusion policy

The public repository intentionally excludes participant identifiers, participant-level EEG/behavior rows, participant-level association tables, matched-pair tables, filenames that encode participant labels, and cross-validated fold manifests that enumerate participant IDs.

Aggregate statistics, model contrasts, reliability summaries, and multiplicity-corrected inference tables are retained. Where a useful diagnostic originally contained participant identifiers, a de-identified aggregate replacement is provided when possible.

`docs/excluded_public_artifacts.tsv` records the exclusions applied to the release.
