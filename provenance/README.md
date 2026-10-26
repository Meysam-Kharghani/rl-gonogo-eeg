# Provenance policy

Frozen aggregate results derive from historical executed source in the controlled project archive. The public repository uses **formal publication-facing source names** and omits development naming conventions and environment-specific paths.

`provenance/historical_source_hashes.tsv` records cryptographic hashes and scientific roles for historical source snapshots without requiring environment-specific source paths in the public package. `provenance/release_manifest.tsv` records SHA256 hashes for every distributed file.

Publication-facing code may contain nomenclature/portability edits relative to the historical executed snapshot. These edits must not be confused with a new scientific analysis; result tables remain tied to the historical execution record and completed robustness outputs.
