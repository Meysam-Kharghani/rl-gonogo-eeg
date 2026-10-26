# Public naming and terminology policy

This repository uses publication-facing, concept-based names for public paths, code modules, documentation, and result directories.

## Rules

- Environment-specific filesystem paths and retired development labels are prohibited from the public interface.
- Article-ordinal labels and internal chronological/version nicknames are not public scientific terminology.
- Public filenames describe the scientific function of an artifact.
- Technical model IDs such as `M6`, `E8`, or `N4` may be retained in code and machine-readable tables when required for an unambiguous computational crosswalk.
- Manuscript prose should use the conceptual labels in `docs/model_nomenclature.md`.
- Historical executed source is represented by conceptual source role plus SHA-256 digest rather than by an environment-specific source path.
- Controlled inputs are referenced through the formal `controlled_inputs/` schema.

These rules are part of release QA.
