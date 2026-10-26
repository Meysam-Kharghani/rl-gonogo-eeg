# Contributing

This repository accompanies a specific scientific analysis. Corrections that improve reproducibility, documentation, or code reliability are welcome.

Before opening a pull request:

1. open an issue describing the proposed change;
2. keep scientific-result changes separate from documentation-only changes;
3. do not add raw EEG, participant-level features, or identifying data;
4. do not introduce environment-specific absolute paths;
5. run `make validate` and, when reporting assets are affected, `make figures`;
6. document any change that alters a reported numerical result in `docs/changelog.md`.

Changes to aggregate reported results require traceable analysis provenance.
