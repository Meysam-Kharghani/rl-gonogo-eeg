# Reproducibility workflows

## Public aggregate reproduction

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_reporting.txt
make validate
make figures
```

This validates the public package and regenerates publication figures from distributed aggregate tables. Generated assets are written under `reproduced/` and do not overwrite frozen references.

## Controlled downstream rerun

Authorized users can materialize the controlled behavioral, EEG-feature, and cross-fitted RL inputs described in `docs/controlled_input_schema.md`, then rerun the canonical cue, response, association analysis, and robustness scripts under `code/`.

## Feature regeneration

The repository provides the portable feature-generation source, but full signal-to-report regeneration requires the processed continuous EEG/event resources described in `docs/eeg_preprocessing_provenance.md`. Raw acquisition files are not redistributed, and the available project branch does not contain an ICA stage for the analyzed trial-wise data.
