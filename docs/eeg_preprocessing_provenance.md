# EEG data and preprocessing provenance

## Available reproducibility boundary

The original raw EEG acquisition files are **not available** in the current project holdings.
Accordingly, this repository does not claim raw-acquisition-to-result reproducibility.

The earliest available participant EEG inputs are processed continuous EEGLAB `.set/.fdt` files.
Feature-generation code is provided for computational provenance and for controlled reruns that begin
from those processed inputs.

## ICA status

The EEG data used in the present analysis did **not** undergo independent-component analysis (ICA).

Nazari et al. (2024) report ICA-based eyeblink removal for the processing pipeline used in that
publication. That step is not part of the recoverable processing provenance for the present dataset.

## Recovered historical processing evidence

A recovered EEGLAB/ERPLAB source script documents operations including EDF import using BIOSIG,
channel-location assignment, 0.1–30 Hz FIR filtering, rereferencing, event-list/bin handling, epoching,
baseline correction, moving-window peak-to-peak artifact detection, and averaging. Because the raw EDF
files are unavailable, that historical script is retained as provenance evidence rather than a fully
replayable preprocessing workflow.

The public release therefore supports:
1. frozen aggregate result tables -> publication figures;
2. controlled trial-level feature/behavior tables -> statistical analyses;
3. processed continuous `.set/.fdt` -> available EEG feature extraction.

It does not support raw EEG -> processed EEG reconstruction.
