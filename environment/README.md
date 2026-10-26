# Computational environment

`environment.yml` is the preferred portable full-analysis environment for this repository. The environment name is `rl-eeg-prediction-performance-monitoring-analysis`.

`conda-linux-64-explicit.txt` preserves the exact Linux conda package snapshot captured from the validated analysis environment. It is platform-specific and is retained for provenance.

For public aggregate validation and figure regeneration only, `requirements_reporting.txt` provides the minimal pinned dependencies. `requirements_analysis.txt` lists the principal pinned Python dependencies for controlled-input analysis reruns.

Recommended full environment setup:

```bash
conda env create -f environment/environment.yml
conda activate rl-eeg-prediction-performance-monitoring-analysis
```
