# Temporal provenance and cue-window interpretation

The primary cue features are nominally cue-window and their windows end at or before nominal target onset. Strict temporal support is nevertheless limited by the preprocessing chain.

The recovered EEGLAB filtering uses a non-causal/zero-phase FIR implementation, and the time-frequency branch uses forward-backward `sosfiltfilt` before Hilbert transformation. Consequently, samples after a nominal cue time point can influence the processed value at that time.

Subject-level cross-validation prevents model-fitting leakage across participants but cannot reverse temporal mixing already introduced by signal processing.

Accordingly, the repository uses **nominal cue-window** or **nominally pre-target** terminology and does not treat those features as proof of strictly prospective temporal information.

A strictly prospective neural analysis would require raw or minimally processed continuous EEG, a causal filter or explicitly quantified guard interval, and feature windows whose effective support lies entirely before the prediction boundary.
