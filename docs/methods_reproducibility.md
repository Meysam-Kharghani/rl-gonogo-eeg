# Methods reproducibility notes

## Cross-fitted RL targets

EEG-computational study uses the same strict outer participant folds as the computational analysis. For every outer fold, M6 is estimated only from training participants. New-subject latent trajectories for both downstream training and held-out participants are generated from the train-only posterior.


## M6 train-fold sampling

For the primary seed-2026 folds, M6 used four NUTS chains, 600 tuning iterations followed by 600 retained draws per chain, four processing cores, and `target_accept=0.90`. Fold-specific diagnostics are in `results/rl_provenance/actor_control_fold_posterior_diagnostics.csv`; across the five folds, maximum R-hat was 1.01 and no divergences were observed.

## Cue-action benchmark

Models E0–E9 use nested tuning inside each outer-training fold. E0 contains task design and strictly lagged behavioral history. E2 adds actor Q-difference and control. The primary neural contrast is E8−E2, adding the fixed 37-feature nominal cue EEG block to the selected computational action baseline.

Because E0/E2 include current target class, this is conditional outcome prediction, not a decoder restricted to information available before target onset.

## Cue RL-state encoding

Targets are cross-fitted policy logit, actor Q-difference, and control. N0 contains task design + strictly lagged behavioral history. N1–N3 add ERP, TF, or entropy-derived cue features, and N4 adds the combined cue block. Positive standardized MSE gain indicates that EEG lowered held-out target error.

## Response-monitoring signatures

The fixed participant-level family contains six FA-minus-Go-hit signatures. Primary cell eligibility is at least 10 trials per required outcome cell; 38 participants meet that threshold. Thresholds of 15 and 20 trials are sensitivity checks.

## Corrected RL-response inference

For each of six signatures and three RL summaries, the primary OLS model includes risk group as a nuisance covariate. Freedman-Lane residual permutation is used instead of direct predictor shuffling. Residuals are permuted within group strata; the reconstructed outcome is then refit with the full model.

Primary multiplicity correction is BH-FDR across all 18 fixed tests.

## Response-marker QC

The aggregate timing check compares reconstructed response onset (target onset + behavioral RT) with the nearest response marker in the imported event list. All 10,509 responded trials had a matched marker; the aggregate median absolute discrepancy was 0 ms. This is an internal alignment-consistency check, not independent validation of RT accuracy.
