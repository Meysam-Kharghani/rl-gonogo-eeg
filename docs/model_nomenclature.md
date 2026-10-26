# Model nomenclature

The computational backbone uses the **actor–control reinforcement-learning model**. Technical ID `M6` is retained in code and machine-readable tables for traceability but should normally be replaced by the conceptual label in manuscript prose.

Cue-model IDs are also technical implementation labels:

| Technical ID | Conceptual manuscript label |
|---|---|
| E0 | task-design + primary lagged-history action reference |
| E1 | history + policy action model |
| E2 | history + actor/control action model |
| E6 | history + combined cue-EEG model |
| E8 | history + actor/control + combined cue-EEG model |
| E9 | history + full RL + combined cue-EEG model |
| N0 | task-design + lagged-history RL-state encoding reference |
| N4 | reference + combined cue-EEG RL-state encoding model |

Use the conceptual labels in the article. Technical IDs may appear in supplementary computational tables where needed to provide an unambiguous crosswalk to code.
