# Standalone fracture-only rules — model accuracy

Rules use **only** `fracture_prob` (no ICH, no MLS, no official triage call).

```
if fracture_prob >= 0.5: return 1  # Urgent
else:                    return 0  # Non-urgent
```

- Studies: **169** (pos=24, neg=145)

## Submit deployment (apparent)

- Accuracy: **0.9053**
- Macro-F1 (binary / standalone triage): **0.7239**
- Sensitivity / Specificity: **0.3333** / **1.0000**
- Precision / Pos-F1: **1.0000** / **0.5000**
- PR-AUC / ROC-AUC: **0.5646** / **0.7289**
- TP/FP/FN/TN: `[8, 0, 16, 145]`

## Unbiased cross-fit

- Accuracy: **0.8935**
- Macro-F1: **0.7332**
- Sensitivity / Specificity: **0.4167** / **0.9724**
- TP/FP/FN/TN: `[10, 4, 14, 141]`
