# Fracture-only official triage evaluation

Official `triage_from_intermediates` with **ICH volumes and MLS forced to 0**
for both GT and predictions. Only `fracture_prob` drives the class.

- Studies: **169** (pos=24, neg=145)
- Primary metric: **fracture_only_two_class_macro_f1** (classes {0,1})

## Submit deployment (apparent all-OOF)

- Fracture-only two-class Macro-F1: **0.7239**
- Sensitivity / Specificity: **0.3333** / **1.0000**
- Precision / PR-AUC: **1.0000** / **0.5646**
- TP/FP/FN/TN: `[8, 0, 16, 145]`
- Oracle-other-heads Macro-F1 (contrast): **0.9851**
- Masking gap (oracle − fracture-only): **0.2613**

## Unbiased cross-fit (fixed submit aggregator)

- Fracture-only two-class Macro-F1: **0.7332**
- Sensitivity / Specificity: **0.4167** / **0.9724**
- TP/FP/FN/TN: `[10, 4, 14, 141]`

## Unbiased cross-fit (aggregator + threshold)

- Fracture-only two-class Macro-F1: **0.7117**
- Sensitivity / Specificity: **0.3750** / **0.9724**
- TP/FP/FN/TN: `[9, 4, 15, 141]`
