# Fracture model provenance

Active checkpoint: Run A YOLO26s-P2 FULL169, fixed 59-epoch final refit, `models/best.pt`. Its SHA256 is `109bc30f72b27269ce13edf6937fc8c3f7e6462e0f8a87b1074770e83ea44640`. The source `outputs/yolo26s_p2_hu800_ww1600_run_a_full169/weights/final_deployment.pt` had the same hash. No classifier was packaged: `outputs/run_a_fracture_classifier_full169/final_classifier.pt` and its training-completion marker were absent.

The following are **169-Study, patient-held-out Run A family OOF results**, not measurements of the final FULL169 refit and not the full team submission metric:

| OOF system | TP | FP | FN | TN | Sensitivity | Study PR-AUC | ROC-AUC | Fracture F1 | Oracle-other-heads triage Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Detector only | 7 | 3 | 17 | 142 | 0.2916667 | 0.5355486 | 0.7298851 | 0.4117647 | 0.9696273781 |
| Detector + classifier fusion | 8 | 4 | 16 | 141 | 0.3333333 | 0.5592863 | 0.7672414 | 0.4444444 | 0.9696273781 |

The fusion improved fracture-specific OOF metrics but **did not improve oracle-other-heads triage Macro-F1**. It is a fracture-recall-oriented candidate, not proven improvement in the competition's primary metric. Oracle-other-heads evaluation uses ground-truth ICH volumes and MLS; this fracture-only artifact supplies zero placeholders for those outputs, so its final submission Macro-F1 is unknown.

OOF fusion used `top5_mean` classifier probability and `clip(0.75*p_detector + 0.25*p_classifier, 0, 1)` with official threshold `0.5`. Those are **historical experimental settings only**, not active code in this package. Its detector probability was cross-fitted with fold-specific transformations; the active detector-only deployment transformation is the previously recorded all-OOF apparent Run A setting (`top10_percent_mean`, raw operating point `0.39717610677083337`). The all-OOF deployment point is not an unbiased performance estimate.

The old Fold 2 single-checkpoint selection was not justified as globally best by comparisons on different validation cohorts. The selected Run A family was therefore refit on all 169 development Studies for a fixed 59 epochs, with no training-set validation metric selecting `best.pt`. The final epoch `last.pt` was validated and copied byte-identically to `final_deployment.pt`. No fixed-test data or predictions were used or included in this handoff.
