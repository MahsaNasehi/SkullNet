# Fracture workstream report

Status: core study-level path is implemented (OOF aggregator selection, calibration,
isolated fracture QWK, hard-negative/adjacent sampling hooks). Full 5-fold detector
OOF remains the main empirical blocker for excellent competition metrics.

## Dataset

- 338 studies / 320 patients; 28 fracture-positive, 310 negative
- 7,683 DICOMs; 5,176 JSONs; 260 positive slices; 356 boxes
- Missing JSON stays unknown (`missing_json_means_negative: false`)
- Patient-grouped 5-fold split frozen in `splits/folds.json`

## Current empirical snapshot (fold-0 V0 single, max baseline)

- Study AUROC ≈ 0.749, PR-AUC ≈ 0.323
- At official threshold 0.5: sensitivity 0.0 / specificity 1.0 (scores under-confident)
- Isolated fracture QWK ≈ 0.944 when all study scores stay < 0.5 (triage mostly driven by ICH/MLS)

## What was improved in code

1. Nested/stratified OOF fitting for logistic aggregation + auto-select among `max` / `consecutive` / `logistic`
2. Isolated fracture QWK using GT ICH volumes (area→mL) + MLS + predicted `fracture_prob`
3. `consecutive` aggregator down-weights single-slice spikes (suture-like FPs)
4. Hard-negative CSV + fracture-adjacent oversampling in `prepare_yolo`
5. Explicit YOLO `box/cls/dfl` gains in V2 config
6. `scripts/train_eval_all_folds.sh` and `scripts/acceptance_smoke.py`
7. Expanded unit tests (13 passing)

## Required next empirical steps for excellent results

1. Regenerate `/dev/shm` YOLO dataset after reboot
2. Train folds 1–4 (fold 0 V2 may already exist): `scripts/train_eval_all_folds.sh`
3. Re-fit OOF on all five fold CSVs; keep selected aggregator + calibrator in `models/`
4. Mine hard negatives from OOF FPs on verified negatives; rebuild dataset version; retrain
5. Human re-annotation from OOF disagreements only; compare original vs corrected on locked folds
6. Run acceptance smoke offline with local weights/joblib only
7. Prefer model selection by isolated fracture QWK and sensitivity@0.5, not detector mAP
