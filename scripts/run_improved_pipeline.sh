#!/usr/bin/env bash
# Explicit, restart-safe phases for the HU/HNM/MIL fracture pipeline.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src
export YOLO_OFFLINE=1
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/ultralytics-fracture}"
PYTHON="${PYTHON:-python}"
ORIGINAL_CONFIG="${ORIGINAL_CONFIG:-configs/fracture_25d_p2_hu800_original.yaml}"
BASE_CONFIG="${BASE_CONFIG:-configs/fracture_25d_p2_hu800.yaml}"
HNM_CONFIG="${HNM_CONFIG:-configs/fracture_25d_p2_hu800_hnm.yaml}"
ORIGINAL_DATASET="${ORIGINAL_DATASET:-/dev/shm/iaaa_fracture/fracture_dataset_v2_hu800_ww1600_original}"
BASE_DATASET="${BASE_DATASET:-/dev/shm/iaaa_fracture/fracture_dataset_v3_hu800_ww1600_corrected}"
HNM_DATASET="${HNM_DATASET:-/dev/shm/iaaa_fracture/fracture_dataset_v4_hu800_ww1600_corrected_hnm}"
ORIGINAL_RUN="${ORIGINAL_RUN:-v5_hu800_ww1600_original}"
BASE_RUN="${BASE_RUN:-v6_hu800_ww1600_corrected}"
HNM_RUN="${HNM_RUN:-v7_hu800_ww1600_corrected_hnm}"
MIL_RUN="${MIL_RUN:-v1_study_mil_hu800}"
FINAL_DETECTOR_RUN="${FINAL_DETECTOR_RUN:-final_hu800_hnm}"
FINAL_MIL_RUN="${FINAL_MIL_RUN:-final_study_mil_hu800}"

require_checkpoint() {
  local checkpoint="$1"
  if [[ ! -s "$checkpoint" ]]; then
    echo "Missing or empty checkpoint: $checkpoint" >&2
    return 1
  fi
  local bytes
  bytes="$(stat -c %s "$checkpoint")"
  if (( bytes < 100000 )); then
    echo "Implausibly small checkpoint (${bytes} bytes): $checkpoint" >&2
    return 1
  fi
}

train_detector_folds() {
  local config="$1" dataset="$2" run="$3" fold_list="${4:-0 1 2 3 4}"
  for fold in $fold_list; do
    local checkpoint="outputs/fold_${fold}/${run}/weights/best.pt"
    if [[ -e "$checkpoint" ]]; then
      require_checkpoint "$checkpoint"
    else
      "$PYTHON" -m fracture.training.train_detector \
        --config "$config" --fold "$fold" \
        --dataset-yaml "${dataset}/fold_${fold}.yaml" \
        --run-name "$run"
    fi
    local report="reports/fold${fold}_${run}.csv"
    local slices="reports/fold${fold}_${run}_slices.csv"
    local metrics="reports/fold${fold}_${run}_metrics.json"
    if [[ -s "$report" && -s "$slices" && -s "$metrics" ]]; then
      echo "Skipping held-out fold ${fold}; immutable reports already exist"
    elif [[ -e "$report" || -e "$slices" || -e "$metrics" ]]; then
      echo "Partial/empty report set exists for fold ${fold}; choose a new run name" >&2
      return 1
    else
      "$PYTHON" -m fracture.evaluation.predict_fold \
        --config "$config" --fold "$fold" --weights "$checkpoint" \
        --output-dir reports --name "fold${fold}_${run}"
    fi
  done
}

fit_detector_oof() {
  local config="$1" run="$2" destination="$3"
  local prediction_args=()
  for fold in 0 1 2 3 4; do
    prediction_args+=(--fold-csv "reports/fold${fold}_${run}.csv")
  done
  "$PYTHON" -m fracture.evaluation.generate_oof \
    "${prediction_args[@]}" --config "$config" \
    --output-dir "$destination" --models-dir "${destination}/models" \
    --calibration platt
}

case "${1:-help}" in
  audit)
    "$PYTHON" -m fracture.data.audit --config "$ORIGINAL_CONFIG"
    ;;
  analyse-hu)
    "$PYTHON" -m fracture.data.window_analysis \
      --config configs/fracture_25d_p2_pretrained.yaml --fold 0 \
      --output reports/hu_window_analysis_fold0_train.json
    ;;
  prepare-original)
    "$PYTHON" -m fracture.data.prepare_yolo --config "$ORIGINAL_CONFIG"
    ;;
  train-original-fold0)
    test -f "${ORIGINAL_DATASET}/fold_0.yaml"
    train_detector_folds "$ORIGINAL_CONFIG" "$ORIGINAL_DATASET" "$ORIGINAL_RUN" "0"
    ;;
  train-original-oof)
    test -f "${ORIGINAL_DATASET}/fold_0.yaml"
    train_detector_folds "$ORIGINAL_CONFIG" "$ORIGINAL_DATASET" "$ORIGINAL_RUN"
    fit_detector_oof "$ORIGINAL_CONFIG" "$ORIGINAL_RUN" reports/oof_original
    ;;
  review-original-oof)
    review_args=()
    for fold in 0 1 2 3 4; do
      review_args+=(--input "reports/fold${fold}_${ORIGINAL_RUN}_slices.csv")
    done
    "$PYTHON" -m fracture.data.reannotation "${review_args[@]}" \
      --config "$ORIGINAL_CONFIG" \
      --output reports/reannotation_candidates.csv \
      --log reports/reannotation_log.csv \
      --visualization-root visualizations/reannotation
    ;;
  audit-corrections)
    "$PYTHON" -m fracture.data.reannotation_audit \
      --config "$BASE_CONFIG" --log reports/reannotation_log.csv \
      --output reports/reannotation_summary.json
    ;;
  prepare-base)
    "$PYTHON" -m fracture.data.prepare_yolo --config "$BASE_CONFIG"
    ;;
  train-base-oof)
    test -f "${BASE_DATASET}/fold_0.yaml"
    train_detector_folds "$BASE_CONFIG" "$BASE_DATASET" "$BASE_RUN"
    fit_detector_oof "$BASE_CONFIG" "$BASE_RUN" reports/oof_corrected
    ;;
  mine-hard-negatives)
    prediction_args=()
    for fold in 0 1 2 3 4; do
      prediction_args+=(--predictions "reports/fold${fold}_${BASE_RUN}_slices.csv")
    done
    "$PYTHON" -m fracture.training.hard_negative_mining \
      "${prediction_args[@]}" --confidence 0.10 --per-study-limit 10 \
      --output reports/hard_negatives_oof.csv
    ;;
  prepare-hnm)
    test -s reports/hard_negatives_oof.csv
    "$PYTHON" -m fracture.data.prepare_yolo --config "$HNM_CONFIG"
    ;;
  train-hnm-oof)
    test -f "${HNM_DATASET}/fold_0.yaml"
    train_detector_folds "$HNM_CONFIG" "$HNM_DATASET" "$HNM_RUN"
    ;;
  train-mil-oof)
    test -f "${BASE_DATASET}/manifest.csv"
    for fold in 0 1 2 3 4; do
      checkpoint="outputs/fold_${fold}/${MIL_RUN}/weights/best.pt"
      if [[ -e "$checkpoint" ]]; then
        require_checkpoint "$checkpoint"
      else
        "$PYTHON" -m fracture.training.train_study_mil \
          --config "$BASE_CONFIG" --dataset-root "$BASE_DATASET" \
          --fold "$fold" --run-name "$MIL_RUN"
      fi
    done
    ;;
  fuse-oof)
    detector_args=()
    study_args=()
    for fold in 0 1 2 3 4; do
      detector_args+=(--fold-csv "reports/fold${fold}_${HNM_RUN}.csv")
      study_args+=(--study-csv "outputs/fold_${fold}/${MIL_RUN}/heldout_predictions.csv")
    done
    "$PYTHON" -m fracture.evaluation.generate_oof \
      "${detector_args[@]}" "${study_args[@]}" \
      --config "$HNM_CONFIG" --output-dir reports/final_oof \
      --models-dir models/final_hybrid --calibration platt
    ;;
  train-final-detector)
    : "${FINAL_DETECTOR_EPOCHS:?Set from the five OOF best-epoch histories; do not tune on all-data metrics}"
    test -f "${HNM_DATASET}/all.yaml"
    test ! -e "outputs/fold_-1/${FINAL_DETECTOR_RUN}"
    "$PYTHON" -m fracture.training.train_detector \
      --config "$HNM_CONFIG" --fold -1 --dataset-yaml "${HNM_DATASET}/all.yaml" \
      --epochs "$FINAL_DETECTOR_EPOCHS" --patience 0 --run-name "$FINAL_DETECTOR_RUN"
    ;;
  train-final-mil)
    : "${FINAL_MIL_EPOCHS:?Set from the five OOF best-epoch histories; do not tune on all data}"
    test -f "${BASE_DATASET}/manifest.csv"
    test ! -e "outputs/final/${FINAL_MIL_RUN}"
    "$PYTHON" -m fracture.training.train_study_mil \
      --config "$BASE_CONFIG" --dataset-root "$BASE_DATASET" \
      --all-data --epochs "$FINAL_MIL_EPOCHS" --run-name "$FINAL_MIL_RUN"
    ;;
  make-deployment)
    : "${FINAL_DETECTOR:?Set FINAL_DETECTOR to the selected non-empty best.pt}"
    manifest_args=(--detector "$FINAL_DETECTOR" --aggregator models/final_hybrid/aggregator.joblib)
    if [[ -n "${FINAL_CALIBRATOR:-}" ]]; then manifest_args+=(--calibrator "$FINAL_CALIBRATOR"); fi
    if [[ -n "${FINAL_STUDY_MODEL:-}" ]]; then manifest_args+=(--study-model "$FINAL_STUDY_MODEL"); fi
    "$PYTHON" scripts/create_deployment_manifest.py "${manifest_args[@]}" \
      --output models/deployment.yaml --window-level 800 --window-width 1600 \
      --context-distance-mm 5 --input-mode 2.5d --image-size 768
    ;;
  help|*)
    echo "Usage: $0 {audit|analyse-hu|prepare-original|train-original-fold0|train-original-oof|review-original-oof|audit-corrections|prepare-base|train-base-oof|mine-hard-negatives|prepare-hnm|train-hnm-oof|train-mil-oof|fuse-oof|train-final-detector|train-final-mil|make-deployment}"
    ;;
esac
