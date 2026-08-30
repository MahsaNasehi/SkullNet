#!/usr/bin/env bash
# Train every patient fold with the configured detector, then evaluate held-out studies.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/fracture_25d_p2.yaml}"
RUN_NAME="${2:-v1_25d_p2}"
DATASET_ROOT="${DATASET_ROOT:-/dev/shm/iaaa_fracture/fracture_dataset_v1_25d_p2}"
PYTHON="${PYTHON:-/home/mahsa-nasehi/miniconda3/envs/maskfo/bin/python}"
export PYTHONPATH=src
export YOLO_OFFLINE=1
export YOLO_CONFIG_DIR=/tmp/ultralytics-maskfo

if [[ ! -f "${DATASET_ROOT}/fold_0.yaml" ]]; then
  echo "Dataset missing at ${DATASET_ROOT}. Generate with:"
  echo "  PYTHONPATH=src python -m fracture.data.prepare_yolo --config ${CONFIG}"
  exit 1
fi

FOLD_CSVS=()
for fold in 0 1 2 3 4; do
  weights="outputs/fold_${fold}/${RUN_NAME}/weights/best.pt"
  if [[ ! -f "${weights}" ]]; then
    echo "=== Training fold ${fold} ==="
    "${PYTHON}" -m fracture.training.train_detector \
      --config "${CONFIG}" \
      --fold "${fold}" \
      --dataset-yaml "${DATASET_ROOT}/fold_${fold}.yaml" \
      --run-name "${RUN_NAME}"
  else
    echo "=== Skipping train fold ${fold}; found ${weights} ==="
  fi
  echo "=== Predicting fold ${fold} ==="
  name="fold${fold}_${RUN_NAME}"
  "${PYTHON}" -m fracture.evaluation.predict_fold \
    --config "${CONFIG}" \
    --fold "${fold}" \
    --weights "outputs/fold_${fold}/${RUN_NAME}/weights/best.pt" \
    --output-dir reports \
    --name "${name}"
  FOLD_CSVS+=(--fold-csv "reports/${name}.csv")
done

echo "=== Fitting OOF aggregator + calibrator ==="
"${PYTHON}" -m fracture.evaluation.generate_oof \
  "${FOLD_CSVS[@]}" \
  --config "${CONFIG}" \
  --output-dir reports \
  --models-dir models \
  --calibration platt

echo "Done. See reports/final_metrics.json and models/*.joblib"
