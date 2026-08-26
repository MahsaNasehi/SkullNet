# IAAA 2026 skull-fracture workstream

This package implements the offline core path `study_dir -> DICOM/HU/bone window -> YOLO -> study aggregation -> calibration -> float fracture_prob`. It never downloads weights or installs dependencies at prediction time. Corrected annotations override originals without modifying organizer files; missing JSON is unknown by default.

## Dataset status

The supplied dataset is mounted in `iaaa-contest-bct/Data`. It contains 338 studies from 320 patients, 7,683 DICOM files, and 5,176 organizer annotation JSONs. Metadata is stored in `training_df.pkl`; its configured fracture columns agree with all existing `boxes_xywh` labels. See `reports/data_summary.json` for the complete audit.

A full pixel-decode audit passed for all DICOM files, including 3,148 JPEG Lossless instances. Patient-grouped folds are frozen in `splits/folds.json`. The rendered YOLO dataset is deliberately stored in `/dev/shm/iaaa_fracture` because the main filesystem has limited free space. `/dev/shm` is volatile, so regenerate the rendered dataset after a reboot; the source DICOMs remain authoritative.

## `maskfo` environment and required packages

Training has been verified with Python 3.10.12 and the following versions:

| Package | Version |
| --- | --- |
| PyTorch | `2.1.2+cu121` |
| torchvision | `0.16.2+cu121` |
| Ultralytics | `8.4.129` |
| NumPy | `1.26.4` |
| pandas | `2.2.3` |
| pydicom | `3.0.1` |
| pylibjpeg | `2.1.0` |
| pylibjpeg-libjpeg | `2.2.0` |
| pylibjpeg-openjpeg | `2.3.0` |
| SimpleITK | `2.5.6` |
| scikit-learn | `1.6.1` |
| joblib | `1.5.3` |
| albumentations | `1.4.24` |
| OpenCV | `4.9.0.80` |
| PyYAML | `6.0.3` |
| pytest | `8.4.2` |

Activate the existing CUDA-enabled environment. Do not replace its accepted PyTorch/CUDA installation:

```bash
conda activate maskfo

python -m pip install \
  numpy==1.26.4 pandas==2.2.3 pydicom==3.0.1 \
  pylibjpeg==2.1.0 pylibjpeg-libjpeg==2.2.0 pylibjpeg-openjpeg==2.3.0 \
  SimpleITK==2.5.6 ultralytics==8.4.129 scikit-learn==1.6.1 \
  joblib==1.5.3 albumentations==1.4.24 opencv-python==4.9.0.80 \
  PyYAML==6.0.3 pytest==8.4.2
```

Install `tmux` once at the operating-system level:

```bash
sudo apt-get install -y tmux
```

Verify the environment from the repository root:

```bash
cd "/home/mahsa-nasehi/Desktop/New Folder/fracture"
PYTHONPATH=src python -m fracture.utils.check_environment
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.version.cuda)"
PYTHONPATH=src python -m pytest -q
```

## Dataset preparation

The real paths and schema columns are set in `configs/fracture.yaml`. Generate the lossless PNG dataset and fold YAML files with:

```bash
PYTHONPATH=src python -m fracture.data.audit --config configs/fracture.yaml
PYTHONPATH=src python -m fracture.data.prepare_yolo --config configs/fracture.yaml
```

Dataset versions are immutable: generation fails if a version directory already exists. Establish `splits/folds.json` before OOF-assisted review. Only OOF disagreements may enter the review queue, and reviewer decisions must be recorded in `reports/reannotation_log.csv`; model output never changes labels automatically.

## Fold-0 training in `tmux`

The current run is resumed from `outputs/fold_0/detector/weights/last.pt` inside a detached session named `fracture`. This is the exact launch command:

```bash
tmux new-session -d -s fracture \
  -c "/home/mahsa-nasehi/Desktop/New Folder/fracture" \
  "exec env YOLO_CONFIG_DIR=/tmp/ultralytics-maskfo PYTHONPATH=src \
  /home/mahsa-nasehi/miniconda3/envs/maskfo/bin/python \
  -m fracture.training.train_detector \
  --config configs/fracture.yaml \
  --fold 0 \
  --dataset-yaml /dev/shm/iaaa_fracture/fracture_dataset_v0_original/fold_0.yaml \
  --resume-from outputs/fold_0/detector/weights/last.pt \
  --run-name detector"
```

Attach to the running session:

```bash
tmux attach -t fracture
```

Detach without stopping training by pressing `Ctrl+B`, releasing the keys, and then pressing `D`. Check the session and GPU from another terminal with:

```bash
tmux ls
watch -n 2 nvidia-smi
tail -f outputs/fold_0/detector/results.csv
```

For a fresh fold-0 run rather than a resume, omit `--resume-from` and add `--epochs 100 --batch-size 1 --workers 2`. Use a new `--run-name` if the requested output directory already exists.

## Latest fold-0 results

The random-initialization fold-0 baseline stopped normally at epoch 27 because validation fitness did not improve for the configured patience of 20 epochs. Its best `mAP50-95` occurred at epoch 7:

| Metric | Best-epoch value |
| --- | ---: |
| Precision | `0.15675` |
| Recall | `0.12500` |
| mAP50 | `0.07426` |
| mAP50-95 | `0.03794` |

Epoch 26 produced the highest `mAP50` (`0.10329`) but a lower `mAP50-95` (`0.03352`). The final epoch-27 metrics were precision `0.19500`, recall `0.11250`, mAP50 `0.05324`, and mAP50-95 `0.02210`. The fold-0 validation set contains only six positive studies and 80 fracture boxes, so these single-fold metrics have high variance.

An end-to-end held-out inference smoke test produced:

| Study | Ground truth | Raw study score | Total inference time |
| --- | --- | ---: | ---: |
| `2265` | Positive (6 positive slices) | `0.17444` | `2.714 s` |
| `1734` | Negative | `0.12292` | `0.286 s` |

The current study score is the maximum slice-detection confidence with no fitted aggregator or calibration model. It is therefore an uncalibrated ranking score, not a clinically meaningful probability. The small positive/negative separation and low detection metrics make this a working pipeline baseline, not a deployment-ready or competition-ready model. Detailed training metrics are in `outputs/fold_0/detector/results.csv`; the inference smoke report is in `reports/inference_smoke_fold0.json`.

## Improvement checklist

- [ ] Obtain explicit approval for local pretrained weights and fine-tune them instead of training YOLO11s from random initialization.
- [ ] Train all five patient-grouped folds and report mean and per-fold detection and study-level metrics.
- [ ] Generate strictly out-of-fold slice and study predictions for model selection, aggregation, calibration, and error analysis.
- [ ] Ensemble the five fold models at inference time and compare it with each individual fold model.
- [ ] Compare the current single-slice input with 2.5D input using the preceding, current, and following CT slices as the three channels.
- [ ] Improve small-fracture sensitivity by testing a P2/high-resolution detection head, 768–1024 pixel inputs where VRAM allows, and skull-focused crops or overlapping patches.
- [ ] Oversample positive and fracture-adjacent slices so batch-size-1 training receives more informative positive updates.
- [ ] Retain representative negatives and mine hard negatives such as sutures, vessels, motion, and reconstruction artifacts from out-of-fold predictions.
- [ ] Review false positives and false negatives manually; record accepted corrections in `reports/reannotation_log.csv` without automatically changing organizer labels.
- [ ] Fit a study-level aggregator on out-of-fold features such as top-k confidence, detected-slice count, total detections, and consecutive positive slices.
- [ ] Fit Platt or isotonic calibration on out-of-fold study predictions and evaluate calibration separately from ranking performance.
- [ ] Tune detection confidence and NMS thresholds on out-of-fold data rather than interpreting the detector's raw maximum score as a probability.
- [ ] Evaluate study-level sensitivity, specificity, ROC-AUC/PR-AUC, QWK, calibration, and subgroup errors in addition to slice-level mAP.
- [ ] Benchmark inference across positive, negative, large, and compressed-DICOM studies under the final evaluator-style environment.
- [ ] Package only the selected local weights, aggregator, calibrator, configuration, and inference code; rerun the offline evaluator acceptance test.

## Inference API

```python
from fracture import FracturePredictor

predictor = FracturePredictor(
    "models/best.pt",
    aggregator_path="models/aggregator.joblib",  # omit for max
    calibrator_path="models/calibrator.joblib",  # omit for none
    aggregation_method="logistic",
    calibration_method="platt",
)
fracture_prob = predictor.predict(study_dir)
```

The API receives no annotations or training metadata. `model.py` and `submission.py` are thin adapters around it.

## Required empirical completion

Once data is available: inspect metadata/schema; run the audit; freeze patient/study folds; visualize GT; generate versioned datasets; train all folds; generate OOF slice predictions; review candidates; train original/corrected comparisons on identical folds; fit the logistic aggregator and calibrator on OOF only; calculate detection/study/QWK metrics; benchmark positive, negative, large, and compressed-DICOM studies; then run evaluator-style inference with only submission artifacts. Record all results in the existing report templates.

The project must not be called competition-complete until those empirical steps pass.
