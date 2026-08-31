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

### Create the environment on a new CUDA server

The following creates the environment from scratch. Select the PyTorch wheel for the GPU architecture; the wheel bundles its CUDA runtime, so do not install the system CUDA toolkit merely for this project.

```bash
conda create -n maskfo python=3.10.12 pip -y
conda activate maskfo

python -m pip install --upgrade pip setuptools wheel

# RTX 30/40-series and older CUDA GPUs (for example the original RTX 3050):
python -m pip install \
  torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# RTX 50-series Blackwell GPUs (for example RTX 5060 Ti, capability sm_120):
# Use this command INSTEAD of the cu121 command above.
python -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  numpy==1.26.4 pandas==2.2.3 pydicom==3.0.1 \
  pylibjpeg==2.1.0 pylibjpeg-libjpeg==2.2.0 pylibjpeg-openjpeg==2.3.0 \
  SimpleITK==2.5.6 ultralytics==8.4.129 scikit-learn==1.6.1 \
  joblib==1.5.3 albumentations==1.4.24 opencv-python==4.9.0.80 \
  PyYAML==6.0.3 pytest==8.4.2
```

After cloning or copying the repository, install only the local package metadata; `--no-deps` prevents pip from replacing the pinned CUDA stack:

```bash
cd /absolute/path/to/fracture
python -m pip install --no-deps -e .
```

Install `tmux` once at the operating-system level:

```bash
sudo apt-get install -y tmux
```

Verify the driver, CUDA-enabled wheel, package imports, and tests from the repository root:

```bash
cd /absolute/path/to/fracture
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('Torch CUDA:', torch.version.cuda)"
PYTHONPATH=src python -m fracture.utils.check_environment
PYTHONPATH=src python -m pytest -q
```

Do not rely only on `torch.cuda.is_available()`. Verify that the wheel contains kernels for the installed GPU and execute a real CUDA operation:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print(torch.cuda.get_arch_list()); x=torch.randn(2048,2048,device='cuda'); y=x@x; torch.cuda.synchronize(); print('CUDA computation passed:', y.shape)"
```

For an RTX 5060 Ti, capability must be `(12, 0)`, the compiled architectures must include `sm_120`, and the matrix multiplication must pass without a compatibility warning. The driver-reported CUDA version in `nvidia-smi` can be newer than the wheel runtime; that is normal.

## Dataset preparation

The real paths and schema columns are set in `configs/fracture.yaml`. Generate the lossless PNG dataset and fold YAML files with:

```bash
PYTHONPATH=src python -m fracture.data.audit --config configs/fracture.yaml
PYTHONPATH=src python -m fracture.data.prepare_yolo --config configs/fracture.yaml
```

Dataset versions are immutable: generation fails if a version directory already exists. Establish `splits/folds.json` before OOF-assisted review. Only OOF disagreements may enter the review queue, and reviewer decisions must be recorded in `reports/reannotation_log.csv`; model output never changes labels automatically.

## Second-server handoff: controlled COCO-pretrained Fold 0

Do not start folds 1–4 yet. The first experiment is a controlled Fold-0 comparison in which only initialization changes. Both arms use the same frozen split, 2.5D images, P2–P5 head, 768-pixel resolution, sampling, losses, optimizer, seed, augmentations, max study aggregation, and no calibration.

### 1. Copy the required private/local data

Git does not contain the contest dataset. Copy the following to the same relative locations on the new server:

```text
iaaa-contest-bct/Data/training/
iaaa-contest-bct/Data/annotations/
iaaa-contest-bct/Data/training_df.pkl
```

For the final A/B evaluation, also copy the historical V2 baseline checkpoint and its training history:

```text
outputs/fold_0/v1_25d_p2/weights/best.pt
outputs/fold_0/v1_25d_p2/results.csv
```

The historical V0 checkpoint is needed only to reproduce/audit the local-transfer initialization arm:

```text
outputs/fold_0/detector/weights/best.pt
```

Check the paths from the repository root:

```bash
cd /absolute/path/to/fracture
test -d iaaa-contest-bct/Data/training
test -d iaaa-contest-bct/Data/annotations
test -f iaaa-contest-bct/Data/training_df.pkl
test -f splits/folds.json
df -h /dev/shm
```

The rendered dataset needs approximately 1 GB in `/dev/shm`; leave additional space for cache files.

### 2. Cache and verify official YOLO11s weights once

This is the only step that requires internet access. It downloads the official Ultralytics `yolo11s.pt` into the ignored development cache and writes SHA-256 provenance metadata.

```bash
cd /absolute/path/to/fracture
conda activate maskfo
unset YOLO_OFFLINE
PYTHONPATH=src python -m fracture.utils.pretrained \
  --config configs/fracture_25d_p2_pretrained.yaml
```

Verify the transfer into the actual custom P2 model while offline:

```bash
YOLO_OFFLINE=1 PYTHONPATH=src python -m fracture.utils.pretrained \
  --config configs/fracture_25d_p2_pretrained.yaml \
  --verify-transfer \
  --output reports/fold0_initialization_ab/pretrained_initialization_audit.json
```

Expected architecture evidence is four detection strides `[4, 8, 16, 32]`. The verified checkpoint used during development had SHA-256 `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`; the command fails if cached metadata and checkpoint content disagree.

### 3. Render the immutable V1 2.5D dataset

`/dev/shm` is volatile, so run this after every server reboot if the directory is absent. Generation deliberately refuses to overwrite an existing version.

```bash
cd /absolute/path/to/fracture
conda activate maskfo

if [ ! -f /dev/shm/iaaa_fracture/fracture_dataset_v1_25d_p2/fold_0.yaml ]; then
  PYTHONPATH=src python -m fracture.data.prepare_yolo \
    --config configs/fracture_25d_p2_pretrained.yaml
fi

cat /dev/shm/iaaa_fracture/fracture_dataset_v1_25d_p2/fold_stats.csv
sha256sum /dev/shm/iaaa_fracture/fracture_dataset_v1_25d_p2/manifest.csv
```

For the frozen dataset, Fold 0 should contain 816 sampled training images (408 positive and 408 negative) and 1,633 validation images. The full manifest should contain 7,508 images, 260 positive slices, and 356 boxes.

### 4. Start the new Fold-0 run in tmux

First ensure the chosen output name does not already exist. Never overwrite the historical V2 directory or a partial experiment—choose a new run name if necessary.

```bash
cd /absolute/path/to/fracture
FRACTURE_RUN=v2_25d_p2_coco_pretrained
test ! -e "outputs/fold_0/${FRACTURE_RUN}"

tmux new-session -d -s fracture -c "$PWD" \
  "exec env YOLO_CONFIG_DIR=/tmp/ultralytics-maskfo PYTHONPATH=src YOLO_OFFLINE=1 \
  conda run --no-capture-output -n maskfo python \
  -m fracture.training.train_detector \
  --config configs/fracture_25d_p2_pretrained.yaml \
  --fold 0 \
  --dataset-yaml /dev/shm/iaaa_fracture/fracture_dataset_v1_25d_p2/fold_0.yaml \
  --run-name ${FRACTURE_RUN}"

tmux set-option -t fracture remain-on-exit on
```

Monitor it from another terminal:

```bash
tmux list-sessions
tmux attach -t fracture
```

Detach without stopping training with `Ctrl+B`, then `D`. Non-interactive monitoring commands are:

```bash
tmux capture-pane -p -t fracture -S -80
watch -n 2 nvidia-smi
tail -f outputs/fold_0/v2_25d_p2_coco_pretrained/results.csv
```

`results.csv` appears after the first completed epoch. Early stopping uses patience 35; do not select the model from one early epoch.

### 5. Resume the new run after an interruption

Use the new experiment's `last.pt`, never the historical V2 checkpoint. The checkpoint restores its original optimizer, scheduler, dataset, seed, and run directory.

```bash
cd /absolute/path/to/fracture
test -f outputs/fold_0/v2_25d_p2_coco_pretrained/weights/last.pt

tmux new-session -d -s fracture-resume -c "$PWD" \
  "exec env YOLO_CONFIG_DIR=/tmp/ultralytics-maskfo PYTHONPATH=src YOLO_OFFLINE=1 \
  conda run --no-capture-output -n maskfo python \
  -m fracture.training.train_detector \
  --config configs/fracture_25d_p2_pretrained.yaml \
  --fold 0 \
  --dataset-yaml /dev/shm/iaaa_fracture/fracture_dataset_v1_25d_p2/fold_0.yaml \
  --run-name v2_25d_p2_coco_pretrained_resume \
  --resume-from outputs/fold_0/v2_25d_p2_coco_pretrained/weights/last.pt"

tmux set-option -t fracture-resume remain-on-exit on
```

### 6. Evaluate both detectors identically

Run these only after training has completed. Both commands use 768-pixel inference, raw maximum slice confidence, and no calibration. They also save slice boxes for error analysis.

```bash
cd /absolute/path/to/fracture
conda activate maskfo

YOLO_OFFLINE=1 PYTHONPATH=src python -m fracture.evaluation.predict_fold \
  --config configs/fracture_25d_p2_ab_local_transfer.yaml \
  --fold 0 \
  --weights outputs/fold_0/v1_25d_p2/weights/best.pt \
  --output-dir reports/fold0_initialization_ab \
  --name baseline_v2

YOLO_OFFLINE=1 PYTHONPATH=src python -m fracture.evaluation.predict_fold \
  --config configs/fracture_25d_p2_pretrained.yaml \
  --fold 0 \
  --weights outputs/fold_0/v2_25d_p2_coco_pretrained/weights/best.pt \
  --output-dir reports/fold0_initialization_ab \
  --name coco_pretrained_v2
```

Create the full comparison, score distributions, box-size analysis, review CSVs, isolated-fracture QWK, and visual cases:

```bash
YOLO_OFFLINE=1 PYTHONPATH=src python -m fracture.utils.pretrained \
  --config configs/fracture_25d_p2_ab_local_transfer.yaml \
  --verify-transfer \
  --output reports/fold0_initialization_ab/baseline_initialization_audit.json

YOLO_OFFLINE=1 PYTHONPATH=src python -m fracture.evaluation.compare_initializations \
  --config configs/fracture_25d_p2_pretrained.yaml \
  --baseline-studies reports/fold0_initialization_ab/baseline_v2.csv \
  --pretrained-studies reports/fold0_initialization_ab/coco_pretrained_v2.csv \
  --baseline-slices reports/fold0_initialization_ab/baseline_v2_slices.csv \
  --pretrained-slices reports/fold0_initialization_ab/coco_pretrained_v2_slices.csv \
  --baseline-training-results outputs/fold_0/v1_25d_p2/results.csv \
  --pretrained-training-results outputs/fold_0/v2_25d_p2_coco_pretrained/results.csv \
  --baseline-initialization reports/fold0_initialization_ab/baseline_initialization_audit.json \
  --pretrained-initialization reports/fold0_initialization_ab/pretrained_initialization_audit.json \
  --output-dir reports/fold0_initialization_ab
```

Do not fit a final aggregator or calibrator on Fold 0; it has only six positive studies. Do not start folds 1–4 until this report supports that decision.

### 7. Offline acceptance and runtime checks

```bash
YOLO_OFFLINE=1 PYTHONPATH=src python scripts/acceptance_smoke.py \
  --weights outputs/fold_0/v2_25d_p2_coco_pretrained/weights/best.pt \
  --study-dir iaaa-contest-bct/Data/training/2265 \
  --study-dir iaaa-contest-bct/Data/training/1734 \
  --input-mode 2.5d \
  --image-size 768 \
  --aggregation-method max \
  --calibration-method none \
  --device 0 \
  --output reports/fold0_initialization_ab/acceptance_smoke.json

YOLO_OFFLINE=1 PYTHONPATH=src python -m fracture.inference.benchmark \
  --weights outputs/fold_0/v2_25d_p2_coco_pretrained/weights/best.pt \
  --study-dir iaaa-contest-bct/Data/training/2265 \
  --study-dir iaaa-contest-bct/Data/training/1734 \
  --input-mode 2.5d \
  --image-size 768 \
  --aggregation-method max \
  --calibration-method none \
  --device 0 \
  --output reports/fold0_initialization_ab/runtime_benchmark.json
```

For a stronger p95 estimate, append additional positive, negative, large, and JPEG-Lossless studies with repeated `--study-dir` arguments.

To prove final inference does not need the generic development checkpoint, move it temporarily, run the smoke test above, and restore it even if the test fails:

```bash
mv artifacts/pretrained/yolo11s.pt /tmp/yolo11s.pt.temporarily-unavailable

YOLO_OFFLINE=1 PYTHONPATH=src python scripts/acceptance_smoke.py \
  --weights outputs/fold_0/v2_25d_p2_coco_pretrained/weights/best.pt \
  --study-dir iaaa-contest-bct/Data/training/2265 \
  --input-mode 2.5d --image-size 768 \
  --aggregation-method max --calibration-method none \
  --device 0 \
  --output reports/fold0_initialization_ab/checkpoint_independence_smoke.json

mv /tmp/yolo11s.pt.temporarily-unavailable artifacts/pretrained/yolo11s.pt
```

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

Full inference across all 68 held-out fold-0 studies gave AUROC `0.74866` and PR-AUC `0.32299`. At the official `0.5` threshold it produced zero true positives, six false negatives, and specificity `1.0`. An exploratory threshold of `0.203125` reached sensitivity `0.6667`, specificity `0.7903`, and F1 `0.3478`; this threshold is diagnostic only and must not be treated as an unbiased final threshold. See `reports/fold0_v0_single_metrics.json` and `reports/fold0_v0_single.csv`.

## Active V2 experiment: 2.5D + P2

V2 addresses the observed baseline weaknesses without downloading external weights:

- Three adjacent CT slices (`z-1`, `z`, `z+1`) are used as the input channels.
- A stride-4 P2 detection head is added for small fracture boxes.
- Input resolution increases from 640 to 768 pixels.
- Fold-0 training sampling changes from 204 positive/1,020 negative images to 408 positive/408 negative samples.
- Compatible backbone and neck parameters initialize from the locally trained V0 checkpoint; 297 of 593 tensors transfer into the modified architecture.
- AdamW uses a lower `0.0005` learning rate, cosine decay, and patience 35.
- Batch size 2 uses approximately 2.33 GB of the RTX 3050's available VRAM.
- `YOLO_OFFLINE=1` prevents package/version and weight network checks.

### Historical V2 status

The existing local-transfer V2 run is incomplete but preserved as the fixed comparison baseline. It contains 22 completed epochs; its best validation result occurred at displayed epoch 2:

| Metric | Existing V2 best |
| --- | ---: |
| Precision | `0.20050` |
| Recall | `0.06250` |
| mAP50 | `0.04758` |
| mAP50-95 | `0.02694` |

Its final completed epoch (22) had precision `0.08269`, recall `0.13750`, mAP50 `0.01915`, and mAP50-95 `0.00943`. These detector results are poor and motivate the controlled initialization experiment, but they do not prove that COCO initialization is better. Use the commands in the second-server handoff to measure that claim.

Do not resume into `outputs/fold_0/v1_25d_p2/`, because Ultralytics would modify the historical baseline. Do not run folds 1–4 until the Fold-0 baseline-versus-pretrained comparison is complete.

## Improvement checklist

- [x] Add explicit, mutually exclusive random/local-transfer/official-COCO initialization modes with cached checkpoint provenance.
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
    input_mode="2.5d",
    image_size=768,
)
fracture_prob = predictor.predict(study_dir)
```

The API receives no annotations or training metadata. `model.py` and `submission.py` are thin adapters around it.

## Required empirical completion

Once data is available: inspect metadata/schema; run the audit; freeze patient/study folds; visualize GT; generate versioned datasets; train all folds; generate OOF slice predictions; review candidates; train original/corrected comparisons on identical folds; fit the logistic aggregator and calibrator on OOF only; calculate detection/study/QWK metrics; benchmark positive, negative, large, and compressed-DICOM studies; then run evaluator-style inference with only submission artifacts. Record all results in the existing report templates.

The project must not be called competition-complete until those empirical steps pass.
