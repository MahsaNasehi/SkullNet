# IAAA 2026 skull-fracture workstream

This package implements the offline core path `study_dir -> DICOM/HU window -> detector + optional study MIL -> OOF-selected aggregation/calibration -> float fracture_prob`. It never downloads weights or installs dependencies at prediction time. Corrected annotations override originals without modifying organizer files; missing JSON is unknown by default.

## Training — start here

This is the short operational path for the current experiment. Run every command
from the repository root. The current experiment uses 2.5D CT input, a custom
YOLO11s P2 detector, and the selected bone window **WL=800, WW=1600**.

### 1. Activate the environment and verify the repository

```bash
cd /absolute/path/to/fracture
conda activate maskfo

PYTHONPATH=src python -m pytest -q
test -d iaaa-contest-bct/Data/training
test -d iaaa-contest-bct/Data/annotations
test -f iaaa-contest-bct/Data/training_df.pkl
test -s artifacts/pretrained/yolo11s.pt
df -h /dev/shm
```

Replace `/absolute/path/to/fracture` with the actual clone location. For example,
the local workstation path is currently:

```bash
cd "/home/mahsa-nasehi/Desktop/IAAA/fracture"
```

### 2. Prepare the HU dataset

The rendered training data lives in `/dev/shm` and disappears after a reboot.
Generate it only when `fold_0.yaml` is absent:

```bash
if [ ! -f /dev/shm/iaaa_fracture/fracture_dataset_v2_hu800_ww1600_original/fold_0.yaml ]; then
  PYTHONPATH=src python -m fracture.data.prepare_yolo \
    --config configs/fracture_25d_p2_hu800_original.yaml
fi
```

The generator never overwrites an existing directory. If the command reports
`Dataset version already exists` while `fold_0.yaml` is missing, the existing
directory is incomplete. Preserve it under another name and regenerate:

```bash
mv \
  /dev/shm/iaaa_fracture/fracture_dataset_v2_hu800_ww1600_original \
  /dev/shm/iaaa_fracture/fracture_dataset_v2_hu800_ww1600_original.incomplete

PYTHONPATH=src python -m fracture.data.prepare_yolo \
  --config configs/fracture_25d_p2_hu800_original.yaml
```

If the `.incomplete` name already exists, use `.incomplete_2` or another unique
name. Confirm successful preparation before training:

```bash
test -f /dev/shm/iaaa_fracture/fracture_dataset_v2_hu800_ww1600_original/fold_0.yaml
cat /dev/shm/iaaa_fracture/fracture_dataset_v2_hu800_ww1600_original/fold_stats.csv
```

### 3. Start the current Fold-0 training in tmux

Do not start the same run twice. First check that neither a training process nor
the tmux session is already active:

```bash
pgrep -af "fracture.training.train_detector"
tmux list-sessions 2>/dev/null || true
```

Then launch the run. Capturing the active environment's Python path makes the
tmux job independent of shell activation after detaching:

```bash
mkdir -p logs
MASKFO_PYTHON="$(command -v python)"

tmux new-session -d -s fracture_hu800 -c "$PWD" \
  "env PYTHON='$MASKFO_PYTHON' ./scripts/run_improved_pipeline.sh train-original-fold0 \
  2>&1 | tee logs/train-original-fold0.log"
```

Monitor it with either command:

```bash
tmux attach -t fracture_hu800
```

```bash
tail -f logs/train-original-fold0.log
```

Detach from tmux without stopping training with `Ctrl+B`, then `D`. The run is
written to:

```text
outputs/fold_0/v5_hu800_ww1600_original/
reports/fold0_v5_hu800_ww1600_original.csv
reports/fold0_v5_hu800_ww1600_original_slices.csv
reports/fold0_v5_hu800_ww1600_original_metrics.json
```

### 4. Check Fold 0 after it finishes

```bash
test -s outputs/fold_0/v5_hu800_ww1600_original/weights/best.pt
cat reports/fold0_v5_hu800_ww1600_original_metrics.json

python - <<'PY'
import pandas as pd

path = "outputs/fold_0/v5_hu800_ww1600_original/results.csv"
df = pd.read_csv(path)
best = df.loc[df["metrics/mAP50-95(B)"].idxmax()]
print(best[[
    "epoch",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]])
PY
```

Keep `best.pt`, `results.csv`, and all generated reports together. A zero-byte
checkpoint is corrupted and must never be used.

## What to run after Fold 0

Do not jump directly to final training. Use the following order so all model
selection remains out-of-fold (OOF) and patient-separated.

### Phase A — complete the original-label OOF baseline

If Fold 0 completed successfully, this command reuses it and trains folds 1–4:

```bash
./scripts/run_improved_pipeline.sh train-original-oof
```

For a long unattended run, launch it in a new tmux session:

```bash
MASKFO_PYTHON="$(command -v python)"
tmux new-session -d -s fracture_oof -c "$PWD" \
  "env PYTHON='$MASKFO_PYTHON' ./scripts/run_improved_pipeline.sh train-original-oof \
  2>&1 | tee logs/train-original-oof.log"
```

### Phase B — review label disagreements

Create the review queue from held-out predictions:

```bash
./scripts/run_improved_pipeline.sh review-original-oof
```

Review `reports/reannotation_candidates.csv` manually and record decisions in
`reports/reannotation_log.csv`. Predictions are suggestions only; the pipeline
does not silently modify organizer labels. Then validate the corrections:

```bash
./scripts/run_improved_pipeline.sh audit-corrections
```

### Phase C — train with reviewed/corrected labels

```bash
./scripts/run_improved_pipeline.sh prepare-base
./scripts/run_improved_pipeline.sh train-base-oof
```

### Phase D — hard-negative mining and retraining

```bash
./scripts/run_improved_pipeline.sh mine-hard-negatives
./scripts/run_improved_pipeline.sh prepare-hnm
./scripts/run_improved_pipeline.sh train-hnm-oof
```

### Phase E — study-level model and OOF fusion

The detector localizes suspicious slices; the MIL branch and calibrated fusion
produce the final study-level fracture probability required by the challenge.

```bash
./scripts/run_improved_pipeline.sh train-mil-oof
./scripts/run_improved_pipeline.sh fuse-oof
```

### Phase F — final all-data models and deployment package

Choose the epoch counts from the five OOF best-epoch histories, not from metrics
on the complete training set:

```bash
export FINAL_DETECTOR_EPOCHS=<selected_detector_epochs>
./scripts/run_improved_pipeline.sh train-final-detector

export FINAL_MIL_EPOCHS=<selected_mil_epochs>
./scripts/run_improved_pipeline.sh train-final-mil
```

Finally, provide the selected non-empty artifacts and create the deployment
manifest:

```bash
export FINAL_DETECTOR=outputs/fold_-1/final_hu800_hnm/weights/best.pt
export FINAL_STUDY_MODEL=outputs/final/final_study_mil_hu800/weights/best.pt
export FINAL_CALIBRATOR=models/final_hybrid/calibrator.joblib

./scripts/run_improved_pipeline.sh make-deployment
```

The detailed environment notes, historical experiments, and recovery commands
below are reference material. For a normal new training run, follow the steps in
**Training — start here** first.

## Dataset status

The supplied dataset is mounted in `iaaa-contest-bct/Data`. It contains 338 studies from 320 patients, 7,683 DICOM files, and 5,176 organizer annotation JSONs. Metadata is stored in `training_df.pkl`; its configured fracture columns agree with all existing `boxes_xywh` labels. See `reports/data_summary.json` for the complete audit.

The study target is strongly imbalanced: 28/338 studies are fracture-positive (`8.28%`) and 310/338 are negative (`91.72%`). There are 260 positive slices and 356 boxes. Splits are patient-grouped; sampling/oversampling is applied only to each training partition, while every held-out study and all of its slices remain in validation.

A full pixel-decode audit passed for all DICOM files, including 3,148 JPEG Lossless instances. Patient-grouped folds are frozen in `splits/folds.json`. The rendered YOLO dataset is deliberately stored in `/dev/shm/iaaa_fracture` because the main filesystem has limited free space. `/dev/shm` is volatile, so regenerate the rendered dataset after a reboot; the source DICOMs remain authoritative.

## Official environment versus the historical `maskfo` server

The final evaluator contract is Python `>=3.12,<3.13`, PyTorch `>=2.10,<3`, NumPy `>=2,<2.3`, and Ultralytics `>=8.3.240,<9`. `pyproject.toml` now expresses that contract. The existing RTX 5060 Ti server (`Python 3.10.12`, `torch 2.7.1+cu128`) is still useful for reproducing the historical runs, but passing its tests does not replace the required final Python 3.12 acceptance run.

Create a clean final-compatibility environment with:

```bash
conda create -n skullnet312 python=3.12 pip -y
conda activate skullnet312
python -m pip install --upgrade pip setuptools wheel

# Install a torch >=2.10 wheel that supports the server GPU/CUDA policy, then:
python -m pip install -e '.[dev]'
PYTHONPATH=src python -m fracture.utils.check_environment
PYTHONPATH=src python -m pytest -q
```

The environment checker is read-only: it reports versions/CUDA and never installs or downloads anything. Final inference is forced offline and loads only local artifacts.

## Historical `maskfo` environment and required packages

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

Because project metadata now intentionally enforces the official Python 3.12 contract, do not install this package into the historical Python 3.10 environment. Run it from the repository with `PYTHONPATH=src`:

```bash
cd /absolute/path/to/fracture
PYTHONPATH=src python -m pytest -q
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
PYTHONPATH=src python -m pytest -q
```

`fracture.utils.check_environment` intentionally fails under this legacy environment; run that acceptance check in `skullnet312` instead.

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

### First controlled improvement: 1024-pixel input

The first follow-up experiment increases only the Ultralytics train/validation
resolution from 768 to 1024 pixels. It retains the corrected COCO initialization,
YOLO11s P2–P5 architecture, 2.5D channels, Fold-0 split, sampled examples, batch
size, optimizer, augmentations, seed, and early-stopping settings. The existing
lossless rendered dataset is reused because its PNGs are stored at their original
resolution; `imgsz` is applied by Ultralytics when loading them.

After pulling this code on the remote server, verify the prerequisites:

```bash
cd /home/lung/lung_git/temp/SkullNet
conda activate maskfo

test -s artifacts/pretrained/yolo11s.pt
test -f /dev/shm/iaaa_fracture/fracture_dataset_v1_25d_p2/fold_0.yaml
test ! -e outputs/fold_0/v4_25d_p2_coco_pretrained_1024
```

If `/dev/shm` was cleared by a reboot, recreate the same immutable dataset first:

```bash
PYTHONPATH=src python -m fracture.data.prepare_yolo \
  --config configs/fracture_25d_p2_pretrained.yaml
```

Start the experiment in a persistent tmux session:

```bash
tmux new-session -d -s fracture_1024 -c "$PWD" \
  "exec env YOLO_CONFIG_DIR=/tmp/ultralytics-maskfo PYTHONPATH=src YOLO_OFFLINE=1 \
  conda run --no-capture-output -n maskfo python \
  -m fracture.training.train_detector \
  --config configs/fracture_25d_p2_pretrained_1024.yaml \
  --fold 0 \
  --dataset-yaml /dev/shm/iaaa_fracture/fracture_dataset_v1_25d_p2/fold_0.yaml \
  --epochs 60 \
  --run-name v4_25d_p2_coco_pretrained_1024"

tmux set-option -t fracture_1024 remain-on-exit on
tmux attach -t fracture_1024
```

Detach with `Ctrl+B`, then `D`. Monitor without attaching using:

```bash
tmux capture-pane -p -t fracture_1024 -S -80
tail -f outputs/fold_0/v4_25d_p2_coco_pretrained_1024/results.csv
watch -n 2 nvidia-smi
```

The expected startup evidence is `Image sizes 1024 train, 1024 val`, detection
strides `[4, 8, 16, 32]`, `Transferred 593/593 items from pretrained weights`, and
`"training_initialization_verified": true` in:

```text
outputs/fold_0/v4_25d_p2_coco_pretrained_1024_initialization.json
```

Compare its best row against the corrected 768-pixel run without selecting on a
single final epoch:

```bash
python - <<'PY'
import pandas as pd

runs = {
    "768": "outputs/fold_0/v3_25d_p2_coco_pretrained_fixed/results.csv",
    "1024": "outputs/fold_0/v4_25d_p2_coco_pretrained_1024/results.csv",
}
for resolution, path in runs.items():
    df = pd.read_csv(path)
    best = df.loc[df["metrics/mAP50-95(B)"].idxmax()]
    print(resolution, best[[
        "epoch", "metrics/precision(B)", "metrics/recall(B)",
        "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    ]].to_dict())
PY
```

The 1024 model should advance only if study-level evaluation also improves; the
six positive Fold-0 studies make detector metrics alone too unstable for a final
decision.

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

## HU-window decision and implemented end-to-end pipeline

### Train-only HU analysis

The screenshot settings were registered and measured on the training side of Fold 0 only: 270 studies, 204 positive slices, 276 boxes, 356,019 bone-region pixels (`HU >= 150`). Boxes are region labels rather than fracture masks, so this selects a defensible starting point—not a guaranteed optimum. Full numeric output is in `reports/hu_window_analysis_fold0_train.json`.

| Level / width | HU interval | Bone pixels clipped low/high | Within-bone dynamic std | Interpretation |
| --- | --- | ---: | ---: | --- |
| 500 / 2500 (old) | -750…1750 | 0.00% / 0.21% | 0.139 | Preserves range but compresses cortical contrast |
| 925 / 2050 | -100…1950 | 0.00% / 0.09% | 0.170 | Safe broad window from the screenshots |
| 1125 / 450 | 900…1350 | 76.51% / 3.78% | 0.262 | Too narrow; most bone collapses to black/white |
| 550 / 1350 | -125…1225 | 0.00% / 6.77% | 0.238 | Better contrast but clips dense cortex |
| 700 / 850 | 275…1125 | 16.44% / 10.28% | 0.345 | Visually sharp but severe two-sided clipping |
| **800 / 1600 (selected)** | **0…1600** | **0.00% / 0.71%** | **0.216** | Stronger contrast with little high-HU clipping |

The new baseline therefore uses fixed `WL=800`, `WW=1600` for validation/inference. Training duplicates receive deterministic, train-only jitter of `level ±150` and `width ±20%` (minimum width 1000). A common window is applied to all three 2.5D channels, and neighbours are selected near `z±5 mm` when physical positions are available. The narrow `1125/450` setting and pseudocolor are not primary inputs: the former destroys most within-cortex intensity ordering, while a colormap would consume the three RGB channels currently carrying z-context. They can be evaluated later only as controlled OOF ablations.

### Implemented phases

The code now supports the final task directly (`whole CT study -> fracture_prob`), so bounding boxes are not required at inference. The detector branch supplies localization evidence, a compact study-MIL branch learns directly from complete-study labels, and the final OOF logistic model decides whether fusion actually helps. The new manifest preserves all 7,683 slices for the study branch, but marks the 175 unknown slices as ineligible for detector targets and masks them out of the auxiliary slice loss. No study is scored by a detector, MIL model, or calibrator that was fitted on that study during model selection.

Run one phase at a time. Long GPU phases should be placed in `tmux`; on a single GPU do not launch detector jobs in parallel.

```bash
cd /home/lung/lung_git/temp/SkullNet
conda activate maskfo
chmod +x scripts/run_improved_pipeline.sh

# Phase 0: read-only data/HU checks
./scripts/run_improved_pipeline.sh audit
./scripts/run_improved_pipeline.sh analyse-hu

# Phase 1: immutable HU800 dataset using organizer boxes only
./scripts/run_improved_pipeline.sh prepare-original

# Immediate decision gate: run Fold 0 first and compare study-level PR-AUC/AUROC
./scripts/run_improved_pipeline.sh train-original-fold0

# Continue only if the fixed Fold-0 comparison justifies the HU change
./scripts/run_improved_pipeline.sh train-original-oof

# Phase 2: OOF disagreements -> human review queue and PNGs
./scripts/run_improved_pipeline.sh review-original-oof
# Human reviews candidates, writes accepted JSONs under
# iaaa-contest-bct/Data/annotations_corrected/<series>/<sop>.json,
# and records every decision in reports/reannotation_log.csv.
./scripts/run_improved_pipeline.sh audit-corrections

# Phase 3: same folds/model/HU settings with accepted corrections
./scripts/run_improved_pipeline.sh prepare-base
./scripts/run_improved_pipeline.sh train-base-oof

# Phase 4: only verified-negative OOF false positives become hard negatives
./scripts/run_improved_pipeline.sh mine-hard-negatives
./scripts/run_improved_pipeline.sh prepare-hnm
./scripts/run_improved_pipeline.sh train-hnm-oof

# Phase 5: independent low-memory whole-study MIL branch
./scripts/run_improved_pipeline.sh train-mil-oof

# Phase 6: complete five-fold OOF selection, fusion, and cross-fit calibration
./scripts/run_improved_pipeline.sh fuse-oof

# Phase 7: only after choosing epoch counts from OOF histories, refit on all data
FINAL_DETECTOR_EPOCHS=NN ./scripts/run_improved_pipeline.sh train-final-detector
FINAL_MIL_EPOCHS=NN ./scripts/run_improved_pipeline.sh train-final-mil
```

For example, start the first long five-fold phase in a persistent tmux pane:

```bash
mkdir -p logs
tmux new-session -d -s fracture_hu_oof -c "$PWD" \
  "conda run --no-capture-output -n maskfo \
  ./scripts/run_improved_pipeline.sh train-original-fold0 \
  2>&1 | tee logs/train-original-fold0.log"
tmux set-option -t fracture_hu_oof remain-on-exit on
tmux attach -t fracture_hu_oof
```

Detach with `Ctrl+B`, then `D`; inspect later with `tmux capture-pane -p -t fracture_hu_oof -S -100` or `tail -f logs/train-original-fold0.log`. A session that displays `[exited]` has finished or failed; `remain-on-exit` preserves its final pane so the error is not lost. Compare `reports/fold0_v5_hu800_ww1600_original_metrics.json` with `reports/v3_coco_pretrained_eval_metrics.json` before spending GPU time on folds 1–4; visual contrast alone is not an acceptance criterion.

`train-original-oof`, `train-base-oof`, and `train-hnm-oof` each train folds 0–4 sequentially and write per-study and per-slice OOF CSVs. `generate_oof` refuses incomplete folds by default; `--allow-incomplete-oof` creates diagnostics but deliberately does not write deployable models. Aggregator selection prioritizes PR-AUC and sensitivity rather than the misleadingly high class-imbalanced QWK. Calibration metrics are cross-fitted, while the final saved calibrator is fitted only after all OOF decisions are complete.

After inspecting `reports/final_oof/final_metrics.json`, explicitly select non-empty artifacts and create the no-argument evaluator manifest:

```bash
FINAL_DETECTOR=outputs/fold_-1/final_hu800_hnm/weights/last.pt \
FINAL_STUDY_MODEL=outputs/final/final_study_mil_hu800/weights/best.pt \
FINAL_CALIBRATOR=models/final_hybrid/calibrator.joblib \
./scripts/run_improved_pipeline.sh make-deployment

YOLO_OFFLINE=1 PYTHONPATH=src python scripts/acceptance_smoke.py \
  --weights "$FINAL_DETECTOR" \
  --study-model "$FINAL_STUDY_MODEL" \
  --aggregator models/final_hybrid/aggregator.joblib \
  --calibrator models/final_hybrid/calibrator.joblib \
  --study-dir iaaa-contest-bct/Data/training/2265 \
  --input-mode 2.5d --image-size 768 \
  --window-level 800 --window-width 1600 --context-distance-mm 5 \
  --device 0
```

The final detector and MIL commands refit the OOF-selected configurations on all 338 studies for fixed epoch counts; their validation output is in-sample and must not be quoted as performance. Use the detector's `last.pt` for this fixed-epoch refit—not the in-sample-selected `best.pt`. A deployable choice must be justified by complete five-fold OOF results. An ensemble can be added only after its calibration is evaluated under the same leakage-safe protocol.

### What was fixed

- [x] Rejected zero-byte/truncated detector, resume, and study-model checkpoints before loading.
- [x] Fixed optional aggregator/calibrator handling in the acceptance smoke test.
- [x] Made HU preprocessing identical across dataset generation, held-out evaluation, CLI submission, benchmark, and final predictor.
- [x] Added physical-distance 2.5D context for irregular 5/8 mm slice spacing.
- [x] Restricted window jitter to training duplicates and rendered variants only for repeat-eligible slices, limiting `/dev/shm` growth.
- [x] Made hard-negative mining OOF-only and restricted it to metadata-verified negative slices; unknown slices cannot silently become negatives.
- [x] Added OOF review queues/visualizations and mandatory accepted-decision provenance for corrected annotations.
- [x] Added compact whole-study MIL training, detector+MIL fusion, physical run-length features, cross-fitted calibration, and incomplete-OOF safeguards.
- [x] Added a local deployment manifest so the official no-argument `Model()` adapter can load preprocessing and artifacts without network/runtime installation.

No new accuracy number is claimed until the new five-fold runs finish. The existing best corrected-COCO Fold-0 detector remains weak (`mAP50≈0.068`, `mAP50-95≈0.038`); the purpose of these changes is to run a controlled, leakage-safe experiment rather than infer improvement from visual contrast alone.

## Inference API

```python
from fracture import FracturePredictor

predictor = FracturePredictor(
    "models/best.pt",
    aggregator_path="models/aggregator.joblib",  # omit for max
    calibrator_path="models/calibrator.joblib",  # omit for none
    study_model_path="models/study_mil.pt",       # optional hybrid branch
    aggregation_method="logistic",
    calibration_method="platt",
    input_mode="2.5d",
    image_size=768,
    window_level=800,
    window_width=1600,
    context_distance_mm=5,
)
fracture_prob = predictor.predict(study_dir)
```

The API receives no annotations or training metadata. `model.py` and `submission.py` are thin adapters around it.

## Required empirical completion

Once data is available: inspect metadata/schema; run the audit; freeze patient/study folds; visualize GT; generate versioned datasets; train all folds; generate OOF slice predictions; review candidates; train original/corrected comparisons on identical folds; fit the logistic aggregator and calibrator on OOF only; calculate detection/study/QWK metrics; benchmark positive, negative, large, and compressed-DICOM studies; then run evaluator-style inference with only submission artifacts. Record all results in the existing report templates.

The project must not be called competition-complete until those empirical steps pass.
