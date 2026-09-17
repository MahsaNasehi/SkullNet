# Reproducibility references — active detector only

These small source/configuration files document the Run A FULL169 detector refit. They are **references**, not a self-contained retraining package: datasets, COCO initialization weights, generated lists, caches, logs and optimizer checkpoints are deliberately absent.

- `yolo26s-p2.yaml`: detector architecture.
- `run_protocol.json`: frozen FULL169 cohort, seed, fixed epoch count, training hyperparameters and source hashes.
- `args.yaml`: Ultralytics arguments recorded by the completed run.
- `train_run_a_full169.py`, `run_a_full169_protocol.py`, `train_yolo26s_p2.py`: training and initialization logic.
- `prepare_yolo26_dataset.py`: HU/window/2.5D data preparation reference.
- `fracture_postprocessing.json`: active detector aggregation and calibration; historical fusion settings are marked inactive.

No classifier architecture/training source is included because the FULL169 classifier has **not** been finalized and fusion is not in the active submission. The original repository retains the OOF experiment source.
