# Fracture submission handoff — Run A FULL169 detector

`Model().predict(study_dir)` returns the seven required intermediate keys. **Only `fracture_prob` is predicted here.** The five ICH volumes and `MLS_mm` were already zero placeholders in this project's fracture-only submission; they have not been replaced or modified. The teammate must integrate real ICH/MLS components before treating this as a complete triage submission.

## Active fracture model

- `models/best.pt` is a byte-identical copy of the finalized Run A FULL169 detector (`final_deployment.pt`, fixed 59-epoch refit), SHA256 `109bc30f72b27269ce13edf6937fc8c3f7e6462e0f8a87b1074770e83ea44640`.
- No FULL169 `final_classifier.pt` or classifier completion marker existed at packaging. **Fusion is not active**; no classifier weight was substituted or invented.
- The former Fold 2 checkpoint and submission files are preserved in `../submit_backup_before_full169_20260916/`.

## Exact fracture inference

All DICOM slices of the one selected target series are read, including valid slices without metadata or annotation JSON. Multiple series fail closed unless `target_series_uid` is passed to `Model(...)`. Slices are ordered by physical position (InstanceNumber fallback), decoded to HU, MONOCHROME1 is handled, and a bone window of WL=800/WW=1600 is applied. Each input uses physical 2.5D context at ±5 mm; YOLO runs at `imgsz=768`, confidence `0.01`, and NMS IoU `0.5`.

The detector's per-slice maximum post-NMS confidences are combined by `top10_percent_mean`. The established Run A deployment mapping monotonically sends raw score `0.39717610677083337` to `fracture_prob=0.5`; official fracture presence is `fracture_prob >= 0.5`. Output is one finite float in `[0,1]`.

## Smoke test

From the repository root, after installing the project's requirements:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -c 'import sys; sys.path.insert(0, "submit"); from model import Model; m = Model(device="cpu"); p = m.predict("iaaa-contest-bct/Data/training/1488"); print(p); assert list(p) == ["V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH", "fracture_prob", "MLS_mm"]; assert 0 <= p["fracture_prob"] <= 1'
```

This manual command performs CPU inference on a development Study and can be slow; it was **not** run while preparing this handoff. `MODEL_PROVENANCE.md` distinguishes OOF evidence from the unmeasured final refit. `training_reference/` contains compact training source/configuration references, not datasets or executable training assets.
