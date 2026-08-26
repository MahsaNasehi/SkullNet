# Fracture workstream report

Status: dataset audit and source implementation complete; pixel-level preparation, training, model evaluation, and final acceptance remain pending the official dependency environment and trained local artifacts.

## Inspection findings

- 338 studies from 320 patients; 28 fracture-positive and 310 fracture-negative studies.
- 7,683 DICOM files; metadata covers 7,508 and leaves 175 extra slices unknown.
- 5,176 JSON annotations: 260 positive slices, 4,916 explicit empty-box slices, and 356 fracture boxes.
- Metadata marks 7,248 slices negative, including 2,332 without JSON. Every existing JSON agrees with `SkullFracture`.
- All annotations use `boxes_xywh` and 512×512 shapes; no malformed, non-positive, or out-of-frame boxes were found.
- Box area ranges from 70 to 101,695 px² (median 2,200 px²).
- All 7,683 DICOM headers are 512×512 MONOCHROME2 and provide physical ordering fields. Transfer syntaxes: 3,148 JPEG Lossless, 2,256 Explicit VR Little Endian, and 2,279 Implicit VR Little Endian.
- Twenty-two studies have more DICOM files than metadata declares. Those extra 175 slices remain unknown and are excluded from detector negatives.
- No corrected annotations existed; a separate `annotations_corrected` root was created without changing organizer data.
- Fixed five-fold patient-grouped splits contain 68/68/68/67/67 studies and 6/6/6/5/5 positive studies.
- Python 3.12.3 exists locally, but required imaging/training packages are absent. Pixel decoding, training, metrics, re-annotation effect, runtime, and VRAM cannot yet be truthfully reported.

Run pixel decoding, preparation, training, OOF evaluation, and acceptance commands described in the README in the official environment. Do not replace pending metrics with invented values.
