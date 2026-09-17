# Fracture-only IAAA submission artifact

Contents:

- `model.py`: standalone DICOM/HU/2.5D inference and `Model().predict(study_dir)` API.
- `models/best.pt`: YOLO26s-P2 Run A Fold 2 checkpoint selected by held-out
  `oracle_other_heads_macro_f1`.
- `SELECTION_REPORT.json`: OOF selection provenance, hashes and limitations.

The returned dictionary contains the required seven intermediate keys. This
artifact predicts skull fracture only; the five ICH volumes and `MLS_mm` are
zero placeholders. Consequently, the recorded oracle-other-head OOF Macro-F1
is not the expected final Macro-F1 of this standalone artifact. A complete team
submission must integrate real ICH and MLS predictions.

The deployment fracture score uses full-study `top10_percent_mean` aggregation.
Raw score `0.39717610677083337` is monotonically mapped to the official fracture
cutoff `0.5`. This setting was selected on all 169 OOF Studies and is labelled
apparent/deployment, not unbiased performance.

Selection was re-audited after all five `AUG_MILD_768` folds completed. Run A
remains selected: authoritative full-study OOF oracle-other-head Macro-F1 was
0.969627 for Run A, versus 0.954958 for Run B 1024 and 0.938469 for
`AUG_MILD_768`. No fixed-test result was used for this selection.

Minimal API usage:

```python
from model import Model

model = Model()
result = model.predict("/path/to/one/study_directory")
```
