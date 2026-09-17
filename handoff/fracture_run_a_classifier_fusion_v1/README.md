# Fracture-only fusion handoff

This component returns only `fracture_prob`; the integrating team supplies ICH volumes and MLS to official triage.

Use `PYTHONPATH=src python -c 'from predictor import FractureFusionPredictor; print(FractureFusionPredictor().predict_study("/path/to/one/target-series-study"))'` from this directory. Pass `target_series_uid` when the study directory contains multiple series; ambiguous series fail closed.

See `PROVENANCE.json` for OOF scope and limitations. No fixed-test evidence is claimed. The model uses two separate forward passes; batch_size is configurable.
