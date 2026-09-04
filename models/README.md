# Final local inference artifacts

`models/deployment.yaml` is created only after complete OOF model selection. It
points to the chosen local detector, optional StudyMIL checkpoint, aggregator,
and calibrator and records the exact HU preprocessing contract. See
`deployment.example.yaml` and `scripts/create_deployment_manifest.py`.

Final inference must remain offline. Do not put download URLs or runtime install
commands in the deployment manifest.
