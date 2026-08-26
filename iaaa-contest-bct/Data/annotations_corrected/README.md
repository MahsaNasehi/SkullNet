# Corrected fracture annotations

Reviewer-approved corrections go under `{series_id}/{SOPInstanceUID}.json` using the organizer JSON schema. Never edit `../annotations`. Every correction must have a matching accepted entry in `../../../reports/reannotation_log.csv`.

When no corrected JSON exists, the pipeline resolves the organizer annotation from `../annotations`.
