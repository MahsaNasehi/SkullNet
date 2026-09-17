"""CPU CLI for exact official triage conversion and three-class Macro-F1.

Input JSON files must each map study_id to the exact seven-intermediate
schema accepted by ``triage_from_intermediates``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from triage_macro_f1 import triage_from_intermediates, triage_macro_f1_report


def load_mapping(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("%s must contain a non-empty study_id -> intermediates mapping" % path)
    if not all(isinstance(value, dict) for value in payload.values()):
        raise ValueError("Every study value in %s must be an intermediates mapping" % path)
    return {str(key): value for key, value in payload.items()}


def evaluate(ground_truth: Dict[str, Dict[str, Any]], predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if set(ground_truth) != set(predictions):
        raise ValueError(
            "Study coverage mismatch: missing predictions=%s; unexpected predictions=%s"
            % (sorted(set(ground_truth) - set(predictions))[:10], sorted(set(predictions) - set(ground_truth))[:10])
        )
    studies = sorted(ground_truth)
    truth = [triage_from_intermediates(ground_truth[study]) for study in studies]
    predicted = [triage_from_intermediates(predictions[study]) for study in studies]
    return {
        "metric": "official_three_class_triage_macro_f1",
        "study_ids_sorted": studies,
        **triage_macro_f1_report(truth, predicted),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-json", type=Path, required=True)
    parser.add_argument("--predictions-json", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(load_mapping(args.ground_truth_json), load_mapping(args.predictions_json))
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError("Refusing to overwrite %s" % args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

