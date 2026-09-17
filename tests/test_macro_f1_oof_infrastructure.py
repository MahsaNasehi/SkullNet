import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from aggregate_full_study_oof_cache import (
    CANDIDATE_AGGREGATORS, aggregate_scores, apply_higher_confidence_floor,
)
from audit_oof_dataset_coverage import run as run_coverage_audit
from cache_full_study_oof import (
    SCORE_PROTOCOLS, enforce_vram_guard, select_explicit_fold_studies,
)
from dicom_series_guard import classify_header, select_and_order_target_headers
from compare_full_study_oof_caches import (
    delta_metrics, metrics_for_probability, paired_tables, validate_complete_cache,
    validate_exact_paired_slice_coverage,
)
from evaluate_official_triage_macro_f1 import evaluate as evaluate_official
from fracture_macro_f1_oof import (
    cross_fitted_postprocess,
    monotonic_operating_point_map,
    oracle_triage_report,
)
from triage_macro_f1 import triage_from_intermediates, triage_macro_f1_report, validate_intermediates
from oof_cohort import derive_explicit_validation_cohort
from filter_full_study_cache_cohort import materialize_filtered_cache


def intermediate(**changes):
    row = {
        "V_EDH": 0.0, "V_SDH": 0.0, "V_IPH": 0.0,
        "V_SAH": 0.0, "V_IVH": 0.0, "fracture_prob": 0.0, "MLS_mm": 0.0,
    }
    row.update(changes)
    return row


class OfficialTriageTests(unittest.TestCase):
    def test_official_boundary_rules(self):
        self.assertEqual(triage_from_intermediates(intermediate()), 0)
        self.assertEqual(triage_from_intermediates(intermediate(fracture_prob=0.5)), 1)
        self.assertEqual(triage_from_intermediates(intermediate(MLS_mm=5.0)), 1)
        self.assertEqual(triage_from_intermediates(intermediate(MLS_mm=5.0, fracture_prob=0.5)), 2)
        self.assertEqual(triage_from_intermediates(intermediate(V_IPH=15.0, fracture_prob=0.5)), 2)
        self.assertEqual(triage_from_intermediates(intermediate(V_IPH=70.0)), 2)

    def test_exact_schema_validation(self):
        self.assertEqual(validate_intermediates(intermediate())["V_EDH"], 0.0)
        with self.assertRaises(ValueError):
            validate_intermediates({key: value for key, value in intermediate().items() if key != "MLS_mm"})
        with self.assertRaises(ValueError):
            validate_intermediates(dict(intermediate(), unexpected=1))

    def test_macro_report_has_fixed_three_classes(self):
        report = triage_macro_f1_report([0, 1, 2], [0, 1, 1])
        self.assertEqual(report["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 1, 0]])
        self.assertEqual(set(report["classwise"]), {"0", "1", "2"})
        self.assertAlmostEqual(report["accuracy"], 2.0 / 3.0)

    def test_generic_evaluator_requires_exact_study_coverage(self):
        truth = {"a": intermediate(), "b": intermediate(fracture_prob=1.0)}
        report = evaluate_official(truth, dict(truth))
        self.assertAlmostEqual(report["pooled_macro_f1"], 2.0 / 3.0)
        with self.assertRaises(ValueError):
            evaluate_official(truth, {"a": intermediate()})


class CrossFitTests(unittest.TestCase):
    def synthetic_oof(self):
        rows = []
        for fold in range(5):
            rows.extend([
                {"study_id": "%d_n" % fold, "patient_id": "%d_n" % fold, "fold": fold,
                 "fracture_true": False, "top3_mean": 0.1},
                {"study_id": "%d_p" % fold, "patient_id": "%d_p" % fold, "fold": fold,
                 "fracture_true": True, "top3_mean": 0.8},
                {"study_id": "%d_c" % fold, "patient_id": "%d_c" % fold, "fold": fold,
                 "fracture_true": True, "top3_mean": 0.9, "V_IPH": 20.0},
            ])
        table = pd.DataFrame(rows)
        for name in ("V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH", "MLS_mm"):
            if name not in table:
                table[name] = 0.0
            else:
                table[name] = table[name].fillna(0.0)
        return table

    def test_monotonic_map_places_operating_point_at_half(self):
        mapped = monotonic_operating_point_map([0.0, 0.2, 0.4, 1.0], 0.4)
        self.assertTrue(np.all(np.diff(mapped) >= 0))
        self.assertAlmostEqual(mapped[2], 0.5)

    def test_crossfit_never_uses_heldout_fold_for_selection(self):
        table = self.synthetic_oof()
        crossfit, selections = cross_fitted_postprocess(table, "top3_mean")
        self.assertEqual(len(selections), 5)
        self.assertEqual(len(crossfit), len(table))
        report = oracle_triage_report(crossfit, crossfit["crossfit_fracture_prob"])
        self.assertAlmostEqual(report["pooled_macro_f1"], 1.0)
        for selection in selections:
            self.assertEqual(selection["selection_studies"], 12)

    def test_paired_comparator_reports_b_minus_a(self):
        base = self.synthetic_oof().copy()
        base["included_target_series_slices"] = 10
        base["included_metadata_unknown_slices"] = 1
        a, b = base.copy(), base.copy()
        a["crossfit_fracture_prob"] = 0.0
        b["crossfit_fracture_prob"] = np.where(b["fracture_true"], 1.0, 0.0)
        paired = paired_tables(a, b)
        first = metrics_for_probability(paired, "crossfit_fracture_prob_a")
        second = metrics_for_probability(paired, "crossfit_fracture_prob_b")
        delta = delta_metrics(first, second)
        self.assertGreater(delta["oracle_other_heads_macro_f1"], 0.0)
        self.assertGreater(delta["fracture_sensitivity"], 0.0)

    def test_exact_cache_slice_coverage_accepts_label_only_protocol_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first", root / "second"
            for cache, protocol in (
                (first, "target_series_header_guard_v1_all_valid_dicoms"),
                (second, "explicit_validation_studies_v2_all_valid_target_series_dicoms"),
            ):
                path = cache / "fold_0/studies/s.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({
                    "fold": 0, "series_id": "s", "patient_id": "p",
                    "study_inclusion_protocol": protocol,
                    "included_metadata_unknown_slices": 1,
                    "slices": [{"sop_uid": "a"}, {"sop_uid": "b"}],
                }))
            report = validate_exact_paired_slice_coverage(first, second)
            self.assertEqual(report["studies"], 1)
            self.assertEqual(report["slices"], 2)
            self.assertEqual(report["metadata_unknown_slices"], 1)

    def test_exact_cache_slice_coverage_rejects_sop_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, sop in (("first", "a"), ("second", "different")):
                path = root / name / "fold_0/studies/s.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({
                    "fold": 0, "series_id": "s", "patient_id": "p",
                    "slices": [{"sop_uid": sop}],
                }))
            with self.assertRaisesRegex(ValueError, "SOP sequence"):
                validate_exact_paired_slice_coverage(root / "first", root / "second")


class AggregationAndGuardTests(unittest.TestCase):
    def test_frozen_aggregator_set_and_values(self):
        self.assertEqual(
            CANDIDATE_AGGREGATORS,
            ("max", "top2_mean", "top3_mean", "top5_mean", "top10_percent_mean", "consecutive"),
        )
        scores = [0.1, 0.8, 0.6, 0.0]
        self.assertAlmostEqual(aggregate_scores(scores, "max"), 0.8)
        self.assertAlmostEqual(aggregate_scores(scores, "top2_mean"), 0.7)
        self.assertAlmostEqual(aggregate_scores(scores, "consecutive"), 0.7)

    def test_vram_guard_precedes_model_loading(self):
        with patch("cache_full_study_oof.query_free_vram_gib", return_value=4.0):
            with self.assertRaises(RuntimeError):
                enforce_vram_guard("0", False)
            self.assertEqual(enforce_vram_guard("0", True), 4.0)

    def test_confidence_floor_can_only_be_raised_without_claiming_new_nms(self):
        source = {
            "source_confidence_floor": 0.001,
            "boxes": [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]],
            "scores": [0.005, 0.01, 0.2],
        }
        filtered = apply_higher_confidence_floor(source, 0.01)
        self.assertEqual(filtered["scores"], [0.01, 0.2])
        self.assertEqual(filtered["num_detections"], 2)
        self.assertFalse(filtered["nms_reconstructed"])
        with self.assertRaises(ValueError):
            apply_higher_confidence_floor(source, 0.0001)

    def test_protocol_identities_are_frozen(self):
        self.assertEqual(SCORE_PROTOCOLS["proposal_diagnostic"]["confidence_floor"], 0.001)
        self.assertEqual(SCORE_PROTOCOLS["proposal_diagnostic"]["nms_iou"], 0.7)
        self.assertEqual(SCORE_PROTOCOLS["deployment_score"]["confidence_floor"], 0.01)
        self.assertEqual(SCORE_PROTOCOLS["deployment_score"]["nms_iou"], 0.5)

    def test_synthetic_header_classification_and_physical_order(self):
        profile = {
            "canonical_series_uid": "target", "rows": 512, "columns": 512,
            "orientation": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }
        def header(sop, z, **changes):
            value = {
                "filesystem_path": "/tmp/%s.dcm" % sop,
                "SOPInstanceUID": sop, "SeriesInstanceUID": "target", "Modality": "CT",
                "ImageType": "ORIGINAL\\PRIMARY", "SeriesDescription": "HEAD",
                "ProtocolName": "BRAIN", "Rows": 512, "Columns": 512,
                "ImageOrientationPatient": profile["orientation"],
                "ImagePositionPatient": (0.0, 0.0, float(z)), "InstanceNumber": int(z + 10),
            }
            value.update(changes)
            return value
        target = header("a", 2)
        self.assertEqual(classify_header(target, profile)[0], "include_target_series")
        self.assertEqual(
            classify_header(header("b", 1, SeriesInstanceUID="other"), profile)[0],
            "exclude_foreign_series",
        )
        self.assertEqual(
            classify_header(header("c", 3, ImageType="LOCALIZER"), profile)[0],
            "exclude_localizer_or_scout",
        )
        self.assertEqual(
            classify_header(header("d", 4, Rows=256), profile)[0],
            "exclude_incompatible_geometry",
        )
        unknown_target = header("e", 1, metadata_backed=False)
        included, excluded, ordering = select_and_order_target_headers(
            [target, unknown_target, header("b", 0, SeriesInstanceUID="other")], profile
        )
        self.assertEqual([row["SOPInstanceUID"] for row in included], ["e", "a"])
        self.assertFalse(included[0]["metadata_backed"])
        self.assertEqual(len(excluded), 1)
        self.assertEqual(ordering, "orientation_position_projection")


class CoverageAuditTests(unittest.TestCase):
    def test_materializer_filters_extra_and_preserves_selected_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source", root / "filtered"
            folds = []
            for fold in range(5):
                study, patient = "s%d" % fold, "p%d" % fold
                folds.append({
                    "fold": fold, "study_ids": [study], "patient_ids": [patient],
                    "study_to_patient": {study: patient},
                })
                studies = source / ("fold_%d" % fold) / "studies"
                studies.mkdir(parents=True)
                payload = {
                    "fold": fold, "series_id": study, "patient_id": patient,
                    "prediction_protocol": "heldout_patient_full_study",
                    "score_generation_protocol": "deployment_score",
                    "study_inclusion_protocol": "all_valid_slices",
                    "inference_signature": {
                        "imgsz": 768, "conf": 0.01, "nms_iou": 0.5,
                        "score_generation_protocol": "deployment_score",
                    },
                    "slices": [{"sop_uid": "a"}, {"sop_uid": "b"}],
                }
                selected_path = studies / (study + ".json")
                selected_path.write_text(json.dumps(payload))
                actual = [study]
                if fold == 3:
                    extra = dict(payload, series_id="extra")
                    (studies / "extra.json").write_text(json.dumps(extra))
                    actual.append("extra")
                (source / ("fold_%d" % fold) / "COMPLETE.json").write_text(json.dumps({
                    "fold": fold, "studies": len(actual), "expected_study_ids": actual,
                }))
            cohort = {
                "cohort_protocol": "synthetic_explicit", "folds": folds,
                "total_unique_validation_studies": 5, "per_fold_study_counts": [1] * 5,
            }
            report = materialize_filtered_cache(
                source, destination, cohort, 768, allowed_source_extras={3: {"extra"}}
            )
            self.assertEqual(report["total_studies"], 5)
            self.assertEqual(report["prediction_recomputation"], False)
            selected_source = source / "fold_3/studies/s3.json"
            selected_target = destination / "fold_3/studies/s3.json"
            self.assertEqual(selected_source.read_bytes(), selected_target.read_bytes())
            self.assertEqual(len(json.loads(selected_target.read_text())["slices"]), 2)
            self.assertFalse((destination / "fold_3/studies/extra.json").exists())
            expected = {fold: {"s%d" % fold} for fold in range(5)}
            validate_complete_cache(destination, expected)

    def test_explicit_study_selection_does_not_expand_by_patient(self):
        metadata = pd.DataFrame([
            {"study_id": "selected", "patient_id": "same_patient", "sop_uid": "a"},
            {"study_id": "selected", "patient_id": "same_patient", "sop_uid": "b"},
            {"study_id": "extra_same_patient", "patient_id": "same_patient", "sop_uid": "c"},
        ])
        slices, studies = select_explicit_fold_studies(
            metadata, {"selected"}, {"same_patient"}, fold=0
        )
        self.assertEqual(set(studies["study_id"]), {"selected"})
        self.assertEqual(set(slices["sop_uid"]), {"a", "b"})

    def test_manifest_derived_explicit_validation_coverage_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, lists = [], []
            for fold in range(5):
                image = root / ("image_%d.png" % fold)
                image.touch()
                rows.append({
                    "split": "train", "patient_id": "p%d" % fold,
                    "study_id": "s%d" % fold, "image_path": str(image),
                })
                listing = root / ("fold_%d_val.txt" % fold)
                listing.write_text(str(image) + "\n")
                lists.append(listing)
            cohort = derive_explicit_validation_cohort(
                pd.DataFrame(rows), lists, expected_counts=(1, 1, 1, 1, 1), expected_total=5
            )
            self.assertEqual(cohort["per_fold_study_counts"], [1, 1, 1, 1, 1])
            self.assertEqual(cohort["total_unique_validation_studies"], 5)

    def _write_completed_cache(self, root, expected, omit=None, extra=None):
        omit = omit or set()
        extra = extra or {}
        for fold in range(5):
            studies_dir = root / ("fold_%d" % fold) / "studies"
            studies_dir.mkdir(parents=True)
            actual = set(expected[fold]) - set(omit)
            actual.update(extra.get(fold, set()))
            for study in actual:
                (studies_dir / (study + ".json")).write_text(
                    '{"fold":%d,"series_id":"%s"}' % (fold, study)
                )
            marker = root / ("fold_%d" % fold) / "COMPLETE.json"
            marker.write_text(json.dumps({
                "expected_study_ids": sorted(expected[fold]), "studies": len(actual),
            }))

    def test_completed_cache_with_extra_study_hard_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = {fold: {"s%d" % fold} for fold in range(5)}
            root = Path(directory)
            self._write_completed_cache(root, expected, extra={3: {"unexpected"}})
            with self.assertRaisesRegex(RuntimeError, "Unexpected cached Studies"):
                validate_complete_cache(root, expected)

    def test_completed_cache_missing_expected_study_hard_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = {fold: {"s%d" % fold} for fold in range(5)}
            root = Path(directory)
            self._write_completed_cache(root, expected, omit={"s4"})
            with self.assertRaisesRegex(RuntimeError, "Expected cached Studies missing"):
                validate_complete_cache(root, expected)

    def test_synthetic_path_and_fold_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dicom, annotations = root / "dicom", root / "annotations"
            manifest_rows, metadata_rows, assignment_rows, val_paths = [], [], [], []
            for fold in range(5):
                study, patient, sop = "s%d" % fold, "p%d" % fold, "u%d" % fold
                (dicom / study).mkdir(parents=True)
                (dicom / study / (sop + ".dcm")).touch()
                if fold == 0:
                    (annotations / study).mkdir(parents=True)
                    (annotations / study / (sop + ".json")).write_text("{}")
                image = root / ("image%d.png" % fold)
                image.touch()
                manifest_rows.append({"split": "train", "patient_id": patient, "study_id": study, "image_path": str(image)})
                metadata_rows.append({
                    "dicom_series.id": study, "dicom_series.PatientID": patient,
                    "dicom_series.SOPInstanceUID": sop, "SkullFracture": fold == 0,
                })
                assignment_rows.append({"patient_id": patient, "partition": "trainval", "validation_fold": fold})
                path = root / ("fold_%d_val.txt" % fold)
                path.write_text(str(image) + "\n")
                val_paths.append(path)
            (dicom / "unknown").mkdir()
            (dicom / "unknown" / "unknown_uid.dcm").touch()
            manifest = root / "manifest.csv"
            metadata = root / "metadata.pkl"
            assignments = root / "assignments.csv"
            pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
            pd.DataFrame(metadata_rows).to_pickle(metadata)
            pd.DataFrame(assignment_rows).to_csv(assignments, index=False)
            args = argparse.Namespace(
                metadata=metadata, dicom_root=dicom, annotation_root=annotations,
                manifest=manifest, assignments=assignments, val_lists=val_paths,
                output=root / "report.json",
            )
            report = run_coverage_audit(args)
            self.assertEqual(report["dataset"]["total_dicom_slices"], 6)
            self.assertEqual(report["dataset"]["json_backed_dicom_slices"], 1)
            self.assertEqual(report["dataset"]["metadata_backed_non_json_slices"], 4)
            self.assertEqual(report["dataset"]["metadata_unknown_dicom_slices"], 1)
            self.assertEqual(report["split_coverage"]["covered_development_studies"], 5)


if __name__ == "__main__":
    unittest.main()
