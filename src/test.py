"""Gradio viewer for SkullNet DICOM slices, annotations, and pipeline metadata."""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fracture.data.annotations import resolve_annotation, xywh_to_yolo
from fracture.data.dicom import SliceRecord, load_study
from fracture.data.metadata import load_metadata
from fracture.data.windows import bone_window, make_input
from fracture.utils.config import load_config, require_path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/fracture_25d_p2_pretrained.yaml"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)


def _aggregate_25d_display(tensor: np.ndarray, display_mode: str) -> np.ndarray:
    """Collapse 2.5D channels into a view for human review (not model input)."""
    if display_mode == "rgb":
        return tensor
    if display_mode == "max":
        return np.max(tensor, axis=2).astype(np.uint8)
    if display_mode == "min":
        return np.min(tensor, axis=2).astype(np.uint8)
    if display_mode == "min_clahe":
        minip = np.min(tensor, axis=2).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(minip)
    if display_mode == "mean":
        return np.mean(tensor, axis=2).round().astype(np.uint8)
    if display_mode == "center":
        return tensor[..., 1]
    if display_mode == "overlay":
        return _overlay_neighbor_diff(tensor)
    raise ValueError(f"Unsupported display_mode: {display_mode}")


def _overlay_neighbor_diff(tensor: np.ndarray) -> np.ndarray:
    """Grayscale center slice with color hints for inter-slice changes."""
    z_prev = tensor[..., 0].astype(np.int16)
    z_center = tensor[..., 1].astype(np.int16)
    z_next = tensor[..., 2].astype(np.int16)
    base = cv2.cvtColor(tensor[..., 1], cv2.COLOR_GRAY2RGB)
    overlay = np.zeros_like(base)
    # Brighter in a neighbor than center (structure shift between slices).
    overlay[..., 0] = np.clip(z_prev - z_center, 0, 255).astype(np.uint8)
    overlay[..., 2] = np.clip(z_next - z_center, 0, 255).astype(np.uint8)
    # Darker center vs neighbors (fracture-like lucent lines).
    overlay[..., 1] = np.clip((z_prev + z_next) // 2 - z_center, 0, 255).astype(np.uint8)
    return cv2.addWeighted(base, 0.8, overlay, 0.55, 0)


def _draw_boxes(image: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    for index, (x, y, width, height) in enumerate(boxes, start=1):
        x0, y0 = int(round(x)), int(round(y))
        x1, y1 = int(round(x + width)), int(round(y + height))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 220, 80), 2)
        cv2.putText(
            canvas,
            f"box {index}",
            (x0, max(16, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 80),
            1,
            cv2.LINE_AA,
        )
    return canvas


class DatasetExplorer:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path.resolve()
        self.cfg = load_config(self.config_path)
        self.data = self.cfg["data"]
        self.prep = self.cfg["preprocessing"]
        self.dicom_root = require_path(self.cfg, "data", "dicom_root")
        self.annotation_root = require_path(self.cfg, "data", "annotation_root")
        corrected = self.data.get("corrected_annotation_root")
        self.corrected_root = Path(corrected).expanduser().resolve() if corrected else None
        self.metadata_path = require_path(self.cfg, "data", "metadata_path")
        self.series_col = self.data["series_id_column"]
        self.sop_col = self.data["sop_uid_column"]
        self.patient_col = self.data["patient_id_column"]
        self.fracture_col = self.data["fracture_label_column"]
        self.studies = sorted(path.name for path in self.dicom_root.iterdir() if path.is_dir())
        self._metadata_frame = None
        self._metadata_index: dict[tuple[str, str], dict[str, Any]] = {}
        self._positive_studies: list[str] | None = None

    @property
    def positive_studies(self) -> list[str]:
        if self._positive_studies is None:
            frame = self.metadata_frame
            grouped = frame.groupby(self.series_col)[self.fracture_col].max()
            self._positive_studies = sorted(
                str(study_id) for study_id, label in grouped.items() if bool(label)
            )
        return self._positive_studies

    def studies_for_filter(self, only_positive: bool) -> list[str]:
        return self.positive_studies if only_positive else self.studies

    @property
    def metadata_frame(self):
        if self._metadata_frame is None:
            self._metadata_frame = load_metadata(self.metadata_path)
            keep_cols = [
                self.series_col,
                self.sop_col,
                self.patient_col,
                self.fracture_col,
            ]
            optional_cols = [
                self.data.get("annotation_path_column"),
                "dicom_series.PixelSpacing0",
                "dicom_series.PixelSpacing1",
                "dicom_series.SliceThickness",
                "dicom_series.Rows",
                "dicom_series.Columns",
                "MidlineShiftMM",
                "triage_class",
            ]
            columns = [col for col in keep_cols + optional_cols if col and col in self._metadata_frame.columns]
            for _, row in self._metadata_frame[columns].iterrows():
                key = (str(row[self.series_col]), str(row[self.sop_col]))
                self._metadata_index[key] = {
                    col: (row[col].item() if hasattr(row[col], "item") else row[col])
                    for col in columns
                }
        return self._metadata_frame

    def metadata_for_slice(self, study_id: str, sop_uid: str) -> dict[str, Any] | None:
        self.metadata_frame
        return self._metadata_index.get((study_id, sop_uid))

    @lru_cache(maxsize=16)
    def study_records(self, study_id: str) -> tuple[SliceRecord, ...]:
        return tuple(load_study(self.dicom_root / study_id))

    def study_slice_count(self, study_id: str) -> int:
        return len(self.study_records(study_id))

    def _study_summary(
        self,
        study_id: str,
        records: tuple[SliceRecord, ...],
        study_fracture: bool | None,
    ) -> dict[str, Any]:
        annotated_slices = 0
        fracture_boxes = 0
        for item in records:
            item_annotation = resolve_annotation(
                self.annotation_root,
                self.corrected_root,
                study_id,
                item.sop_uid,
                image_shape=item.shape,
            )
            if item_annotation is None:
                continue
            annotated_slices += 1
            fracture_boxes += len(item_annotation.boxes)
        return {
            "skull_fracture_label": study_fracture,
            "annotated_slices_in_study": annotated_slices,
            "fracture_boxes_in_study": fracture_boxes,
        }

    def render_slice(
        self,
        study_id: str,
        slice_index: int,
        input_mode: str,
        show_boxes: bool,
        display_mode: str = "rgb",
    ) -> tuple[np.ndarray, str]:
        records = self.study_records(study_id)
        if not records:
            raise ValueError(f"No slices found for study {study_id}")

        slice_index = int(np.clip(slice_index, 0, len(records) - 1))
        record = records[slice_index]
        windows = [
            bone_window(
                item.hu,
                level=float(self.prep["window_level"]),
                width=float(self.prep["window_width"]),
                monochrome1=(item.photometric_interpretation == "MONOCHROME1"),
            )
            for item in records
        ]
        tensor = make_input(
            windows,
            slice_index,
            mode=input_mode,
            boundary_mode=str(self.prep.get("boundary_mode", "repeat")),
        )
        if input_mode == "2.5d":
            image = _aggregate_25d_display(tensor, display_mode)
        else:
            image = tensor[..., 0]

        annotation = resolve_annotation(
            self.annotation_root,
            self.corrected_root,
            study_id,
            record.sop_uid,
            image_shape=record.shape,
        )
        boxes = [(box.x, box.y, box.width, box.height) for box in annotation.boxes] if annotation else []
        if show_boxes and boxes:
            image = _draw_boxes(image, boxes)

        hu = record.hu
        metadata_row = self.metadata_for_slice(study_id, record.sop_uid)
        study_fracture = None
        if metadata_row is not None:
            study_fracture = bool(metadata_row.get(self.fracture_col))

        payload: dict[str, Any] = {
            "config": str(self.config_path),
            "study_id": study_id,
            "slice_index": slice_index,
            "total_slices": len(records),
            "sop_uid": record.sop_uid,
            "dicom_path": str(record.path),
            "instance_number": record.instance_number,
            "shape_hw": list(record.shape),
            "pixel_spacing": list(record.pixel_spacing) if record.pixel_spacing else None,
            "slice_thickness_mm": record.slice_thickness,
            "physical_position": record.physical_position,
            "image_orientation": list(record.image_orientation) if record.image_orientation else None,
            "image_position": list(record.image_position) if record.image_position else None,
            "transfer_syntax_uid": record.transfer_syntax_uid,
            "photometric_interpretation": record.photometric_interpretation,
            "hu_stats": {
                "min": float(hu.min()),
                "max": float(hu.max()),
                "mean": float(hu.mean()),
                "std": float(hu.std()),
            },
            "preprocessing": {
                "window_level": float(self.prep["window_level"]),
                "window_width": float(self.prep["window_width"]),
                "input_mode": input_mode,
                "display_mode": display_mode if input_mode == "2.5d" else "grayscale",
                "boundary_mode": self.prep.get("boundary_mode", "repeat"),
                "image_size_training": int(self.prep.get("image_size", 0)) or None,
            },
            "annotation": None,
            "metadata": metadata_row,
            "study_level": self._study_summary(study_id, records, study_fracture),
        }

        if annotation is not None:
            height, width = record.shape
            payload["annotation"] = {
                "provenance": annotation.provenance,
                "path": str(annotation.source_path),
                "num_boxes": len(annotation.boxes),
                "boxes_xywh": [[box.x, box.y, box.width, box.height] for box in annotation.boxes],
                "boxes_yolo": [
                    list(xywh_to_yolo(box, width, height))
                    for box in annotation.boxes
                ],
            }

        return image, _to_json(payload)


def _on_filter_change(
    explorer: DatasetExplorer,
    only_positive: bool,
    input_mode: str,
    display_mode: str,
    show_boxes: bool,
):
    import gradio as gr

    choices = explorer.studies_for_filter(only_positive)
    if not choices:
        raise gr.Error("No positive studies found in metadata.")
    study_id = choices[0]
    count = explorer.study_slice_count(study_id)
    midpoint = count // 2
    image, info_json = explorer.render_slice(
        study_id, midpoint, input_mode, show_boxes, display_mode
    )
    return (
        gr.Dropdown(choices=choices, value=study_id),
        gr.Slider(minimum=0, maximum=max(0, count - 1), value=midpoint, step=1),
        image,
        info_json,
    )


def _on_study_change(
    explorer: DatasetExplorer,
    study_id: str,
    input_mode: str,
    display_mode: str,
    show_boxes: bool,
):
    import gradio as gr

    count = explorer.study_slice_count(study_id)
    midpoint = count // 2
    image, info_json = explorer.render_slice(
        study_id, midpoint, input_mode, show_boxes, display_mode
    )
    return (
        gr.Slider(minimum=0, maximum=max(0, count - 1), value=midpoint, step=1),
        image,
        info_json,
    )


def _on_view(
    explorer: DatasetExplorer,
    study_id: str,
    slice_index: int,
    input_mode: str,
    display_mode: str,
    show_boxes: bool,
):
    image, info_json = explorer.render_slice(
        study_id, slice_index, input_mode, show_boxes, display_mode
    )
    return image, info_json


def build_demo(explorer: DatasetExplorer):
    import gradio as gr

    default_study = explorer.studies[0]
    default_count = explorer.study_slice_count(default_study)
    positive_count = len(explorer.positive_studies)

    with gr.Blocks(title="SkullNet Dataset Explorer") as demo:
        gr.Markdown(
            "# SkullNet Dataset Explorer\n"
            "Browse DICOM studies with the same bone-window and 2.5D preprocessing used in training."
        )

        with gr.Row():
            only_positive = gr.Checkbox(
                value=False,
                label=f"Only positive studies ({positive_count} with skull fracture)",
            )
            study_dropdown = gr.Dropdown(
                choices=explorer.studies,
                value=default_study,
                label="Study ID",
                filterable=True,
            )
            slice_slider = gr.Slider(
                minimum=0,
                maximum=max(0, default_count - 1),
                value=default_count // 2,
                step=1,
                label="Slice index",
            )

        with gr.Row():
            input_mode = gr.Radio(
                choices=["single", "2.5d"],
                value=str(explorer.prep.get("input_mode", "2.5d")),
                label="Input mode (model pipeline)",
            )
            display_mode = gr.Radio(
                choices=[
                    ("MinIP (recommended)", "min"),
                    ("MinIP + CLAHE", "min_clahe"),
                    ("Grayscale overlay (center + depth hints)", "overlay"),
                    ("RGB (z-1, z, z+1)", "rgb"),
                    ("Grayscale max (MIP)", "max"),
                    ("Grayscale mean", "mean"),
                    ("Center slice only (z)", "center"),
                ],
                value="min",
                label="2.5D display (visualization only)",
            )
            show_boxes = gr.Checkbox(value=True, label="Draw fracture boxes")

        with gr.Row():
            image_output = gr.Image(label="Slice preview", type="numpy")
            json_output = gr.Code(label="Slice metadata (JSON)", language="json")

        only_positive.change(
            fn=lambda checked, mode, disp, boxes: _on_filter_change(
                explorer, checked, mode, disp, boxes
            ),
            inputs=[only_positive, input_mode, display_mode, show_boxes],
            outputs=[study_dropdown, slice_slider, image_output, json_output],
        )

        study_dropdown.change(
            fn=lambda study_id, mode, disp, boxes: _on_study_change(
                explorer, study_id, mode, disp, boxes
            ),
            inputs=[study_dropdown, input_mode, display_mode, show_boxes],
            outputs=[slice_slider, image_output, json_output],
        )

        for component in (slice_slider, input_mode, display_mode, show_boxes):
            component.change(
                fn=lambda study_id, slice_index, mode, disp, boxes: _on_view(
                    explorer, study_id, slice_index, mode, disp, boxes
                ),
                inputs=[study_dropdown, slice_slider, input_mode, display_mode, show_boxes],
                outputs=[image_output, json_output],
            )

        demo.load(
            fn=lambda study_id, slice_index, mode, disp, boxes: _on_view(
                explorer, study_id, slice_index, mode, disp, boxes
            ),
            inputs=[study_dropdown, slice_slider, input_mode, display_mode, show_boxes],
            outputs=[image_output, json_output],
        )

    return demo


def main() -> None:
    import gradio  # noqa: F401

    parser = argparse.ArgumentParser(description="Launch Gradio dataset explorer for SkullNet.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to fracture config YAML (default: configs/fracture_25d_p2_pretrained.yaml)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    explorer = DatasetExplorer(config_path)
    demo = build_demo(explorer)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
