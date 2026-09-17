"""Create an immutable Tier1-only box-validation view for one outer fold."""
from __future__ import annotations

import argparse
from pathlib import Path

from fracture.data.weak_study import fold_partition, load_study_catalog, write_tier1_validation_view
from fracture.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    catalog = load_study_catalog(config)
    _, validation = fold_partition(
        catalog, config["split"].get("path", "splits/folds.json"), args.fold
    )
    dataset_root = Path(config["data"]["output_root"]) / config["data"]["annotation_version"]
    print(write_tier1_validation_view(
        manifest_path=dataset_root / "manifest.csv",
        validation_studies=validation,
        output_dir=args.output_dir,
        fold=args.fold,
    ))


if __name__ == "__main__":
    main()
