"""Validate (never modify) the official execution environment."""
from __future__ import annotations
import importlib.metadata as metadata
import platform, sys

REQUIRED = {"numpy": (2, 0), "pydicom": (3, 0), "torch": (2, 10), "ultralytics": (8, 3, 240), "pandas": None, "scikit-learn": None, "joblib": None, "PyYAML": None}


def main() -> int:
    errors: list[str] = []
    print(f"Python: {platform.python_version()}")
    if sys.version_info[:2] != (3, 12):
        errors.append("Python must be >=3.12,<3.13")
    for package, minimum in REQUIRED.items():
        try:
            version = metadata.version(package)
            print(f"{package}: {version}")
            if minimum:
                numeric = tuple(int(x) for x in version.split("+")[0].split(".")[:len(minimum)])
                if numeric < minimum:
                    errors.append(f"{package}>={'.'.join(map(str, minimum))} required")
        except metadata.PackageNotFoundError:
            errors.append(f"Missing package: {package}")
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Torch CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    except ImportError:
        pass
    if errors:
        print("Environment validation FAILED:\n- " + "\n- ".join(errors))
        return 1
    print("Environment validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
