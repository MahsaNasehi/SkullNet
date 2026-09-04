"""Validate (never modify) the official execution environment."""
from __future__ import annotations
import importlib.metadata as metadata
import platform, re, sys

REQUIRED = {
    "numpy": (2, 0),
    "pydicom": (3, 0, 1),
    "torch": (2, 10),
    "ultralytics": (8, 3, 240),
    "pandas": None,
    "scikit-learn": None,
    "joblib": None,
    "PyYAML": None,
}
UPPER_EXCLUSIVE = {"numpy": (2, 3), "torch": (3, 0), "ultralytics": (9, 0)}


def _numeric_version(version: str, length: int) -> tuple[int, ...]:
    values = [int(value) for value in re.findall(r"\d+", version.split("+")[0])[:length]]
    return tuple(values + [0] * (length - len(values)))


def main() -> int:
    errors: list[str] = []
    print(f"Python: {platform.python_version()}")
    if not (sys.version_info >= (3, 12) and sys.version_info < (3, 13)):
        errors.append("Official evaluator requires Python >=3.12,<3.13")
    for package, minimum in REQUIRED.items():
        try:
            version = metadata.version(package)
            print(f"{package}: {version}")
            if minimum:
                numeric = _numeric_version(version, len(minimum))
                if numeric < minimum:
                    errors.append(f"{package}>={'.'.join(map(str, minimum))} required")
            upper = UPPER_EXCLUSIVE.get(package)
            if upper and _numeric_version(version, len(upper)) >= upper:
                errors.append(f"{package} must be <{'.'.join(map(str, upper))}")
        except metadata.PackageNotFoundError:
            errors.append(f"Missing package: {package}")
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Torch CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
        if torch.cuda.is_available():
            print(f"GPU capability: {torch.cuda.get_device_capability(0)}")
            print(f"Compiled architectures: {torch.cuda.get_arch_list()}")
            sample = torch.ones((64, 64), device="cuda")
            _ = sample @ sample
            torch.cuda.synchronize()
            print("CUDA computation: passed")
    except ImportError:
        pass
    if errors:
        print("Environment validation FAILED:\n- " + "\n- ".join(errors))
        return 1
    print("Environment validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
