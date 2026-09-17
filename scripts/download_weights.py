"""Download and verify the official Ultralytics YOLO26s COCO checkpoint."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "weights" / "yolo26s.pt"


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if not DESTINATION.is_file():
        downloaded = Path(YOLO("yolo26s.pt").ckpt_path).resolve()
        if downloaded != DESTINATION.resolve():
            shutil.copy2(downloaded, DESTINATION)
    YOLO(str(DESTINATION), task="detect")  # validates checkpoint readability
    digest = hashlib.sha256(DESTINATION.read_bytes()).hexdigest()
    print(f"{DESTINATION}\nsha256={digest}")


if __name__ == "__main__":
    main()
