"""Standalone helper to fetch the YuNet ONNX model into ./models/."""

from __future__ import annotations

import sys
from pathlib import Path

from detectors import find_yunet_path


def main() -> int:
    model_dir = Path(__file__).parent / "models"
    path = find_yunet_path(model_dir, download=True)
    if path is None:
        print("Download failed.", file=sys.stderr)
        return 1
    print(f"Model ready at {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
