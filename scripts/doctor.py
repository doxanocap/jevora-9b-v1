"""Inspect the local Jevora release environment without loading a model."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    valid = sys.version_info[:2] == (3, 13) and sys.prefix != sys.base_prefix
    print(f"Jevora: {project}")
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.system()} / {platform.machine()}")
    print(f"Virtual environment: {'OK' if valid else 'check Python 3.13 and .venv'}")
    print(f"HF_TOKEN: {'set' if os.environ.get('HF_TOKEN') else 'not set (optional)'}")
    print("Model package: pass its unpacked path as PACKAGE=... to Make targets.")
    print("No model loaded and no network request made.")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
