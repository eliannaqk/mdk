#!/usr/bin/env python3
"""Compatibility wrapper for numbered pipeline script."""

from pathlib import Path
import runpy

MODULE_PATH = Path(__file__).with_name("download_structures.py")


def main() -> None:
    """Execute the canonical download_structures module if present."""
    if not MODULE_PATH.exists():
        raise FileNotFoundError(f"Expected {MODULE_PATH} to exist for execution")
    runpy.run_path(str(MODULE_PATH), run_name="__main__")


if __name__ == "__main__":
    main()
