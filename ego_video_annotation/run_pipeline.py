#!/usr/bin/env python3
"""CLI entry point kept at repository root for the assessment requirement."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from egoanno.pipeline import main  # noqa: E402


if __name__ == "__main__":
    main()
