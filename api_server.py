"""ASGI entrypoint for the FastAPI backend (keeps command short and PYTHONPATH configured)."""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interview_guide.api.app import create_app

app = create_app()
