"""Import pydicom, falling back to the in-repo source checkout.

This workspace contains the pydicom sources under src/ but the package is
not installed (offline environment, no flit_core to build it). A minimal
dist-info shim in .pymeta/ satisfies pydicom's importlib.metadata version
lookup, so adding both directories to sys.path is enough.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap() -> None:
    try:
        import pydicom  # noqa: F401
        return
    except Exception:
        pass
    root = Path(__file__).resolve().parent.parent
    for extra in (root / ".pymeta", root / "src"):
        if extra.is_dir() and str(extra) not in sys.path:
            sys.path.insert(0, str(extra))


_bootstrap()

import pydicom  # noqa: E402,F401

