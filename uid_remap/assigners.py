"""Helpers for building old->new UID assign functions."""

from __future__ import annotations

import hashlib
from typing import Callable

# 2.25.<decimal> is the ISO UUID/ULID-style root recommended by DICOM for
# locally generated UIDs; a 128-bit decimal arc stays well under 64 chars.
DEFAULT_ROOT = "2.25"


def hash_assigner(root: str = DEFAULT_ROOT, salt: str = "") -> Callable[[str], str]:
    """Deterministic assigner: new UID derives only from (root, salt, old).

    Same inputs always produce the same mapping, so it is safe to reuse
    across batches; combine with a persisted mapping for stability when
    the salt changes.
    """
    def assign(old: str) -> str:
        digest = hashlib.sha256(f"{salt}{old}".encode("utf-8")).digest()
        number = int.from_bytes(digest[:16], "big")
        return f"{root}.{number}"
    return assign


def chained_assigner(mapping: dict[str, str],
                     fallback: Callable[[str], str]) -> Callable[[str], str]:
    """Assigner that prefers a persisted mapping, falling back otherwise."""
    def assign(old: str) -> str:
        if old in mapping:
            return mapping[old]
        return fallback(old)
    return assign

