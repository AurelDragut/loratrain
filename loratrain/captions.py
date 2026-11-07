"""Caption helpers: normalising text and matching a trigger token."""

from __future__ import annotations

import re


def normalize(caption: str) -> str:
    """Collapse all whitespace runs, including newlines, into single spaces."""
    return " ".join(caption.split())


def contains_token(caption: str, token: str) -> bool:
    """True when *token* appears in *caption* as its own word.

    Substring matching would count ``mylora`` inside ``mylorax``, which is the
    kind of false positive that hides a broken dataset.
    """
    if not token:
        return False
    pattern = r"(?<![0-9A-Za-z_])" + re.escape(token) + r"(?![0-9A-Za-z_])"
    return re.search(pattern, caption, re.IGNORECASE) is not None
