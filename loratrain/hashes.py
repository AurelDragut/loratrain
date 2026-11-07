"""File digests and perceptual hashes for duplicate detection."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_digest(path: Path | str, chunk: int = 1 << 20) -> str:
    """SHA-256 of the file bytes: catches byte-identical copies."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def perceptual_available() -> bool:
    """True when Pillow can be imported, which dHash needs to decode pixels."""
    try:
        from PIL import Image  # noqa: F401
    except Exception:
        return False
    return True


def difference_hash(path: Path | str) -> int | None:
    """64-bit difference hash: catches rescales and re-encodes, not heavy crops.

    The image is reduced to 9x8 greyscale and each pixel compared to its right
    neighbour; the 64 comparison results become the bits.
    """
    try:
        from PIL import Image
    except Exception:
        return None

    try:
        with Image.open(path) as image:
            resampling = getattr(Image, "Resampling", Image)
            small = image.convert("L").resize((9, 8), resampling.LANCZOS)
            pixels = list(small.getdata())
    except Exception:
        return None

    if len(pixels) < 72:
        return None

    bits = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            left = pixels[offset + column]
            right = pixels[offset + column + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def hamming(left: int, right: int) -> int:
    """Number of differing bits between two hashes."""
    return bin(left ^ right).count("1")
