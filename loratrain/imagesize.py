"""Image dimensions without a hard Pillow dependency.

Pillow is used when it is installed, because it is the only reliable reader for
every format. Without it, PNG, JPEG, WebP and BMP headers are parsed directly,
which covers what a LoRA dataset realistically contains.
"""

from __future__ import annotations

from pathlib import Path

try:  # most machines that run sd-scripts already have it
    from PIL import Image
except Exception:  # pragma: no cover - only without Pillow
    Image = None


def read_size(path: Path) -> tuple[int, int] | None:
    """Return (width, height) for *path*, or None when the format is unknown."""
    path = Path(path)

    if Image is not None:
        try:
            with Image.open(path) as im:
                return (int(im.size[0]), int(im.size[1]))
        except Exception:
            pass

    try:
        with open(path, "rb") as handle:
            size = _size_from_bytes(handle.read(65536))
            if size is not None:
                return size
            handle.seek(0)
            return _size_from_bytes(handle.read())
    except OSError:
        return None


def _size_from_bytes(data: bytes) -> tuple[int, int] | None:
    if len(data) < 26:
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return _png(data)
    if data[:2] == b"\xff\xd8":
        return _jpeg(data)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _webp(data)
    if data[:2] == b"BM":
        return _bmp(data)
    return None


def _png(data: bytes) -> tuple[int, int] | None:
    if data[12:16] != b"IHDR":
        return None
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    if width and height:
        return (width, height)
    return None


def _jpeg(data: bytes) -> tuple[int, int] | None:
    index = 2
    length = len(data)
    while index + 3 < length:
        if data[index] != 0xFF:
            index += 1
            continue

        marker = data[index + 1]

        # padding, standalone markers, SOI/EOI/restart markers
        if marker == 0xFF:
            index += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            index += 2
            continue
        if marker in (0xD9, 0xDA):
            return None

        if index + 4 > length:
            return None
        segment = int.from_bytes(data[index + 2:index + 4], "big")
        if segment < 2:
            return None

        # SOF0..SOF15 minus the non-frame markers
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if index + 9 > length:
                return None
            height = int.from_bytes(data[index + 5:index + 7], "big")
            width = int.from_bytes(data[index + 7:index + 9], "big")
            if width and height:
                return (width, height)
            return None

        index += 2 + segment
    return None


def _webp(data: bytes) -> tuple[int, int] | None:
    chunk = data[12:16]

    if chunk == b"VP8 ":
        if len(data) < 30 or data[23:26] != b"\x9d\x01\x2a":
            return None
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        if width and height:
            return (width, height)
        return None

    if chunk == b"VP8L":
        if len(data) < 25 or data[20] != 0x2F:
            return None
        bits = int.from_bytes(data[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return (width, height)

    if chunk == b"VP8X":
        if len(data) < 30:
            return None
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return (width, height)

    return None


def _bmp(data: bytes) -> tuple[int, int] | None:
    if len(data) < 26:
        return None
    header = int.from_bytes(data[14:18], "little")
    if header < 40:
        return None
    width = int.from_bytes(data[18:22], "little", signed=True)
    height = int.from_bytes(data[22:26], "little", signed=True)
    width = abs(width)
    height = abs(height)
    if width and height:
        return (width, height)
    return None
