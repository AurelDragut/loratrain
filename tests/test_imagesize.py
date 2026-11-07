import struct

from loratrain.imagesize import _size_from_bytes


def _png(width, height):
    header = b"\x89PNG\r\n\x1a\n"
    ihdr = b"IHDR" + struct.pack(">II", width, height) + b"\x08\x06\x00\x00\x00"
    return header + struct.pack(">I", 13) + ihdr + b"\x00\x00\x00\x00"


def _jpeg(width, height):
    soi = b"\xff\xd8"
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof0 = (
        b"\xff\xc0"
        + struct.pack(">H", 11)
        + b"\x08"
        + struct.pack(">HH", height, width)
        + b"\x01\x01\x11\x00"
    )
    return soi + app0 + sof0


def test_png_size():
    assert _size_from_bytes(_png(640, 480)) == (640, 480)


def test_jpeg_size():
    assert _size_from_bytes(_jpeg(1920, 1080)) == (1920, 1080)


def test_unknown_format_returns_none():
    assert _size_from_bytes(b"nope" * 10) is None
