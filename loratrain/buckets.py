"""Approximate the bucket sd-scripts will put an image in.

sd-scripts keeps the source aspect ratio and fits each image into a bucket whose
sides are multiples of ``steps`` and stay between the min and max resolution. The
result is close enough to spot the two problems that actually ruin a dataset
before training starts: images that get upscaled into their bucket (soft detail,
soft eyes) and images that get cropped to a sliver.
"""

from __future__ import annotations

import math


def bucket_for(
    width: int,
    height: int,
    resolution: int = 1024,
    min_reso: int = 1024,
    max_reso: int = 1536,
    steps: int = 64,
) -> tuple[int, int]:
    """Return the (width, height) bucket for an image of the given size."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    aspect = width / height
    area = float(resolution) * float(resolution)

    bucket_w = int(round(math.sqrt(area * aspect) / steps) * steps)
    bucket_h = int(round(math.sqrt(area / aspect) / steps) * steps)

    bucket_w = max(steps, bucket_w)
    bucket_h = max(steps, bucket_h)

    bucket_w = min(max(bucket_w, min_reso), max_reso)
    bucket_h = min(max(bucket_h, min_reso), max_reso)

    return int(bucket_w), int(bucket_h)


def upscale_factor(width: int, height: int, bucket: tuple[int, int]) -> float:
    """How much the image is scaled to fill its bucket. Above 1.0 means upscaled."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    bucket_w, bucket_h = bucket
    return math.sqrt((bucket_w * bucket_h) / float(width * height))
