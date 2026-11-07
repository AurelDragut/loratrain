"""Scan a training folder: sizes, buckets, captions, findings."""

from __future__ import annotations

import collections
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .buckets import bucket_for, upscale_factor
from .captions import contains_token, normalize
from .imagesize import read_size

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jfif", ".tif", ".tiff"})


@dataclass
class ImageEntry:
    name: str
    width: int
    height: int
    bucket: tuple[int, int]
    upscale: float
    caption: str | None


@dataclass
class Finding:
    level: str  # "error", "warn" or "note"
    message: str


@dataclass
class Dataset:
    root: Path
    resolution: int
    entries: list[ImageEntry] = field(default_factory=list)
    unreadable: list[tuple[str, str]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.entries)

    def buckets(self) -> list[tuple[tuple[int, int], int]]:
        counts = collections.Counter(entry.bucket for entry in self.entries)
        return counts.most_common()

    def missing_captions(self) -> list[ImageEntry]:
        return [entry for entry in self.entries if entry.caption is None]

    def upscaled(self) -> list[ImageEntry]:
        return [entry for entry in self.entries if entry.upscale > 1.01]


def scan(
    root: Path | str,
    *,
    resolution: int,
    min_reso: int,
    max_reso: int,
    steps: int = 64,
    caption_ext: str = ".txt",
) -> Dataset:
    """Read every image in *root* and its sidecar caption."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    if caption_ext and not caption_ext.startswith("."):
        caption_ext = "." + caption_ext

    dataset = Dataset(root=root, resolution=resolution)

    paths = sorted(
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )

    for path in paths:
        size = read_size(path)
        if size is None:
            dataset.unreadable.append((path.name, "unknown format or unreadable"))
            continue

        width, height = size
        if width <= 0 or height <= 0:
            dataset.unreadable.append((path.name, "zero dimensions"))
            continue

        bucket = bucket_for(width, height, resolution, min_reso, max_reso, steps)
        caption_path = path.with_suffix(caption_ext)
        caption = None
        if caption_path.is_file():
            caption = normalize(caption_path.read_text(encoding="utf-8", errors="replace"))

        dataset.entries.append(
            ImageEntry(
                name=path.name,
                width=width,
                height=height,
                bucket=bucket,
                upscale=upscale_factor(width, height, bucket),
                caption=caption,
            )
        )

    return dataset


def report(
    dataset: Dataset,
    *,
    trigger: str | None = None,
    expected: tuple[int, int] | None = None,
) -> tuple[str, list[Finding]]:
    """Render the audit as text plus a list of findings."""
    findings: list[Finding] = []
    lines: list[str] = []

    lines.append(f"dataset    {dataset.root}")
    lines.append(f"images     {dataset.count}")

    if dataset.count == 0:
        findings.append(Finding("error", "no readable images found"))
        return "\n".join(lines), findings

    if dataset.unreadable:
        lines.append(f"unreadable {len(dataset.unreadable)}")
        findings.append(
            Finding(
                "warn",
                f"{len(dataset.unreadable)} file(s) could not be read and will be skipped",
            )
        )

    lines.append("")
    lines.append(f"buckets (approximate, target {dataset.resolution})")
    for (bucket_w, bucket_h), count in dataset.buckets():
        share = 100.0 * count / dataset.count
        lines.append(f"  {bucket_w} x {bucket_h:<5} {count:>4}  {share:5.1f}%")

    short_sides = [min(entry.width, entry.height) for entry in dataset.entries]
    lines.append("")
    lines.append(
        "short side  min {}, median {}, max {}".format(
            min(short_sides), int(statistics.median(short_sides)), max(short_sides)
        )
    )

    upscaled = dataset.upscaled()
    if upscaled:
        share = 100.0 * len(upscaled) / dataset.count
        lines.append(f"upscaled    {len(upscaled)} ({share:.0f}%)")
        findings.append(
            Finding(
                "warn",
                f"{len(upscaled)} image(s) ({share:.0f}%) are smaller than their bucket and "
                "get upscaled; fine detail is learned from a soft version",
            )
        )
        if share > 30:
            findings.append(
                Finding(
                    "warn",
                    "most of the dataset is upscaled: lower --min-bucket to match the "
                    "smallest images, or re-source them at the preset resolution",
                )
            )

    missing = dataset.missing_captions()
    if missing:
        lines.append(f"no caption  {len(missing)}")
        findings.append(
            Finding(
                "warn",
                f"{len(missing)} image(s) have no caption file; they teach nothing and "
                "dilute the trigger",
            )
        )

    if trigger:
        with_trigger = sum(
            1
            for entry in dataset.entries
            if entry.caption and contains_token(entry.caption, trigger)
        )
        covered = 100.0 * with_trigger / dataset.count
        lines.append(
            f"trigger     '{trigger}' in {with_trigger}/{dataset.count} captions ({covered:.0f}%)"
        )
        if covered < 95.0:
            findings.append(
                Finding(
                    "warn",
                    f"trigger '{trigger}' appears in only {covered:.0f}% of captions; "
                    "inconsistent captions make a concept render wrong instead of absent",
                )
            )
    else:
        findings.append(
            Finding("note", "no --trigger given, so caption consistency was not checked")
        )

    captions = collections.Counter(
        entry.caption for entry in dataset.entries if entry.caption is not None
    )
    repeated = [caption for caption, count in captions.items() if count > 1]
    if repeated:
        lines.append(f"repeated    {len(repeated)} caption(s) shared by several images")
        findings.append(
            Finding(
                "note",
                f"{len(repeated)} caption(s) are used by more than one image; harmless for "
                "filler, suspicious for the subject",
            )
        )

    short_captions = [
        entry for entry in dataset.entries if entry.caption is not None and len(entry.caption.split()) < 3
    ]
    if short_captions:
        findings.append(
            Finding("note", f"{len(short_captions)} caption(s) are under three words")
        )

    extreme = sum(
        1
        for entry in dataset.entries
        if entry.width / entry.height > 3 or entry.height / entry.width > 3
    )
    if extreme:
        findings.append(
            Finding(
                "note",
                f"{extreme} image(s) are past a 3:1 aspect ratio and will be cropped hard",
            )
        )

    if expected is not None:
        low, high = expected
        if dataset.count < low:
            findings.append(
                Finding(
                    "warn",
                    f"{dataset.count} images is below the {low}-{high} this preset expects",
                )
            )
        elif dataset.count > high:
            findings.append(
                Finding(
                    "note",
                    f"{dataset.count} images is above the {low}-{high} this preset expects; "
                    "training will be slower and the rank may need raising",
                )
            )

    return "\n".join(lines), findings
