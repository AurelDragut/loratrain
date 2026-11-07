"""Caption a dataset with a vision model, and check it while doing so.

The order matters: duplicates and size are decided before a single model call, so a
broken dataset does not cost an hour of inference. Captions are only written when
asked for, and a copy of the model replies is cached by file digest so a re-run
costs nothing until an image changes.
"""

from __future__ import annotations

import concurrent.futures
import json
from dataclasses import dataclass, field
from pathlib import Path

from .attributes import deviations, vote
from .dataset import list_images
from .dedupe import DuplicateReport, find_duplicates
from .hashes import file_digest
from .imagesize import read_size
from .vlm import VlmConfig, VlmError, caption_image, resolve_flavor

DEFAULT_PROMPT = (
    "Label this training image for a LoRA dataset. Describe only what is visibly "
    "present; do not guess at identity, location or context.\n\n"
    "Return one JSON object and nothing else, with these keys:\n"
    '  "caption": one sentence under 40 words describing the subject and the framing,\n'
    '  "attributes": an object of short lowercase values for the subject only, using '
    '"hair_color", "hair_style", "eye_color", "skin_tone", "expression", "pose" '
    "and \"framing\" when they are visible,\n"
    '  "tags": a list of short lowercase tags for clothing, setting and notable details.\n\n'
    "Do not write 'image', 'photo' or 'picture' unless the framing itself is the point."
)


@dataclass
class CaptionRequest:
    dataset: Path
    trigger: str = ""
    min_side: int = 1024
    style: str = "sentence"
    subject: str = "character"
    duplicates_mode: str = "skip"  # skip | keep | only
    near_threshold: int = 8
    workers: int = 1
    overwrite: bool = False
    refresh: bool = False
    write: bool = False
    limit: int | None = None
    caption_ext: str = ".txt"
    prompt: str = DEFAULT_PROMPT
    cache_path: Path | None = None
    vlm: VlmConfig | None = None


@dataclass
class CaptionItem:
    path: Path
    width: int = 0
    height: int = 0
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)
    caption: str = ""
    deviations: dict[str, str] = field(default_factory=dict)
    reason: str | None = None  # set when the image was not captioned


@dataclass
class CaptionReport:
    dataset: Path
    min_side: int = 0
    flavor: str = ""
    items: list[CaptionItem] = field(default_factory=list)
    duplicates: DuplicateReport = field(default_factory=DuplicateReport)
    small: list[CaptionItem] = field(default_factory=list)
    written: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)
    skipped_duplicates: list[Path] = field(default_factory=list)
    dominant: dict[str, str] = field(default_factory=dict)


def assemble(
    trigger: str,
    summary: str,
    tags: list[str],
    style: str,
    deviating: dict[str, str],
) -> str:
    """Build the caption text from the model's parts.

    The trigger comes first (it is ``keep_tokens = 1`` in the config). Discrepancies
    are appended as explicit ``attribute: value`` pairs so the deviation is bound to
    words instead of being absorbed into the trigger token.
    """
    parts: list[str] = []
    if trigger.strip():
        parts.append(trigger.strip())

    if style in ("sentence", "both") and summary.strip():
        parts.append(summary.strip())

    if style in ("tags", "both") and tags:
        parts.append(", ".join(tags))
    elif style == "tags" and not tags and summary.strip():
        parts.append(summary.strip())

    text = ", ".join(part for part in parts if part)

    if deviating:
        note = ", ".join(f"{key}: {value}" for key, value in sorted(deviating.items()))
        text = f"{text}, {note}" if text else note

    return text


def _clean_tags(value) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    return []


def load_cache(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    items = data.get("items")
    return items if isinstance(items, dict) else {}


def save_cache(path: Path, entries: dict) -> None:
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "items": entries}
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def run(request: CaptionRequest) -> CaptionReport:
    root = Path(request.dataset)
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    if request.caption_ext and not request.caption_ext.startswith("."):
        request.caption_ext = "." + request.caption_ext

    paths = list_images(root)
    report = CaptionReport(dataset=root, min_side=request.min_side)
    report.duplicates = find_duplicates(paths, threshold=request.near_threshold)

    skip: set[Path] = set()
    if request.duplicates_mode == "skip":
        skip = report.duplicates.duplicate_paths()
    report.skipped_duplicates = sorted(skip)

    for path in paths:
        size = read_size(path)
        width, height = size if size else (0, 0)
        item = CaptionItem(path=path, width=width, height=height)
        if path in skip:
            item.reason = "duplicate"
        elif request.min_side and width and height and min(width, height) < request.min_side:
            report.small.append(item)
        report.items.append(item)

    if request.duplicates_mode == "only":
        return report

    pending = [item for item in report.items if item.reason is None]
    if request.limit is not None and request.limit >= 0:
        for item in pending[request.limit:]:
            item.reason = "limit"
        pending = pending[: request.limit]

    if request.vlm is None:
        for item in pending:
            item.reason = "no VLM"
        return report
    if not pending:
        return report

    report.flavor = resolve_flavor(request.vlm)

    cache_path = Path(request.cache_path) if request.cache_path else root / ".loratrain-cache.json"
    cache = load_cache(cache_path)

    def work(item: CaptionItem):
        digest = ""
        try:
            digest = file_digest(item.path)
        except OSError:
            pass

        if digest and not request.refresh and digest in cache:
            return item, digest, cache[digest], None

        try:
            data = caption_image(item.path, request.prompt, request.vlm)
        except (VlmError, OSError) as exc:
            return item, digest, None, str(exc)
        return item, digest, data, None

    def apply(item: CaptionItem, digest: str, data: dict) -> None:
        item.summary = str(data.get("caption") or "").strip()

        attributes = data.get("attributes")
        if not isinstance(attributes, dict):
            attributes = {}

        raw_tags = data.get("tags")
        if raw_tags is None:
            raw_tags = attributes.get("tags")
        item.tags = _clean_tags(raw_tags)

        item.attributes = {
            str(key).strip().lower(): str(value).strip()
            for key, value in attributes.items()
            if isinstance(value, (str, int, float)) and str(value).strip()
        }
        if digest:
            cache[digest] = {
                "caption": item.summary,
                "tags": item.tags,
                "attributes": item.attributes,
            }

    # One call first: a wrong URL or model name should fail once, not once per image.
    first = pending[0]
    item, digest, data, error = work(first)
    if error is not None:
        report.failed.append((first.path, error))
        first.reason = "failed"
        return report
    apply(first, digest, data)

    rest = pending[1:]
    if rest:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, request.workers)) as pool:
            for item, digest, data, error in pool.map(work, rest):
                if error is not None:
                    report.failed.append((item.path, error))
                    item.reason = "failed"
                else:
                    apply(item, digest, data)

    save_cache(cache_path, cache)

    if request.subject == "character":
        captioned = {
            str(item.path): item.attributes for item in report.items if item.attributes
        }
        report.dominant = vote(captioned, min_share=0.8, min_items=3)
        for item in report.items:
            if item.attributes:
                item.deviations = deviations(item.attributes, report.dominant)

    for item in report.items:
        if item.reason is not None:
            continue
        item.caption = assemble(
            request.trigger, item.summary, item.tags, request.style, item.deviations
        )

    if request.write:
        for item in report.items:
            if item.reason is not None or not item.caption:
                continue
            target = item.path.with_suffix(request.caption_ext)
            if target.exists() and not request.overwrite:
                continue
            target.write_text(item.caption + "\n", encoding="utf-8")
            report.written.append(target)

    return report


def render_report(report: CaptionReport, preview: int = 5) -> str:
    lines: list[str] = []
    lines.append(f"dataset      {report.dataset}")
    lines.append(f"images       {len(report.items)}")
    if report.flavor:
        lines.append(f"endpoint     {report.flavor}")

    groups = report.duplicates.groups
    if groups:
        covered = sum(len(group.members) for group in groups)
        lines.append(
            f"duplicates   {len(groups)} group(s), {covered} images "
            f"({len(report.duplicates.exact)} exact, {len(report.duplicates.near)} near)"
        )
        for group in groups[:preview]:
            names = " = ".join(path.name for path in group.members[:4])
            if len(group.members) > 4:
                names += f" = ... ({len(group.members)})"
            lines.append(f"  {group.kind:<5} {names}")
        if len(groups) > preview:
            lines.append(f"  ... {len(groups) - preview} more group(s)")
    else:
        lines.append("duplicates   none" if report.duplicates.perceptual else
                     "duplicates   none exact (install pillow for near-duplicate detection)")

    if report.small:
        names = ", ".join(item.path.name for item in report.small[:4])
        lines.append(f"under size   {len(report.small)} image(s) below {report.min_side}px short side")
        lines.append(f"  {names}")

    captioned = [item for item in report.items if item.caption]
    lines.append(f"captioned    {len(captioned)}")

    if report.skipped_duplicates:
        lines.append(f"skipped      {len(report.skipped_duplicates)} duplicate(s)")
    if report.written:
        lines.append(f"written      {len(report.written)} caption file(s)")
    if report.failed:
        lines.append(f"failed       {len(report.failed)} call(s)")
        for path, message in report.failed[:3]:
            lines.append(f"  {path.name}: {message}")

    if report.dominant:
        lines.append("")
        lines.append("consensus")
        for key, value in sorted(report.dominant.items()):
            lines.append(f"  {key:<12} {value}")

    deviating = [item for item in report.items if item.deviations]
    if deviating:
        lines.append("")
        lines.append(f"deviations   {len(deviating)} image(s) disagree with the subject")
        for item in deviating[:10]:
            detail = ", ".join(f"{key}: {value}" for key, value in sorted(item.deviations.items()))
            lines.append(f"  {item.path.name}  {detail}")
        lines.append("  these are written into their captions; consider dropping them instead")

    if captioned and preview:
        lines.append("")
        lines.append("first captions")
        for item in captioned[:preview]:
            lines.append(f"  {item.path.name}")
            lines.append(f"    {item.caption}")

    return "\n".join(lines)
