"""Consensus over per-image attributes, to catch an inconsistent subject.

If 47 of 50 images of a character show brown hair and three show blonde, the three
are not noise to be averaged away: a LoRA trains on all of them, so the trigger
token learns that "brown" and "blonde" are both plausible. Catching that happens
here, and the caption writer states the outlier's value explicitly instead of
letting it blend in.
"""

from __future__ import annotations

import collections


def vote(
    items: dict[str, dict[str, str]],
    *,
    min_share: float = 0.8,
    min_items: int = 3,
) -> dict[str, str]:
    """Return attribute -> dominant value, only for attributes with a clear majority.

    Attributes absent from too many images never reach ``min_share`` and are left
    out, which is what keeps "pose" from being reported as inconsistent.
    """
    if len(items) < min_items:
        return {}

    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for attributes in items.values():
        for key, value in (attributes or {}).items():
            if not isinstance(value, str):
                continue
            key = key.strip().lower()
            value = value.strip()
            if key and value:
                counts[key][value.lower()] += 1

    total = len(items)
    dominant: dict[str, str] = {}
    for key, counter in counts.items():
        value, count = counter.most_common(1)[0]
        if count / total >= min_share:
            dominant[key] = value
    return dominant


def deviations(attributes: dict[str, str], dominant: dict[str, str]) -> dict[str, str]:
    """Attributes where this image disagrees with the consensus value."""
    result: dict[str, str] = {}
    for key, value in (attributes or {}).items():
        if not isinstance(value, str):
            continue
        key = key.strip().lower()
        value = value.strip()
        if not value or key not in dominant:
            continue
        if value.lower() != dominant[key]:
            result[key] = value
    return result
