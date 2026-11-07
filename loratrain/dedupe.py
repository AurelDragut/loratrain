"""Grouping exact and near-duplicate images."""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from pathlib import Path

from .hashes import difference_hash, file_digest, hamming, perceptual_available


@dataclass
class DuplicateGroup:
    kind: str  # "exact" or "near"
    members: list[Path]

    @property
    def representative(self) -> Path:
        return self.members[0]


@dataclass
class DuplicateReport:
    groups: list[DuplicateGroup] = field(default_factory=list)
    perceptual: bool = False

    @property
    def exact(self) -> list[DuplicateGroup]:
        return [group for group in self.groups if group.kind == "exact"]

    @property
    def near(self) -> list[DuplicateGroup]:
        return [group for group in self.groups if group.kind == "near"]

    def duplicate_paths(self, keep_representative: bool = True) -> set[Path]:
        """Every member of a group, or all but the first of each group."""
        paths: set[Path] = set()
        for group in self.groups:
            members = group.members[1:] if keep_representative else group.members
            paths.update(members)
        return paths


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def find_duplicates(paths: list[Path | str], *, threshold: int = 8) -> DuplicateReport:
    """Cluster paths by identical bytes and then by perceptual hash.

    Exact detection always runs. Near-duplicate clustering needs Pillow and is
    skipped (with ``perceptual`` False) when it is not installed.
    """
    paths = [Path(path) for path in paths]
    report = DuplicateReport(perceptual=perceptual_available())
    if len(paths) < 2:
        return report

    digests: dict[int, str] = {}
    by_digest: dict[str, list[int]] = collections.defaultdict(list)
    for index, path in enumerate(paths):
        try:
            digest = file_digest(path)
        except OSError:
            continue
        digests[index] = digest
        by_digest[digest].append(index)

    union = _UnionFind(len(paths))

    for indices in by_digest.values():
        for other in indices[1:]:
            union.union(indices[0], other)

    if report.perceptual:
        hashes: dict[int, int] = {}
        for index in digests:
            value = difference_hash(paths[index])
            if value is not None:
                hashes[index] = value

        hashed = list(hashes)
        for left in range(len(hashed)):
            for right in range(left + 1, len(hashed)):
                if hamming(hashes[hashed[left]], hashes[hashed[right]]) <= threshold:
                    union.union(hashed[left], hashed[right])

    components: dict[int, list[int]] = collections.defaultdict(list)
    for index in digests:
        components[union.find(index)].append(index)

    for indices in components.values():
        if len(indices) < 2:
            continue
        unique_digests = {digests[index] for index in indices}
        kind = "exact" if len(unique_digests) == 1 else "near"
        members = [paths[index] for index in sorted(indices)]
        report.groups.append(DuplicateGroup(kind=kind, members=members))

    report.groups.sort(key=lambda group: (-len(group.members), str(group.members[0])))
    return report
