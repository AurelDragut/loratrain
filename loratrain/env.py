"""Locating kohya's sd-scripts and the flags the installed version accepts.

sd-scripts renames training flags between releases. Rather than guess, ask the
installed copy: run ``<script> --help`` once, collect the ``--flags`` it mentions,
and drop or rename anything a preset sets that this version does not know.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

KNOWN_SCRIPTS = ("sdxl_train_network.py", "flux_train_network.py", "train_network.py")
FLAG_RE = re.compile(r"--([a-zA-Z0-9_]+)")


@dataclass
class Probe:
    flags: set[str]
    text: str

    def mentions(self, needle: str) -> bool:
        return needle in self.text


class SdScripts:
    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser().resolve()

    @classmethod
    def discover(cls, hint: str | None = None) -> "SdScripts | None":
        """Find an sd-scripts checkout, or None."""
        candidates: list[Path] = []

        if hint:
            candidates.append(Path(hint).expanduser())

        from_env = os.environ.get("LORATRAIN_SD_SCRIPTS")
        if from_env:
            candidates.append(Path(from_env).expanduser())

        candidates.extend(
            [
                Path.cwd(),
                Path.cwd() / "sd-scripts",
                Path.home() / "sd-scripts",
                Path.home() / "kohya_ss" / "sd-scripts",
                Path.home() / "kohya_ss",
            ]
        )

        for candidate in candidates:
            try:
                if not candidate.is_dir():
                    continue
            except OSError:
                continue
            if any((candidate / name).is_file() for name in KNOWN_SCRIPTS):
                return cls(candidate)

        return None

    def script(self, name: str) -> Path | None:
        path = self.root / name
        return path if path.is_file() else None

    def probe(self, script: str, python: str = "python") -> Probe | None:
        """Ask the script which flags it accepts. None when that is not possible."""
        path = self.script(script)
        if path is None:
            return None

        try:
            proc = subprocess.run(
                [python, str(path), "--help"],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=180,
            )
        except (OSError, subprocess.SubprocessError):
            return None

        text = (proc.stdout or "") + (proc.stderr or "")
        flags = set(FLAG_RE.findall(text))
        if not flags:
            return None
        return Probe(flags=flags, text=text)


def which_accelerate() -> str | None:
    return shutil.which("accelerate")
