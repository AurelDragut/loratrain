"""Locating kohya's sd-scripts, the interpreter, and the flags a version accepts.

sd-scripts renames training flags between releases. Rather than guess, ask the
installed copy: run ``<script> --help`` once, collect the ``--flags`` it mentions,
and drop or rename anything a preset sets that this version does not know.

Three things have to line up for a run to work, and they can live in different
places: the scripts, the interpreter that has torch and accelerate, and the
``accelerate`` executable itself. On a venv install the last two are together,
which is why ``--python`` decides where ``accelerate`` is looked for first.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
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
        """Find the folder holding the training scripts, or None."""
        candidates: list[Path] = []

        if hint:
            candidates.append(Path(hint).expanduser())

        from_env = os.environ.get("LORATRAIN_SD_SCRIPTS")
        if from_env:
            candidates.append(Path(from_env).expanduser())

        home = Path.home()
        candidates.extend(
            [
                Path.cwd(),
                Path.cwd() / "sd-scripts",
                home / "sd-scripts",
                home / "kohya_ss",
                home / "kohya-ss",
                # Linux installs often live outside $HOME; these are the usual spots.
                Path("/opt/kohya_ss"),
                Path("/opt/kohya-ss"),
                Path("/opt/sd-scripts"),
                Path("/usr/local/sd-scripts"),
            ]
        )

        for candidate in candidates:
            found = cls._resolve(candidate)
            if found is not None:
                return cls(found)

        return None

    @staticmethod
    def _resolve(candidate: Path) -> Path | None:
        """The folder holding the scripts: *candidate* itself, or its sd-scripts/ child.

        Accepting the child is what lets ``--sd-scripts ~/kohya_ss`` work, which is
        the path people actually know.
        """
        for folder in (candidate, candidate / "sd-scripts"):
            try:
                if not folder.is_dir():
                    continue
            except OSError:
                continue
            if any((folder / name).is_file() for name in KNOWN_SCRIPTS):
                return folder
        return None

    def script(self, name: str) -> Path | None:
        path = self.root / name
        return path if path.is_file() else None

    def probe(self, script: str, python: str | None = None) -> Probe | None:
        """Ask the script which flags it accepts. None when that is not possible."""
        path = self.script(script)
        if path is None:
            return None

        interpreter = python or sys.executable
        try:
            proc = subprocess.run(
                [interpreter, str(path), "--help"],
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


def which_accelerate(python: str | None = None) -> str | None:
    """The accelerate executable, preferring the one in *python*'s environment.

    A venv's ``bin/python`` is usually a symlink to the system interpreter, so both
    the interpreter's own folder and its resolved folder are checked — the venv
    first, because that is where accelerate sits next to torch and sd-scripts'
    dependencies.
    """
    candidates: list[Path] = []

    if python:
        interpreter = Path(python).expanduser()
        folders: list[Path] = [interpreter.parent]
        try:
            resolved = interpreter.resolve()
            if resolved.parent != interpreter.parent:
                folders.append(resolved.parent)
        except OSError:
            pass

        for folder in folders:
            for name in ("accelerate", "accelerate.exe"):
                path = folder / name
                if path not in candidates:
                    candidates.append(path)

    found = shutil.which("accelerate")
    if found:
        candidates.append(Path(found))

    for candidate in candidates:
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        except OSError:
            continue

    return None
