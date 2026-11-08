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

KNOWN_SCRIPTS = (
    "sdxl_train_network.py",
    "flux_train_network.py",
    "train_network.py",
    # musubi-tuner keeps these at the repository root, next to sd-scripts' own.
    "krea2_train_network.py",
    "flux_2_train_network.py",
    "zimage_train_network.py",
    "minimax_h3_train_network.py",
)
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
    def discover(cls, hint: str | None = None, script: str | None = None) -> "SdScripts | None":
        """Find the folder holding the training scripts, or None.

        With *script*, a folder that holds that particular script wins over one
        that only holds some other engine's: sd-scripts and musubi-tuner commonly
        sit side by side, and only one of them has ``krea2_train_network.py``.
        """
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
                home / "musubi-tuner",
                # Linux installs often live outside $HOME; these are the usual spots.
                Path("/opt/kohya_ss"),
                Path("/opt/kohya-ss"),
                Path("/opt/sd-scripts"),
                Path("/opt/musubi-tuner"),
                Path("/usr/local/sd-scripts"),
            ]
        )

        # A second drive is a normal place to keep a 20 GB checkout, and it is not
        # any of the paths above: /run/media/<user>/<label> or /media/<user>/<label>.
        for mount in _mount_roots():
            candidates.extend(
                [
                    mount,
                    mount / "sd-scripts",
                    mount / "kohya_ss",
                    mount / "kohya-ss",
                    mount / "musubi-tuner",
                ]
            )

        fallback: Path | None = None
        for candidate in candidates:
            found = cls._resolve(candidate)
            if found is None:
                continue
            if script is None or (found / script).is_file():
                return cls(found)
            if fallback is None:
                fallback = found

        return cls(fallback) if fallback is not None else None

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

    def python(self) -> str | None:
        """The interpreter that can run these scripts: a venv beside or above them.

        sd-scripts needs torch and accelerate, and on a working install they sit in
        a venv next to the checkout (``sd-scripts/.venv``) or beside it
        (``kohya_ss/.venv``). Using it by default is what makes ``--python``
        unnecessary in the normal case.
        """
        for folder in (self.root / ".venv", self.root.parent / ".venv", self.root / "venv", self.root.parent / "venv"):
            for name in ("bin/python", "bin/python3", "Scripts/python.exe"):
                path = folder / name
                try:
                    if path.is_file():
                        return str(path)
                except OSError:
                    continue
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


def _mount_roots() -> list[Path]:
    """The volumes mounted for this user, one level deep, in a stable order."""
    user = Path.home().name
    roots: list[Path] = []
    for parent in (Path("/run/media") / user, Path("/media") / user):
        try:
            entries = sorted(parent.iterdir())
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir():
                    roots.append(entry)
            except OSError:
                continue
    return roots


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
