"""Composing and running the accelerate command."""

from __future__ import annotations

import codecs
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

SPLIT_RE = re.compile(r"[\r\n]+")


def build_command(
    accelerate: str,
    script: Path,
    config: Path,
    *,
    num_cpu_threads: int = 1,
    mixed_precision: str = "bf16",
) -> list[str]:
    return [
        accelerate,
        "launch",
        "--num_cpu_threads_per_process",
        str(num_cpu_threads),
        "--mixed_precision",
        mixed_precision,
        str(script),
        "--config_file",
        str(config),
    ]


def run_command(
    command: list[str],
    *,
    cwd: Path | str,
    log_path: Path | str | None = None,
    on_line: Callable[[str], None] | None = None,
) -> int:
    """Run a training command, streaming output and teeing it to a log.

    Output is passed through as raw bytes so tqdm's in-place redrawing still looks
    right in a terminal, while a copy is decoded and split on both ``\\r`` and
    ``\\n`` for ``on_line``. A reader that waits for newlines would see nothing
    until the whole progress bar finished, because tqdm only emits ``\\r`` while it
    is running.
    """
    log_file = None
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(path, "wb")

    stdout = sys.stdout.buffer if hasattr(sys.stdout, "buffer") else None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    proc = None

    try:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        assert proc.stdout is not None

        while True:
            chunk = proc.stdout.read(8192)
            if not chunk:
                break

            if stdout is not None:
                stdout.write(chunk)
                stdout.flush()
            else:  # pragma: no cover - only when stdout is not a real stream
                sys.stdout.write(chunk.decode("utf-8", "replace"))
                sys.stdout.flush()

            if log_file is not None:
                log_file.write(chunk)
                log_file.flush()

            if on_line is not None:
                pending += decoder.decode(chunk)
                parts = SPLIT_RE.split(pending)
                pending = parts.pop()
                for part in parts:
                    if part.strip():
                        on_line(part)

        if on_line is not None:
            pending += decoder.decode(b"", final=True)
            for part in SPLIT_RE.split(pending):
                if part.strip():
                    on_line(part)

        return proc.wait()
    except FileNotFoundError as exc:
        raise RuntimeError(f"could not start {command[0]}: {exc}") from exc
    except KeyboardInterrupt:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise
    finally:
        if log_file is not None:
            log_file.close()
