"""Composing and running the accelerate command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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


def run_command(command: list[str], *, cwd: Path | str, log_path: Path | str | None = None) -> int:
    """Run a training command, streaming output and teeing it to a log."""
    log_file = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        log_file = open(log_path, "a", encoding="utf-8")

    proc = None
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_file is not None:
                log_file.write(line)
                log_file.flush()
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
