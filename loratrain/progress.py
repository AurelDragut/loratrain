"""Reading sd-scripts progress output: are we learning, how fast, and when we stop.

sd-scripts paints a tqdm bar:

    steps:  16%|█▌        | 320/2000 [02:30<13:10, 2.13it/s, loss=0.128]

When the output is a pipe, tqdm repaints with carriage returns and never emits a
newline until the bar finishes, so only the last paint of a chunk is current — and
a reader that waits for newlines sees nothing at all while training runs. The
runner therefore splits on both characters, and ``parse_line`` takes the last
segment.

Everything here is pure: parse text, accumulate, report.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field

STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
IT_RE = re.compile(r"([\d.]+)\s*it/s")
SIT_RE = re.compile(r"([\d.]+)\s*s/it")
LOSS_RE = re.compile(r"(?:avr_)?loss\s*[=:]\s*([^\s,\]]+)", re.IGNORECASE)
LR_RE = re.compile(r"\blr\s*[=:]\s*([\d.eE+-]+)")
TIME_RE = re.compile(r"\[((?:\d+:)?\d+:\d+)<((?:\d+:)?\d+:\d+)")
EPOCH_RE = re.compile(r"epoch\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)


@dataclass
class Sample:
    step: int | None = None
    total: int | None = None
    loss: float | None = None
    lr: float | None = None
    epoch: int | None = None
    epochs: int | None = None
    it_per_s: float | None = None
    elapsed: float | None = None
    remaining: float | None = None


def _duration(text: str) -> float | None:
    try:
        parts = [int(part) for part in text.split(":")]
    except ValueError:
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return float(seconds)


def _number(text: str) -> float | None:
    lowered = text.strip().lower()
    if lowered in ("nan", "+nan", "-nan"):
        return float("nan")
    if lowered in ("inf", "+inf"):
        return float("inf")
    if lowered == "-inf":
        return float("-inf")
    try:
        return float(lowered)
    except ValueError:
        return None


def parse_line(text: str) -> Sample | None:
    """One progress sample out of a tqdm paint or a log line, or None."""
    if not text:
        return None

    chunk = text.split("\r")[-1].strip()
    if not chunk:
        return None

    lowered = chunk.lower()
    if "it/s" not in lowered and "s/it" not in lowered and "loss" not in lowered and "epoch" not in lowered:
        return None

    sample = Sample()

    # Only a progress bar carries step numbers; an "epoch 3/10" line would
    # otherwise be read as step 3 of 10.
    has_bar = "steps:" in lowered or "it/s" in lowered or "s/it" in lowered
    if has_bar:
        steps = STEP_RE.findall(chunk)
        if steps:
            step_text, total_text = steps[-1]
            try:
                sample.step = int(step_text)
                sample.total = int(total_text)
            except ValueError:
                pass

    match = IT_RE.search(chunk)
    if match:
        try:
            sample.it_per_s = float(match.group(1))
        except ValueError:
            pass
    else:
        match = SIT_RE.search(chunk)
        if match:
            try:
                seconds = float(match.group(1))
                if seconds > 0:
                    sample.it_per_s = 1.0 / seconds
            except ValueError:
                pass

    match = LOSS_RE.search(chunk)
    if match:
        sample.loss = _number(match.group(1))

    match = LR_RE.search(chunk)
    if match:
        sample.lr = _number(match.group(1))

    match = TIME_RE.search(chunk)
    if match:
        sample.elapsed = _duration(match.group(1))
        sample.remaining = _duration(match.group(2))

    match = EPOCH_RE.search(chunk)
    if match:
        try:
            sample.epoch = int(match.group(1))
            sample.epochs = int(match.group(2))
        except ValueError:
            pass

    if all(
        value is None
        for value in (
            sample.step,
            sample.loss,
            sample.lr,
            sample.epoch,
            sample.it_per_s,
        )
    ):
        return None

    return sample


def format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


@dataclass
class Monitor:
    """Everything a run needs to say about itself while it is still running."""

    capacity: int = 4000
    samples: list[Sample] = field(default_factory=list)

    def add(self, sample: Sample) -> None:
        self.samples.append(sample)
        if len(self.samples) > self.capacity:
            del self.samples[: len(self.samples) - self.capacity]

    # -- readings ---------------------------------------------------------

    def progress(self) -> tuple[int | None, int | None]:
        for sample in reversed(self.samples):
            if sample.step is not None:
                return sample.step, sample.total
        return None, None

    def latest(self) -> Sample | None:
        return self.samples[-1] if self.samples else None

    def losses(self) -> list[float]:
        return [
            sample.loss
            for sample in self.samples
            if sample.loss is not None and math.isfinite(sample.loss)
        ]

    def diverged(self) -> bool:
        return any(
            sample.loss is not None and not math.isfinite(sample.loss) for sample in self.samples
        )

    def rate(self) -> float | None:
        rates = [sample.it_per_s for sample in self.samples[-50:] if sample.it_per_s]
        if not rates:
            return None
        return statistics.median(rates)

    def eta(self) -> float | None:
        for sample in reversed(self.samples):
            if sample.remaining:
                return float(sample.remaining)
        step, total = self.progress()
        rate = self.rate()
        if step is not None and total and rate:
            return max(0.0, (total - step) / rate)
        return None

    def trend(self, window: int = 50) -> tuple[float, float] | None:
        """(absolute, relative) change between the halves of the last window."""
        values = self.losses()[-window:]
        if len(values) < 6:
            return None
        half = len(values) // 2
        first = statistics.fmean(values[:half])
        second = statistics.fmean(values[half:])
        if first == 0:
            return None
        return (second - first, (second - first) / abs(first))

    def speed_per_100(self, window: int = 50) -> float | None:
        """How much the loss moves per 100 samples. Negative is learning."""
        trend = self.trend(window)
        if trend is None:
            return None
        return trend[0] * 100.0 / (len(self.losses()[-window:]) / 2.0)

    # -- statements -------------------------------------------------------

    def warnings(self, window: int = 50) -> list[str]:
        result: list[str] = []

        if self.diverged():
            result.append(
                "loss went to nan/inf: training diverged. stop it, reload the last "
                "checkpoint and lower the learning rate"
            )
            return result

        trend = self.trend(window)
        if trend is not None:
            _change, relative = trend
            if relative > 0.05:
                result.append(
                    f"loss is rising ({relative * 100:+.1f}% over the last {window} points): "
                    "overfitting, or the learning rate is too high"
                )
            elif relative > -0.005:
                result.append(
                    f"loss is flat ({relative * 100:+.1f}%): more steps are unlikely to help - "
                    "look at the samples, then stop"
                )

        rates = [sample.it_per_s for sample in self.samples if sample.it_per_s]
        if len(rates) >= 30:
            third = max(1, len(rates) // 3)
            early = statistics.median(rates[:third])
            late = statistics.median(rates[-third:])
            if early and late < early * 0.6:
                result.append(
                    f"throughput fell from {early:.2f} to {late:.2f} it/s: data loading or "
                    "latent caching is the usual cause"
                )

        return result

    def status_line(self) -> str:
        parts: list[str] = []

        step, total = self.progress()
        if step is not None and total:
            parts.append(f"step {step}/{total} ({100 * step / total:.0f}%)")
        elif step is not None:
            parts.append(f"step {step}")

        latest = self.latest()
        if latest is not None and latest.loss is not None and math.isfinite(latest.loss):
            parts.append(f"loss {latest.loss:.4f}")

        trend = self.trend()
        if trend is not None:
            parts.append(f"{trend[1] * 100:+.1f}%")

        rate = self.rate()
        if rate:
            parts.append(f"{rate:.2f} it/s")

        eta = self.eta()
        if eta is not None:
            parts.append(f"eta {format_duration(eta)}")

        return "  ".join(parts) if parts else "no progress yet"

    def report(self) -> str:
        lines: list[str] = []

        step, total = self.progress()
        if step is not None and total:
            lines.append(f"steps       {step}/{total} ({100 * step / total:.0f}%)")
        elif step is not None:
            lines.append(f"steps       {step}")

        rate = self.rate()
        if rate:
            lines.append(f"speed       {rate:.2f} it/s")

        values = self.losses()
        if values:
            lines.append(f"loss        first {values[0]:.4f}   now {values[-1]:.4f}")
            if len(values) > 1 and values[0]:
                change = (values[-1] - values[0]) / abs(values[0])
                lines.append(f"change      {change * 100:+.1f}% since the first reading")

        speed = self.speed_per_100()
        if speed is not None:
            lines.append(f"learning    {speed:+.4f} loss per 100 samples")

        trend = self.trend()
        if trend is not None:
            lines.append(f"recent      {trend[1] * 100:+.2f}% over the last {min(len(values), 50)} points")

        rates = [sample.lr for sample in self.samples if sample.lr]
        if rates:
            lines.append(f"lr          {rates[0]:.3g} -> {rates[-1]:.3g}")

        eta = self.eta()
        if eta is not None:
            lines.append(f"eta         {format_duration(eta)}")

        warnings = self.warnings()
        if warnings:
            lines.append("")
            for warning in warnings:
                lines.append(f"warning     {warning}")

        return "\n".join(lines) if lines else "no progress lines found in that log"
