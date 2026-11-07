"""A local web view of a training run: progress and the sample images.

sd-scripts writes sample grids to ``<out>/sample/`` as it trains, and its log holds
everything else. Both are already on disk, so this is a small HTTP server over
them: a page that polls ``/api/state`` twice a second, a JSON snapshot of the
monitor plus the sample list, and the images themselves.

Standard library only, bound to 127.0.0.1 by default.
"""

from __future__ import annotations

import codecs
import json
import math
import re
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .progress import Monitor, parse_line
from .runner import SPLIT_RE

SAMPLE_DIR = "sample"
SAMPLE_STEP_RE = re.compile(r"_(\d{8})_")
SAMPLE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
MAX_SAMPLES = 120
MAX_POINTS = 400


class LogTail:
    """Reads only what is new in the log, so polling stays cheap on a long run."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.monitor = Monitor()
        self.offset = 0
        self.pending = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.lock = threading.Lock()

    def poll(self) -> None:
        with self.lock:
            try:
                size = self.path.stat().st_size
            except OSError:
                return

            if size < self.offset:  # the log was truncated by a new run
                self.offset = 0
                self.pending = ""
                self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

            if size == self.offset:
                return

            try:
                with open(self.path, "rb") as handle:
                    handle.seek(self.offset)
                    data = handle.read()
                    self.offset = handle.tell()
            except OSError:
                return

            text = self.pending + self.decoder.decode(data)
            parts = SPLIT_RE.split(text)
            self.pending = parts.pop()
            for part in parts:
                sample = parse_line(part)
                if sample is not None:
                    self.monitor.add(sample)

    def status(self) -> dict:
        with self.lock:
            return status_dict(self.monitor)


def status_dict(monitor: Monitor) -> dict:
    step, total = monitor.progress()
    latest = monitor.latest()
    trend = monitor.trend()
    rate = monitor.rate()
    eta = monitor.eta()
    learning_rates = [sample.lr for sample in monitor.samples if sample.lr]

    points = [
        [sample.step, round(sample.loss, 6)]
        for sample in monitor.samples
        if sample.step is not None
        and sample.loss is not None
        and math.isfinite(sample.loss)
    ][-MAX_POINTS:]

    loss = None
    if latest is not None and latest.loss is not None and math.isfinite(latest.loss):
        loss = round(latest.loss, 6)

    return {
        "step": step,
        "total": total,
        "percent": round(100.0 * step / total, 1) if step and total else None,
        "loss": loss,
        "trend": round(trend[1] * 100, 2) if trend else None,
        "rate": round(rate, 3) if rate else None,
        "eta": int(eta) if eta is not None else None,
        "lr": [learning_rates[0], learning_rates[-1]] if learning_rates else None,
        "warnings": monitor.warnings(),
        "diverged": monitor.diverged(),
        "readings": len(monitor.samples),
        "points": points,
    }


def scan_samples(folder: Path | str) -> list[dict]:
    """Sample images, newest first, with the step parsed out of the filename."""
    folder = Path(folder)
    if not folder.is_dir():
        return []

    found: list[tuple[float, Path]] = []
    for file in folder.iterdir():
        if not file.is_file() or file.suffix.lower() not in SAMPLE_EXTS:
            continue
        try:
            found.append((file.stat().st_mtime, file))
        except OSError:
            continue

    found.sort(key=lambda item: item[0], reverse=True)

    entries: list[dict] = []
    for mtime, file in found[:MAX_SAMPLES]:
        match = SAMPLE_STEP_RE.search(file.name)
        entries.append(
            {
                "name": file.name,
                "step": int(match.group(1)) if match else None,
                "mtime": int(mtime),
                "url": "/samples/" + urllib.parse.quote(file.name),
            }
        )
    return entries


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>loratrain</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; background: #0e1116; color: #d8dee9;
       font: 14px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
header { padding: 14px 20px; border-bottom: 1px solid #1e2530;
         display: flex; gap: 22px; align-items: baseline; flex-wrap: wrap; }
h1 { font-size: 14px; margin: 0; font-weight: 600; letter-spacing: .04em; }
.k { color: #6b7785; }
main { padding: 18px 20px 40px; display: grid; gap: 18px; }
.bar { height: 6px; background: #1e2530; border-radius: 3px; overflow: hidden; }
.bar > div { height: 100%; width: 0; background: #4c9aff; transition: width .4s; }
canvas { width: 100%; height: 150px; display: block; }
.warn { border-left: 2px solid #d98d3a; background: #1a1410; color: #e0b07a;
        padding: 6px 10px; margin: 4px 0; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
           gap: 12px; }
figure { margin: 0; background: #141a22; border: 1px solid #1e2530;
         border-radius: 6px; overflow: hidden; }
figure img { width: 100%; display: block; aspect-ratio: 1; object-fit: cover; }
figcaption { padding: 5px 8px; font-size: 11px; color: #8b97a6;
             display: flex; justify-content: space-between; gap: 8px; }
a { color: inherit; text-decoration: none; }
.empty { color: #6b7785; padding: 18px 0; }
</style>
</head>
<body>
<header>
  <h1 id="name">loratrain</h1>
  <span id="step" class="k">waiting for the log</span>
  <span id="detail" class="k"></span>
  <span id="eta" class="k"></span>
</header>
<main>
  <div class="bar"><div id="bar"></div></div>
  <canvas id="chart"></canvas>
  <section id="warnings"></section>
  <section>
    <div class="k" id="count">samples</div>
    <div class="gallery" id="gallery"></div>
    <div class="empty" id="empty">no sample images yet - sample_every_n_epochs writes them to &lt;out&gt;/sample/</div>
  </section>
</main>
<script>
const el = {
  name: document.getElementById('name'),
  step: document.getElementById('step'),
  detail: document.getElementById('detail'),
  eta: document.getElementById('eta'),
  bar: document.getElementById('bar'),
  chart: document.getElementById('chart'),
  warnings: document.getElementById('warnings'),
  gallery: document.getElementById('gallery'),
  empty: document.getElementById('empty'),
  count: document.getElementById('count'),
};

let signature = '';

function duration(seconds) {
  if (seconds === null || seconds === undefined) return '';
  let s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  s = s % 60;
  if (h) return h + 'h' + String(m).padStart(2, '0') + 'm';
  if (m) return m + 'm' + String(s).padStart(2, '0') + 's';
  return s + 's';
}

function draw(points) {
  const canvas = el.chart;
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 600;
  const height = canvas.clientHeight || 150;
  if (canvas.width !== Math.floor(width * ratio)) {
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  if (!points || points.length < 2) return;

  const xs = points.map(p => p[0]);
  const ys = points.map(p => p[1]);
  const minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
  const minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);
  const spanX = (maxX - minX) || 1;
  const spanY = (maxY - minY) || 1;
  const pad = 8;

  ctx.strokeStyle = '#4c9aff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  points.forEach((point, index) => {
    const x = pad + (point[0] - minX) / spanX * (width - pad * 2);
    const y = height - pad - (point[1] - minY) / spanY * (height - pad * 2);
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = '#6b7785';
  ctx.font = '11px monospace';
  ctx.fillText(maxY.toFixed(3), pad, 12);
  ctx.fillText(minY.toFixed(3), pad, height - 2);
}

function render(state) {
  const s = state.status || {};
  el.name.textContent = state.name || 'loratrain';

  if (s.step !== null && s.step !== undefined) {
    el.step.textContent = s.total ? 'step ' + s.step + '/' + s.total + ' (' + s.percent + '%)' : 'step ' + s.step;
    el.bar.style.width = (s.percent || 0) + '%';
  } else {
    el.step.textContent = 'waiting for the log';
  }

  const bits = [];
  if (s.loss !== null && s.loss !== undefined) {
    let loss = 'loss ' + s.loss.toFixed(4);
    if (s.trend !== null && s.trend !== undefined) {
      loss += '  ' + (s.trend > 0 ? '+' : '') + s.trend + '%';
    }
    bits.push(loss);
  }
  if (s.rate) bits.push(s.rate.toFixed(2) + ' it/s');
  if (s.lr) bits.push('lr ' + Number(s.lr[1]).toExponential(2));
  el.detail.textContent = bits.join('   ');
  el.eta.textContent = s.eta !== null && s.eta !== undefined ? 'eta ' + duration(s.eta) : '';

  el.warnings.innerHTML = '';
  (s.warnings || []).forEach(function (text) {
    const div = document.createElement('div');
    div.className = 'warn';
    div.textContent = text;
    el.warnings.appendChild(div);
  });

  draw(s.points || []);

  const samples = state.samples || [];
  el.count.textContent = samples.length ? samples.length + ' sample image(s), newest first' : 'samples';
  el.empty.style.display = samples.length ? 'none' : 'block';

  const next = samples.map(function (item) { return item.name; }).join('|');
  if (next !== signature) {
    signature = next;
    el.gallery.innerHTML = '';
    samples.forEach(function (item) {
      const figure = document.createElement('figure');
      const link = document.createElement('a');
      link.href = item.url;
      link.target = '_blank';
      link.rel = 'noreferrer';

      const img = document.createElement('img');
      img.src = item.url;
      img.loading = 'lazy';
      img.alt = item.name;
      link.appendChild(img);

      const caption = document.createElement('figcaption');
      const left = document.createElement('span');
      left.textContent = item.step === null ? '' : 'step ' + item.step;
      const right = document.createElement('span');
      right.textContent = item.name.slice(-8);
      caption.appendChild(left);
      caption.appendChild(right);

      figure.appendChild(link);
      figure.appendChild(caption);
      el.gallery.appendChild(figure);
    });
  }
}

async function load() {
  try {
    const response = await fetch('/api/state', { cache: 'no-store' });
    if (!response.ok) throw new Error(String(response.status));
    render(await response.json());
  } catch (error) {
    el.step.textContent = 'disconnected';
  }
}

load();
setInterval(load, 2000);
</script>
</body>
</html>
"""


def _mime(name: str) -> str:
    suffix = Path(name).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")


def _handler_factory(out: Path, tail: LogTail):
    class Handler(BaseHTTPRequestHandler):
        server_version = "loratrain"

        def log_message(self, format, *args):  # noqa: A002 - keep the console quiet
            return

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - name fixed by the base class
            parsed = urllib.parse.urlsplit(self.path)

            if parsed.path == "/":
                self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return

            if parsed.path == "/api/state":
                tail.poll()
                state = {
                    "name": out.name,
                    "log": str(tail.path),
                    "status": tail.status(),
                    "samples": scan_samples(out / SAMPLE_DIR),
                }
                self._send(json.dumps(state).encode("utf-8"), "application/json")
                return

            if parsed.path.startswith("/samples/"):
                name = urllib.parse.unquote(parsed.path[len("/samples/"):])
                if not name or Path(name).name != name:
                    self._send(b"bad name", "text/plain; charset=utf-8", 400)
                    return
                path = out / SAMPLE_DIR / name
                if not path.is_file():
                    self._send(b"not found", "text/plain; charset=utf-8", 404)
                    return
                self._send(path.read_bytes(), _mime(name))
                return

            self._send(b"not found", "text/plain; charset=utf-8", 404)

    return Handler


def build_server(
    out: Path | str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> tuple[ThreadingHTTPServer, str]:
    out_dir = Path(out).expanduser().resolve()
    tail = LogTail(out_dir / "train.log")
    server = ThreadingHTTPServer((host, port), _handler_factory(out_dir, tail))
    return server, f"http://{host}:{server.server_address[1]}/"


def start_in_thread(server: ThreadingHTTPServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def _open_later(url: str) -> None:
    def open_url() -> None:
        time.sleep(0.5)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=open_url, daemon=True).start()


def run_server(out: Path | str, *, host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    server, url = build_server(out, host=host, port=port)
    print(f"serving {Path(out).expanduser().resolve()} on {url}  (ctrl-c to stop)")

    if open_browser:
        _open_later(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
