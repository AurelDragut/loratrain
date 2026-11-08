"""Command line interface for loratrain."""

from __future__ import annotations

import argparse
import codecs
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .captioning import (
    DEFAULT_PROMPT,
    CaptionRequest,
    render_report as render_caption_report,
    run as run_caption,
)
from .comfy import find_models_dir, scan as scan_comfy, select as select_comfy
from .dataset import Finding, list_images, report as build_report, scan
from .dedupe import find_duplicates
from .env import SdScripts, which_accelerate
from .presets import ALIASES, PRESETS, Preset, presets_text
from .progress import Monitor, parse_line
from .runner import SPLIT_RE, build_command, run_command
from .musubigen import build_dataset, build_flags, flags_to_args
from .musubigen import write_files as write_musubi_files
from .serve import build_server, run_server, start_in_thread
from .tomlgen import build_parts, write_files
from .vlm import VlmConfig, list_models, resolve_flavor


@dataclass
class Plan:
    preset: Preset
    dataset: Path
    out_dir: Path
    name: str
    units: int
    images: int
    repeats: int
    resolution: int
    config: Path
    samples: Path | None
    command: list[str] | None
    sd_root: Path | None
    report_text: str
    findings: list[Finding]
    dropped: list[str]
    comfy: dict[str, str] = field(default_factory=dict)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2

    return handler(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loratrain",
        description="Audit a dataset, generate a kohya sd-scripts LoRA config, then train it.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", metavar="command")

    presets = sub.add_parser("presets", help="list the built-in model presets")
    presets.set_defaults(handler=_cmd_presets)

    check = sub.add_parser("check", help="audit a dataset folder")
    check.add_argument("dataset")
    check.add_argument("--preset", default="sdxl", choices=sorted(PRESETS))
    _add_scan_args(check)
    check.add_argument("--no-duplicates", action="store_true", help="skip duplicate detection")
    check.add_argument("--strict", action="store_true", help="exit non-zero on warnings too")
    check.set_defaults(handler=_cmd_check)

    doctor = sub.add_parser("doctor", help="show what loratrain can see on this machine")
    _add_env_args(doctor)
    doctor.add_argument("--vlm-url", default=None, help="also check a vision endpoint")
    doctor.add_argument("--vlm-model", default=None, help="model the endpoint should be serving")
    doctor.add_argument("--comfy", default=None, help="also report a ComfyUI folder's usable models")
    doctor.set_defaults(handler=_cmd_doctor)

    plan = sub.add_parser("plan", help="audit a dataset, write the config, print the command")
    _add_training_args(plan)
    plan.set_defaults(handler=_cmd_plan)

    run = sub.add_parser("run", help="audit a dataset, write the config and start training")
    _add_training_args(run)
    run.add_argument("--dry-run", action="store_true", help="print everything but do not train")
    run.add_argument("--force", action="store_true", help="train even if the audit found errors")
    run.add_argument("--log", default=None, help="training log path (default: <out>/train.log)")
    run.add_argument(
        "--report-every",
        type=float,
        default=60.0,
        help="seconds between progress readings while training (0 disables)",
    )
    run.add_argument(
        "--serve",
        action="store_true",
        help="start the web view for this run and print its URL",
    )
    run.add_argument("--port", type=int, default=8000, help="port for --serve")
    run.add_argument("--host", default="127.0.0.1", help="host for --serve")
    run.set_defaults(handler=_cmd_run)

    watch = sub.add_parser("watch", help="read a training log: speed, loss trend, eta")
    watch.add_argument("path", help="train.log or the output folder holding it")
    watch.add_argument("--follow", action="store_true", help="keep reading while training writes")
    watch.add_argument("--from-start", action="store_true", help="with --follow, read existing lines too")
    watch.add_argument("--every", type=float, default=30.0, help="seconds between readings when following")
    watch.set_defaults(handler=_cmd_watch)

    serve = sub.add_parser("serve", help="web view of a run: progress and sample images")
    serve.add_argument("out", help="the training output folder")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--no-browser", action="store_true")
    serve.set_defaults(handler=_cmd_serve)

    caption = sub.add_parser("caption", help="check duplicates and size, then caption with a vision model")
    caption.add_argument("dataset")
    caption.add_argument("--preset", default="sdxl", choices=sorted(PRESETS))
    caption.add_argument("--trigger", default=None, help="trigger word to put first in every caption")
    caption.add_argument("--vlm-url", default=None, help="OpenAI-compatible base URL, e.g. http://localhost:1234/v1")
    caption.add_argument("--vlm-model", default=None, help="model name the endpoint expects")
    caption.add_argument(
        "--vlm-flavor", choices=["auto", "openai", "ollama"], default="auto",
        help="auto treats a bare host as Ollama and anything with /v1 as OpenAI-compatible",
    )
    caption.add_argument("--vlm-timeout", type=float, default=None, help="seconds per request (default 300)")
    caption.add_argument("--api-key-env", default="LORATRAIN_VLM_API_KEY")
    caption.add_argument("--prompt", default=None, help="caption instruction sent to the model")
    caption.add_argument("--prompt-file", default=None, help="file holding the caption instruction")
    caption.add_argument(
        "--style", choices=["sentence", "tags", "both"], default="sentence",
        help="caption shape (sentence suits Flux, tags suit SDXL)",
    )
    caption.add_argument(
        "--subject", choices=["character", "general"], default="character",
        help="character checks attributes for consistency across the dataset",
    )
    caption.add_argument("--min-side", type=int, default=None, help="minimum short side (default: preset resolution)")
    caption.add_argument("--workers", type=int, default=1, help="parallel model calls")
    caption.add_argument(
        "--duplicates", choices=["skip", "keep", "only"], default="skip",
        help="skip duplicates, caption them anyway, or only report them",
    )
    caption.add_argument("--near-threshold", type=int, default=8, help="dHash distance for near-duplicates")
    caption.add_argument("--overwrite", action="store_true", help="rewrite existing caption files")
    caption.add_argument("--refresh", action="store_true", help="ignore the caption cache")
    caption.add_argument("--write", action="store_true", help="write the captions to disk")
    caption.add_argument("--limit", type=int, default=None, help="caption at most N images")
    caption.add_argument("--cache", default=None, help="caption cache path")
    caption.add_argument("--caption-ext", default=".txt")
    caption.set_defaults(handler=_cmd_caption)

    return parser


def _add_scan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trigger", default=None, help="trigger word to check for in captions")
    parser.add_argument("--caption-ext", default=".txt", help="caption file extension")


def _add_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sd-scripts", default=None, help="path to the sd-scripts checkout")
    parser.add_argument(
        "--python",
        default=None,
        help="python used to probe sd-scripts flags (default: the venv beside sd-scripts, else this interpreter)",
    )


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset", help="folder holding the images and their captions")
    parser.add_argument("--preset", default="sdxl", choices=sorted(PRESETS))
    parser.add_argument("--out", default=None, help="output folder (default: ./loratrain-out/<name>)")
    parser.add_argument("--name", default=None, help="output name")
    parser.add_argument("--base", default=None, help="base model or transformer weights")
    parser.add_argument("--vae", default=None, help="SDXL VAE (optional)")
    parser.add_argument("--clip-l", dest="clip_l", default=None, help="Flux clip_l weights")
    parser.add_argument("--t5xxl", default=None, help="Flux t5xxl weights")
    parser.add_argument("--ae", default=None, help="Flux autoencoder weights")
    parser.add_argument("--dit", default=None, help="musubi: transformer weights (Krea 2, FLUX.2, ...)")
    parser.add_argument(
        "--text-encoder", dest="text_encoder", default=None, help="musubi: text encoder weights"
    )
    parser.add_argument(
        "--comfy",
        default=None,
        help="ComfyUI root: fill any model path you did not pass from its models/ folder",
    )
    parser.add_argument("--repeats", type=int, default=None, help="num_repeats for the subset")
    _add_scan_args(parser)
    parser.add_argument("--resolution", type=int, default=None, help="training resolution")
    parser.add_argument("--min-bucket", type=int, default=None)
    parser.add_argument("--max-bucket", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None, help="network_dim")
    parser.add_argument("--alpha", type=int, default=None, help="network_alpha")
    parser.add_argument("--lr", type=float, default=None, help="learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="epochs (epoch-based presets)")
    parser.add_argument("--steps", type=int, default=None, help="steps (step-based presets)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="add or override a top-level config key, repeatable",
    )
    _add_env_args(parser)
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="do not drop keys the installed sd-scripts rejects",
    )


def _cmd_presets(args: argparse.Namespace) -> int:
    print(presets_text())
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    preset = PRESETS[args.preset]
    dataset = scan(
        args.dataset,
        resolution=preset.resolution,
        min_reso=preset.min_bucket_reso,
        max_reso=preset.max_bucket_reso,
        caption_ext=args.caption_ext,
    )
    text, findings = build_report(
        dataset,
        trigger=args.trigger,
        expected=preset.images,
        duplicates=None if args.no_duplicates else find_duplicates(list_images(args.dataset)),
        min_side=preset.resolution,
    )
    print(text)

    if findings:
        print()
        print("problems")
        for finding in findings:
            print(f"  {finding.level.upper():<5}  {finding.message}")

    errors = [finding for finding in findings if finding.level == "error"]
    warnings = [finding for finding in findings if finding.level == "warn"]
    if errors:
        return 1
    if warnings and args.strict:
        return 1
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    sd = SdScripts.discover(args.sd_scripts)
    python = _resolve_python(args, sd)

    print(f"loratrain {__version__}")
    print(f"python      {_python_version(python)}  ({python})")
    if sd is None:
        print("sd-scripts  NOT FOUND")
        print("            pass --sd-scripts /path/to/sd-scripts or set LORATRAIN_SD_SCRIPTS")
    else:
        print(f"sd-scripts  {sd.root}")
        for name in ("sdxl_train_network.py", "flux_train_network.py"):
            print(f"  {name:<24} {'found' if sd.script(name) else 'MISSING'}")

        probe = sd.probe("flux_train_network.py", python)
        if probe is None:
            print("  flag probe unavailable (running --help failed)")
        else:
            token_name = next(
                (name for name in ("t5xxl_max_token_length", "t5_max_token_length") if name in probe.flags),
                None,
            )
            print(f"  probed flags: {len(probe.flags)}")
            print(f"  t5 token length flag: --{token_name}" if token_name else "  t5 token length flag: not found")
            print(f"  flux_shift timestep sampling: {'yes' if probe.mentions('flux_shift') else 'no'}")

    accelerate = which_accelerate(python)
    print(f"accelerate  {accelerate or 'NOT FOUND'}")
    if accelerate is None:
        print(f"            try {python} -m pip install accelerate")

    ok = sd is not None and bool(accelerate)

    if args.vlm_url:
        config = VlmConfig(base_url=args.vlm_url, model=args.vlm_model or "")
        flavor = resolve_flavor(config)
        models = list_models(config)
        print(f"vlm         {args.vlm_url} ({flavor})")

        if models is None:
            print("            could not list models - is the server running?")
            ok = False
        else:
            shown = ", ".join(models[:6])
            if len(models) > 6:
                shown += f", ... ({len(models)} total)"
            print(f"            model(s): {shown}")
            if args.vlm_model and args.vlm_model not in models:
                print(f"            WARN {args.vlm_model} is not among them")
                ok = False

    if args.comfy:
        models_dir = find_models_dir(args.comfy)
        found = scan_comfy(args.comfy)
        print(f"comfy       {models_dir}")
        if not found:
            print("            no models found - is this the ComfyUI root?")
            ok = False
        else:
            usable = [model for model in found if model.usable]
            print(f"            {len(usable)} of {len(found)} model(s) usable for training")
            for model in found[:12]:
                print(f"  {'ok' if model.usable else 'NO'}  {model.role:<12} {model.path.name}")
                if model.note:
                    print(f"      {model.note}")

    return 0 if ok else 1


def _cmd_caption(args: argparse.Namespace) -> int:
    preset = PRESETS[args.preset]
    dataset = Path(args.dataset).expanduser()
    if not dataset.is_dir():
        raise SystemExit(f"dataset folder not found: {dataset}")

    prompt = DEFAULT_PROMPT
    if args.prompt_file:
        prompt = Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
    if args.prompt:
        prompt = args.prompt

    vlm = None
    if args.vlm_url:
        if not args.vlm_model:
            raise SystemExit("--vlm-url needs --vlm-model")
        api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
        vlm = VlmConfig(
            base_url=args.vlm_url, model=args.vlm_model, api_key=api_key, flavor=args.vlm_flavor
        )
        if args.vlm_timeout is not None:
            vlm.timeout = args.vlm_timeout

    min_side = args.min_side if args.min_side is not None else preset.resolution

    report = run_caption(
        CaptionRequest(
            dataset=dataset,
            trigger=args.trigger or "",
            min_side=min_side,
            style=args.style,
            subject=args.subject,
            duplicates_mode=args.duplicates,
            near_threshold=args.near_threshold,
            workers=args.workers,
            overwrite=args.overwrite,
            refresh=args.refresh,
            write=args.write,
            limit=args.limit,
            caption_ext=args.caption_ext,
            prompt=prompt,
            cache_path=Path(args.cache).expanduser() if args.cache else None,
            vlm=vlm,
        )
    )
    print(render_caption_report(report))

    if vlm is None:
        print()
        print("no --vlm-url given: only duplicate and size checks ran")

    return 1 if report.failed else 0


def _cmd_serve(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser()
    if not out.is_dir():
        raise SystemExit(f"not a folder: {out}")
    try:
        run_server(out, host=args.host, port=args.port, open_browser=not args.no_browser)
    except OSError as exc:
        raise SystemExit(f"cannot listen on {args.host}:{args.port}: {exc}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    plan = _prepare(args, need_script=False)
    _print_plan(plan)
    return 1 if any(finding.level == "error" for finding in plan.findings) else 0


def _cmd_run(args: argparse.Namespace) -> int:
    plan = _prepare(args, need_script=not args.dry_run)
    _print_plan(plan)

    errors = [finding for finding in plan.findings if finding.level == "error"]

    if args.dry_run:
        return 1 if errors else 0

    if errors and not args.force:
        print()
        print("refusing to train: fix the errors above, or pass --force")
        return 1

    if plan.command is None or plan.sd_root is None:
        print("cannot start training: sd-scripts or accelerate not found")
        return 1

    log_path = Path(args.log).expanduser() if args.log else plan.out_dir / "train.log"

    view_url = None
    if args.serve:
        try:
            server, view_url = build_server(plan.out_dir, host=args.host, port=args.port)
        except OSError as exc:
            print(f"could not start the web view on {args.host}:{args.port}: {exc}")
            view_url = None
        else:
            start_in_thread(server)
            print(f"live view   {view_url}")

    monitor = Monitor()
    last_report = [0.0]

    def on_line(line: str) -> None:
        sample = parse_line(line)
        if sample is None:
            return
        monitor.add(sample)
        if not args.report_every:
            return
        now = time.monotonic()
        if now - last_report[0] >= args.report_every:
            last_report[0] = now
            print(f"  [watch] {monitor.status_line()}")
            sys.stdout.flush()

    print()
    print("starting training - ctrl-c stops it, re-run to resume from the last saved state")
    try:
        code = run_command(plan.command, cwd=plan.sd_root, log_path=log_path, on_line=on_line)
    except KeyboardInterrupt:
        print("\ninterrupted")
        if monitor.samples:
            print()
            print(monitor.report())
        return 130

    if monitor.samples:
        print()
        print(monitor.report())

    print(f"log: {log_path}")
    if view_url:
        print(f"the web view stopped with the run; browse it again with `loratrain serve {plan.out_dir}`")
    return code


def _cmd_watch(args: argparse.Namespace) -> int:
    log = _resolve_log(Path(args.path).expanduser())

    if args.follow:
        return _follow(log, every=max(1.0, args.every), from_start=args.from_start)

    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SystemExit(f"cannot read {log}: {exc}")

    monitor = Monitor()
    for line in text.splitlines():
        sample = parse_line(line)
        if sample is not None:
            monitor.add(sample)

    print(f"log: {log}")
    print(monitor.report())
    return 0


def _follow(log: Path, *, every: float, from_start: bool) -> int:
    monitor = Monitor()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    last = 0.0

    print(f"following {log} (ctrl-c to stop)")
    try:
        with open(log, "rb") as handle:
            if not from_start:
                handle.seek(0, os.SEEK_END)

            while True:
                chunk = handle.read(8192)
                if chunk:
                    for part in SPLIT_RE.split(decoder.decode(chunk)):
                        if part.strip():
                            sample = parse_line(part)
                            if sample is not None:
                                monitor.add(sample)

                now = time.monotonic()
                if monitor.samples and now - last >= every:
                    last = now
                    print(monitor.status_line())
                    sys.stdout.flush()

                if not chunk:
                    time.sleep(0.5)
    except KeyboardInterrupt:
        print()
        print(monitor.report())
        return 0


def _resolve_log(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "train.log"
        if candidate.is_file():
            return candidate
        logs = sorted(path.glob("*.log"))
        if logs:
            return logs[-1]
        raise SystemExit(f"no train.log in {path}; pass the log file directly")

    if not path.is_file():
        raise SystemExit(f"no such file: {path}")

    return path


def _prepare(args: argparse.Namespace, *, need_script: bool) -> Plan:
    preset = PRESETS[args.preset]

    dataset = Path(args.dataset).expanduser()
    if not dataset.is_dir():
        raise SystemExit(f"dataset folder not found: {dataset}")

    units = _units(args, preset)

    models = {
        "base": args.base,
        "vae": args.vae,
        "clip_l": args.clip_l,
        "t5xxl": args.t5xxl,
        "ae": args.ae,
        "dit": args.dit,
        "text_encoder": args.text_encoder,
    }

    comfy_choice: dict[str, str] = {}
    if args.comfy:
        for key, path in select_comfy(scan_comfy(args.comfy), preset.name).items():
            if not models.get(key):
                models[key] = str(path)
                comfy_choice[key] = str(path)

    missing = [key for key in preset.requires if not models[key]]
    if missing:
        flags = ", ".join("--" + key.replace("_", "-") for key in missing)
        raise SystemExit(f"{preset.name} needs {', '.join(preset.requires)}; missing: {flags}")

    name = args.name or f"{preset.name}_{_slug(dataset.name)}"

    if args.out:
        out_dir = Path(args.out).expanduser()
    else:
        out_dir = Path.cwd() / "loratrain-out" / name
    out_dir = out_dir.resolve()

    resolution = args.resolution or preset.resolution
    min_bucket = args.min_bucket or preset.min_bucket_reso
    max_bucket = args.max_bucket or preset.max_bucket_reso
    if min_bucket > max_bucket:
        raise SystemExit("--min-bucket cannot be larger than --max-bucket")

    repeats = args.repeats if args.repeats is not None else preset.default_repeats

    dataset_scan = scan(
        dataset,
        resolution=resolution,
        min_reso=min_bucket,
        max_reso=max_bucket,
        caption_ext=args.caption_ext,
    )
    report_text, findings = build_report(
        dataset_scan, trigger=args.trigger, expected=preset.images
    )

    overrides = _parse_overrides(args.set)
    resolved_models = {
        key: (str(Path(value).expanduser()) if value else None) for key, value in models.items()
    }

    sd = SdScripts.discover(args.sd_scripts, preset.script)
    script = sd.script(preset.script) if sd is not None else None
    python = _resolve_python(args, sd)

    dropped: list[str] = []
    probe = None
    if sd is not None and not getattr(args, "no_filter", False):
        probe = sd.probe(preset.script, python)

    if preset.engine == "musubi":
        general, datasets = build_dataset(
            preset,
            dataset_dir=dataset.resolve(),
            out_dir=out_dir,
            repeats=repeats,
            resolution=resolution,
            caption_ext=args.caption_ext,
        )
        config, samples = write_musubi_files(
            preset, general, datasets, out_dir=out_dir, trigger=args.trigger
        )
        flags = build_flags(
            preset,
            out_dir=out_dir,
            name=name,
            models=resolved_models,
            units=units,
            seed=args.seed,
            dim=args.dim,
            alpha=args.alpha,
            lr=args.lr,
            dataset_config=config,
            sample_prompts=samples,
            overrides=overrides,
        )
        if probe is not None:
            flags, dropped = _filter(flags, probe.flags)
        extra_args: list[str] | None = flags_to_args(flags)
    else:
        top, general, dataset_section, subsets = build_parts(
            preset,
            dataset_dir=dataset.resolve(),
            out_dir=out_dir,
            name=name,
            models=resolved_models,
            repeats=repeats,
            trigger=args.trigger,
            resolution=resolution,
            min_bucket=min_bucket,
            max_bucket=max_bucket,
            dim=args.dim,
            alpha=args.alpha,
            lr=args.lr,
            units=units,
            seed=args.seed,
            caption_ext=args.caption_ext,
            overrides=overrides,
        )
        if probe is not None:
            top, dropped = _filter(top, probe.flags)
        config, samples = write_files(
            preset,
            top,
            general,
            dataset_section,
            subsets,
            out_dir=out_dir,
            trigger=args.trigger,
        )
        extra_args = None

    if need_script:
        if sd is None:
            raise SystemExit(
                "training scripts not found: pass --sd-scripts /path/to/sd-scripts or a "
                "musubi-tuner checkout, or set LORATRAIN_SD_SCRIPTS"
            )
        if script is None:
            raise SystemExit(f"{preset.script} not found in {sd.root}")

    accelerate = which_accelerate(python)
    command = None
    if script is not None and accelerate is not None:
        if preset.engine == "musubi":
            command = build_command(accelerate, script, extra_args=extra_args)
        else:
            command = build_command(accelerate, script, config)
    elif need_script:
        raise SystemExit(
            f"accelerate not found next to {python} or on PATH; "
            f"try {python} -m pip install accelerate"
        )

    return Plan(
        preset=preset,
        dataset=dataset,
        out_dir=out_dir,
        name=name,
        units=units,
        images=dataset_scan.count,
        repeats=repeats,
        resolution=resolution,
        config=config,
        samples=samples,
        command=command,
        sd_root=sd.root if sd is not None else None,
        report_text=report_text,
        findings=findings,
        dropped=dropped,
        comfy=comfy_choice,
    )


def _print_plan(plan: Plan) -> None:
    print(plan.report_text)
    print()
    print(f"preset     {plan.preset.name}  ({plan.preset.title})")
    print(f"config     {plan.config}")
    if plan.samples is not None:
        print(f"samples    {plan.samples}")

    if plan.preset.unit_kind == "epochs":
        total = plan.images * plan.repeats * plan.units
        print(
            f"budget     {plan.images} images x {plan.repeats} repeats x {plan.units} epochs "
            f"= {total} samples"
        )
    else:
        print(f"budget     {plan.units} steps")

    if plan.preset.notes:
        print(f"note       {plan.preset.notes}")
    if plan.dropped:
        print(f"dropped    {', '.join(plan.dropped)} (not accepted by the installed sd-scripts)")
    if plan.comfy:
        print("comfy      " + ", ".join(f"{key}={Path(value).name}" for key, value in plan.comfy.items()))

    if plan.findings:
        print()
        print("problems")
        for finding in plan.findings:
            print(f"  {finding.level.upper():<5}  {finding.message}")

    print()
    if plan.command is not None and plan.sd_root is not None:
        print("command")
        print("  " + " ".join(plan.command))
        print(f"cwd        {plan.sd_root}  (networks.* modules resolve from here)")
    else:
        print("command    unavailable: sd-scripts or accelerate not found")


def _units(args: argparse.Namespace, preset: Preset) -> int:
    if preset.unit_kind == "epochs":
        if args.steps is not None:
            raise SystemExit(f"{preset.name} trains in epochs; use --epochs, not --steps")
        return args.epochs if args.epochs is not None else preset.units

    if args.epochs is not None:
        raise SystemExit(f"{preset.name} trains in steps; use --steps, not --epochs")
    return args.steps if args.steps is not None else preset.units


def _filter(top: dict, supported: set[str]) -> tuple[dict, list[str]]:
    kept: dict = {}
    dropped: list[str] = []

    for key, value in top.items():
        if key in supported:
            kept[key] = value
            continue

        alias = ALIASES.get(key)
        if alias:
            replacement = next((name for name in alias if name in supported), None)
            if replacement is not None:
                kept[replacement] = value
                continue

        dropped.append(key)

    return kept, dropped


def _resolve_python(args: argparse.Namespace, sd: SdScripts | None = None) -> str:
    """The interpreter to probe and launch with: explicit, then sd-scripts' own venv.

    ``--python`` is only needed when the two live apart; normally the venv holding
    torch and accelerate sits beside the checkout, so it is the right default.
    """
    if getattr(args, "python", None):
        return args.python

    if sd is None:
        sd = SdScripts.discover(getattr(args, "sd_scripts", None))

    if sd is not None:
        interpreter = sd.python()
        if interpreter:
            return interpreter

    return sys.executable


def _python_version(python: str) -> str:
    if os.path.realpath(python) == os.path.realpath(sys.executable):
        return sys.version.split()[0]
    try:
        proc = subprocess.run(
            [python, "-c", "import sys;print(sys.version.split()[0])"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return "?"
    return (proc.stdout or "").strip() or "?"


def _parse_overrides(pairs: list[str]) -> dict:
    overrides: dict = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"--set needs KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip()
        if not key:
            raise SystemExit("--set needs a key")
        overrides[key] = _coerce(raw.strip())
    return overrides


def _coerce(raw: str):
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _slug(text: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in text.strip())
    parts = [part for part in cleaned.split("_") if part]
    return "_".join(parts).lower() or "dataset"
