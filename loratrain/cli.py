"""Command line interface for loratrain."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .dataset import Finding, report as build_report, scan
from .env import SdScripts, which_accelerate
from .presets import ALIASES, PRESETS, Preset, presets_text
from .runner import build_command, run_command
from .tomlgen import build_parts, write_files


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
    check.add_argument("--strict", action="store_true", help="exit non-zero on warnings too")
    check.set_defaults(handler=_cmd_check)

    doctor = sub.add_parser("doctor", help="show what loratrain can see on this machine")
    _add_env_args(doctor)
    doctor.set_defaults(handler=_cmd_doctor)

    plan = sub.add_parser("plan", help="audit a dataset, write the config, print the command")
    _add_training_args(plan)
    plan.set_defaults(handler=_cmd_plan)

    run = sub.add_parser("run", help="audit a dataset, write the config and start training")
    _add_training_args(run)
    run.add_argument("--dry-run", action="store_true", help="print everything but do not train")
    run.add_argument("--force", action="store_true", help="train even if the audit found errors")
    run.add_argument("--log", default=None, help="training log path (default: <out>/train.log)")
    run.set_defaults(handler=_cmd_run)

    return parser


def _add_scan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trigger", default=None, help="trigger word to check for in captions")
    parser.add_argument("--caption-ext", default=".txt", help="caption file extension")


def _add_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sd-scripts", default=None, help="path to the sd-scripts checkout")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="python used to probe sd-scripts flags (default: this interpreter)",
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
    text, findings = build_report(dataset, trigger=args.trigger, expected=preset.images)
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
    print(f"loratrain {__version__}")
    print(f"python      {sys.version.split()[0]}  ({sys.executable})")

    sd = SdScripts.discover(args.sd_scripts)
    if sd is None:
        print("sd-scripts  NOT FOUND")
        print("            pass --sd-scripts /path/to/sd-scripts or set LORATRAIN_SD_SCRIPTS")
    else:
        print(f"sd-scripts  {sd.root}")
        for name in ("sdxl_train_network.py", "flux_train_network.py"):
            print(f"  {name:<24} {'found' if sd.script(name) else 'MISSING'}")

        probe = sd.probe("flux_train_network.py", args.python)
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

    accelerate = which_accelerate()
    print(f"accelerate  {accelerate or 'NOT FOUND (pip install accelerate)'}")

    return 0 if sd is not None and accelerate else 1


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

    print()
    print("starting training - ctrl-c stops it, re-run to resume from the last saved state")
    try:
        code = run_command(plan.command, cwd=plan.sd_root, log_path=log_path)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130

    print(f"log: {log_path}")
    return code


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
    }
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

    top, general, dataset_section, subsets = build_parts(
        preset,
        dataset_dir=dataset.resolve(),
        out_dir=out_dir,
        name=name,
        models={
            key: (str(Path(value).expanduser()) if value else None)
            for key, value in models.items()
        },
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
        overrides=_parse_overrides(args.set),
    )

    sd = SdScripts.discover(args.sd_scripts)
    script = sd.script(preset.script) if sd is not None else None

    dropped: list[str] = []
    if sd is not None and not getattr(args, "no_filter", False):
        probe = sd.probe(preset.script, args.python)
        if probe is not None:
            top, dropped = _filter(top, probe.flags)

    if need_script:
        if sd is None:
            raise SystemExit(
                "sd-scripts not found: pass --sd-scripts /path/to/sd-scripts, "
                "or set LORATRAIN_SD_SCRIPTS"
            )
        if script is None:
            raise SystemExit(f"{preset.script} not found in {sd.root}")

    config, samples = write_files(
        preset,
        top,
        general,
        dataset_section,
        subsets,
        out_dir=out_dir,
        trigger=args.trigger,
    )

    accelerate = which_accelerate()
    command = None
    if script is not None and accelerate is not None:
        command = build_command(accelerate, script, config)
    elif need_script:
        raise SystemExit("accelerate not found: pip install accelerate")

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
