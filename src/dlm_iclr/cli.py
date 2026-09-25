"""Command-line entry to the same Python module APIs used in training and inference."""

import argparse
import json
import os
from pathlib import Path
from .runtime.config import load, run_root, asset, backend_config, path
from .runtime.io import read_rows, write_json


def parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path)
    common.add_argument("--dataset", help="mp20, perov-5, mpts-52")
    common.add_argument("--num-samples", type=int, help="Fixed request count (default: 1000)")
    common.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE")
    main = argparse.ArgumentParser(
        prog="dlm", description="Crystal generation, refinement and physical evaluation"
    )
    commands = main.add_subparsers(dest="command", required=True)
    p = commands.add_parser("config", parents=[common], help="Write a portable configuration")
    p.add_argument("--output", type=Path, default=Path("configs/local.json"))
    p = commands.add_parser("prepare", parents=[common], help="Prepare module datasets from crystal sources")
    p.add_argument("--with-planner", action="store_true", help="Also tokenize optional Planner training data")
    p = commands.add_parser("plans", parents=[common], help="Select valid Plans in original order, without training")
    p.add_argument("--source")
    p.add_argument("--output", type=Path)
    p = commands.add_parser("train", parents=[common], help="Train a module or the full stack")
    p.add_argument("module", choices=["planner", "b0", "c1", "diffusion", "c2", "all"])
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--include-planner", action="store_true", help="Include Planner when module=all")
    p.add_argument(
        "--stage", choices=["warmup", "collect", "label", "compile", "editor", "light", "value", "risk"]
    )
    p = commands.add_parser("sample", parents=[common], help="Sample one stage")
    p.add_argument("module", choices=["planner", "b0", "c1", "diffusion", "c2"])
    p.add_argument("--plans")
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default=None)
    p = commands.add_parser(
        "run", parents=[common], help="Plan -> raw -> refinement -> physics -> editing -> metrics"
    )
    p.add_argument("--plans", required=True)
    p.add_argument("--constructor", choices=["b0", "c1"], default="c1")
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--from-stage", default="c1", choices=["b0", "c1", "diffusion", "hull", "physics", "c2", "evaluate"]
    )
    p.add_argument(
        "--to-stage", default="evaluate", choices=["b0", "c1", "diffusion", "hull", "physics", "c2", "evaluate"]
    )
    p = commands.add_parser("hull", parents=[common], help="Query Materials Project competitor energies")
    p.add_argument("--structures", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--api-key-file", type=Path)
    p = commands.add_parser("evaluate", parents=[common], help="Evaluate saved crystal records")
    p.add_argument("metrics", choices=["direct", "sun"])
    p.add_argument("--structures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--labels", type=Path)
    p.add_argument("--full", action="store_true")
    p.add_argument("--reference", type=Path)
    return main


def main(argv=None):
    args = parser().parse_args(argv)
    config = load(args.config, args.overrides, dataset=args.dataset)
    if args.num_samples is not None:
        if args.num_samples < 1:
            raise ValueError("--num-samples must be positive")
        config["sampling"]["requests"] = args.num_samples
        config["planner"]["sampling"]["requests"] = args.num_samples
    if hasattr(args, "device"):
        if args.device is not None:
            config["runtime"]["devices"] = [args.device]
        args.device = args.device or config["runtime"]["devices"][0]
    if args.command == "config":
        config.pop("_config_dir", None)
        write_json(args.output, config)
        result = {"config": str(args.output)}
    elif args.command == "prepare":
        from .data.adapters import prepare

        result = prepare(config, with_planner=args.with_planner)
    elif args.command == "plans":
        from .data.selection import select
        result = select(config, source=args.source, output=args.output)
    elif args.command == "train":
        if int(os.environ.get("WORLD_SIZE", "1")) > 1 and args.module != "b0":
            raise ValueError(
                "torchrun is supported by B0; use independent workers for sampling and evaluation"
            )
        import importlib

        modules = (["planner"] if args.include_planner else []) + ["b0", "c1", "c2"] if args.module == "all" else [args.module]
        result = {}
        for module in modules:
            implementation = importlib.import_module(
                f"dlm_iclr.{module}." + ("workflow" if module in ("planner", "c2") else "trainer")
            )
            kwargs = {"resume": args.resume}
            kwargs["device"] = args.device
            if module == "c2":
                kwargs["only"] = args.stage
            result[module] = implementation.train(config, **kwargs)
    elif args.command == "sample":
        if args.module == "planner":
            from .planner.sampling import generate_plans

            output = args.output or run_root(config) / "samples/plans.jsonl"
            result = {
                "plans": str(
                    generate_plans(
                        backend_config(config).assets,
                        output,
                        device=args.device,
                        **config["planner"]["sampling"],
                    )
                )
            }
        else:
            from .runtime.pipeline import sample

            result = sample(config, args.module, plans=args.plans, output=args.output)
    elif args.command == "run":
        from .runtime.pipeline import run

        result = run(config, args.plans, output=args.output, start=args.from_stage,
                     end=args.to_stage, constructor=args.constructor)
    elif args.command == "hull":
        from .evaluation.hull import query

        key = args.api_key_file.read_text().strip() if args.api_key_file else None
        result = query(
            read_rows(args.structures)[:config["sampling"]["requests"]],
            args.output or run_root(config) / "hull",
            api_key=key,
            batch_size=config["evaluation"]["hull_batch_size"],
        )
    elif args.metrics == "direct":
        from .evaluation.direct import evaluate_direct

        _, result = evaluate_direct(
            read_rows(args.structures)[:config["sampling"]["requests"]],
            args.output,
            metrics="full" if args.full else config["evaluation"]["direct"],
            reference=args.reference or run_root(config) / "data/structures/test.jsonl",
            dataset=config["dataset"]["name"],
            workers=config["runtime"]["matching_workers"],
            composition=config["evaluation"]["composition"],
            coverage_cutoffs=config["evaluation"]["coverage_cutoffs"],
        )
    else:
        from .evaluation.workflow import evaluate

        limit = config["sampling"]["requests"]
        labels = read_rows(args.labels)[:limit] if args.labels else None
        _, result = evaluate(config, read_rows(args.structures)[:limit], args.output, labels=labels)
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0
