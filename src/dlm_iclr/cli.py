"""Command-line entry points; heavy model imports occur only in model commands."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from .config import Config, load_config
from .io import read_rows, write_rows, write_json


def parser():
    command = argparse.ArgumentParser(
        prog="dlm-iclr", description="Crystal generation, refinement and autonomous KEEP/EDIT"
    )
    commands = command.add_subparsers(dest="command", required=True)
    config = commands.add_parser(
        "config", help="Write an editable configuration with the retained default hyperparameters"
    )
    config.add_argument("--output", type=Path, required=True)
    plans = commands.add_parser("plans", help="Export a real Plan preset or sample fresh Plans")
    plans.add_argument("source", help="H1A2_1200, R03_256, a JSONL path, or generate")
    plans.add_argument("--output", type=Path, required=True)
    plans.add_argument("--config", type=Path)
    plans.add_argument("--requests", type=int)
    plans.add_argument("--seed", type=int, default=17)
    plans.add_argument("--legal-only", action="store_true")
    plans.add_argument("--device", default="cuda:0")
    plans.add_argument("--usage-role", choices=("train", "evaluation"), default="evaluation")
    dataset = commands.add_parser(
        "prepare-data", help="Convert MP-20 or another CIF dataset to the common representation"
    )
    dataset.add_argument("--source", type=Path, required=True)
    dataset.add_argument("--output", type=Path, required=True)
    dataset.add_argument("--dataset", default="mp20")
    dataset.add_argument("--split", choices=("train", "val", "test", "unspecified"), default="train")
    train_data = commands.add_parser(
        "prepare-train", help="Deduplicate training compositions and exclude evaluation sources"
    )
    train_data.add_argument("--source", required=True)
    train_data.add_argument("--output", type=Path, required=True)
    train_data.add_argument("--exclude", action="append", default=[])
    train_data.add_argument("--limit", type=int)
    for name in ("infer", "self-improve"):
        run = commands.add_parser(
            name,
            help="Run G -> F -> E"
            if name == "infer"
            else "Run fixed-Plan evaluation and offline training rounds",
        )
        run.add_argument("--config", type=Path, required=True)
        run.add_argument("--output", type=Path, required=True)
        run.add_argument("--plan-source")
        run.add_argument("--requests", type=int)
        run.add_argument("--legal-only", action=argparse.BooleanOptionalAction, default=None)
        run.add_argument("--gpus", type=int, default=1)
        run.add_argument("--refiner-workers", type=int, default=1)
        run.add_argument("--physics-workers", type=int, default=4)
        run.add_argument("--nu-workers", type=int, default=4)
        if name == "infer":
            run.add_argument(
                "--evaluate", action="store_true", help="Evaluate after all output choices are complete"
            )
        else:
            run.add_argument("--rounds", type=int)
            run.add_argument("--training-plans")
            run.add_argument("--training-seed", type=int)
    evaluation = commands.add_parser("evaluate", help="Evaluate saved selected structures")
    evaluation.add_argument("--config", type=Path, required=True)
    evaluation.add_argument("--run", type=Path, required=True)
    evaluation.add_argument("--gpus", type=int, default=1)
    evaluation.add_argument("--physics-workers", type=int, default=4)
    evaluation.add_argument("--nu-workers", type=int, default=4)
    training = commands.add_parser(
        "train", help="Train an actor or autonomous value model from compiled TRAIN feedback"
    )
    training.add_argument("branch", choices=("G", "E", "value"))
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--checkpoint")
    training.add_argument("--device", default="cuda:0")
    worker = commands.add_parser("worker", help="Run one inference stage on a device")
    worker.add_argument("stage", choices=("G", "F", "E"))
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--plans", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--rank", type=int, default=0)
    worker.add_argument("--world", type=int, default=1)
    worker.add_argument("--device", default="cuda:0")
    return command


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if args.command == "config":
        write_json(args.output, Config().to_dict())
    elif args.command == "plans":
        if args.source == "generate":
            if not args.config:
                command.error("plans generate requires --config")
            from .execution import setup_device
            from .planner import generate_plans

            generate_plans(
                load_config(args.config).assets,
                args.output,
                requests=args.requests or 1200,
                seed=args.seed,
                device=setup_device(args.device),
                usage_role=args.usage_role,
            )
        else:
            from .plans import load_plans

            rows, report = load_plans(
                args.source, requests=args.requests, legal_only=args.legal_only, seed=args.seed
            )
            write_rows(args.output, rows)
            write_json(args.output.with_suffix(".selection.json"), report)
            print(json.dumps(report), flush=True)
    elif args.command == "prepare-data":
        from .datasets import convert_dataset

        print(
            json.dumps(convert_dataset(args.source, args.output, dataset=args.dataset, split=args.split)),
            flush=True,
        )
    elif args.command == "prepare-train":
        from .plans import prepare_training, PRESETS

        report = prepare_training(
            args.source, args.output, exclusions=[*PRESETS, *args.exclude], limit=args.limit
        )
        print(json.dumps({key: value for key, value in report.items() if key != "omitted"}), flush=True)
    elif args.command == "worker":
        from .execution import inference_worker

        inference_worker(
            args.stage,
            args.config,
            args.plans,
            args.output,
            rank=args.rank,
            world=args.world,
            device=args.device,
        )
    elif args.command == "train":
        from .execution import setup_device

        world = int(os.environ.get("WORLD_SIZE", "1"))
        device = setup_device(f"cuda:{os.environ['LOCAL_RANK']}" if world > 1 else args.device)
        import torch.distributed as dist

        if world > 1:
            if args.branch == "value":
                command.error("Value readout training uses a single device")
            dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
        try:
            if args.branch == "value":
                from .value_training import train_value

                train_value(
                    load_config(args.config),
                    args.data,
                    args.output,
                    editor_checkpoint=args.checkpoint,
                    device=device,
                )
            else:
                from .training import train_actor

                train_actor(
                    args.branch,
                    load_config(args.config),
                    args.data,
                    args.output,
                    checkpoint=args.checkpoint,
                    device=device,
                )
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
    else:
        from .execution import setup_device
        from .pipeline import run_inference, evaluate_output, self_improve

        config = load_config(args.config)
        if args.gpus < 1:
            command.error("--gpus must be positive")
        devices = [f"cuda:{i}" for i in range(args.gpus)]
        setup_device(devices[0])
        if args.command == "evaluate":
            _, report = evaluate_output(
                config,
                args.run,
                devices=devices,
                workers_per_device=args.physics_workers,
                nu_workers=args.nu_workers,
                cache=args.run.parent / "cache",
            )
            print(json.dumps(report["counts"]), flush=True)
            return
        for name in ("plan_source", "requests", "legal_only"):
            value = getattr(args, name)
            if value is not None:
                setattr(config.inference, name, value)
        if args.command == "self-improve":
            for name, destination in (
                ("rounds", "rounds"),
                ("training_plans", "plans"),
                ("training_seed", "seed"),
            ):
                value = getattr(args, name)
                if value is not None:
                    setattr(config.training, destination, value)
            self_improve(
                config,
                args.output,
                devices=devices,
                refiner_workers=args.refiner_workers,
                workers_per_device=args.physics_workers,
                nu_workers=args.nu_workers,
            )
        else:
            run_inference(config, args.output, devices=devices, refiner_workers=args.refiner_workers)
            if args.evaluate:
                _, report = evaluate_output(
                    config,
                    args.output,
                    devices=devices,
                    workers_per_device=args.physics_workers,
                    nu_workers=args.nu_workers,
                    cache=args.output.parent / "cache",
                )
                print(json.dumps(report["counts"]), flush=True)
