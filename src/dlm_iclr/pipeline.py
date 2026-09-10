"""Shared orchestration for inference and repeated offline self-improvement."""

from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import time
from .config import Config
from .execution import run_commands
from .io import file_hash, fingerprint, read_json, read_rows, write_json, write_rows
from .plans import load_plans, composition_key


def resolve_plan_source(config, output, devices):
    if config.inference.plan_source != "generate":
        return config.inference.plan_source
    path = Path(output) / "generated_plans.jsonl"
    if not path.exists():
        write_json(Path(output) / "config.json", config.to_dict())
        run_commands(
            [
                (
                    "planner",
                    [
                        "plans",
                        "generate",
                        "--config",
                        Path(output) / "config.json",
                        "--output",
                        path,
                        "--requests",
                        max(config.inference.planner_requests, config.inference.requests or 0),
                        "--seed",
                        config.inference.seed,
                        "--device",
                        devices[0],
                    ],
                )
            ],
            Path(output) / "logs",
        )
    return path


def model_assets_identity(assets):
    """Content-based cache identity, computed from ordinary user-supplied assets."""
    result = {}
    for role in ("base_model", "generator", "editor", "value", "refiner"):
        value = getattr(assets, role)
        path = Path(value)
        if value == "bundled":
            path = Path(__file__).parent / "assets/autonomous_value.pt"
        if path.is_file():
            result[role] = {path.name: file_hash(path)}
        elif path.is_dir():
            names = {
                "config.json",
                "adapter_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "periodic_state_config.json",
                "periodic_state.pt",
                "expert_edit_config.json",
                "expert_edit_modules.pt",
            }
            files = [
                item
                for item in path.iterdir()
                if item.is_file()
                and (
                    item.name in names
                    or item.suffix == ".safetensors"
                    or item.name.startswith("pytorch_model")
                )
            ]
            result[role] = {item.name: file_hash(item) for item in sorted(files)}
        else:
            result[role] = {"model_identifier": value}
    return result


def run_inference(config, output, *, devices=("cuda:0",), refiner_workers=1, plans=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if plans is None:
        source = resolve_plan_source(config, output, devices)
        plans, selection = load_plans(
            source,
            requests=config.inference.requests,
            legal_only=config.inference.legal_only,
            seed=config.inference.seed,
        )
    else:
        plans = deepcopy(plans)
        for ordinal, plan in enumerate(plans):
            plan["ordinal"] = ordinal
        selection = {"requests": len(plans), "selection": "supplied_fixed_source_order"}
    state = {
        "config": config.to_dict(),
        "plans": fingerprint(plans),
        "model_assets": model_assets_identity(config.assets),
    }
    state_path = output / "inputs.json"
    if state_path.exists() and read_json(state_path) != state:
        raise ValueError(
            "Output directory belongs to different Plans or settings; select another output directory"
        )
    write_json(state_path, state)
    write_json(output / "config.json", config.to_dict())
    write_json(output / "selection.json", selection)
    write_rows(output / "plans.jsonl", plans)
    for stage in ("G", "F", "E"):
        worker_count = len(devices) * (refiner_workers if stage == "F" else 1)
        commands = [
            (
                f"{stage}_{rank}",
                [
                    "worker",
                    stage,
                    "--config",
                    output / "config.json",
                    "--plans",
                    output / "plans.jsonl",
                    "--output",
                    output,
                    "--rank",
                    rank,
                    "--world",
                    worker_count,
                    "--device",
                    devices[rank % len(devices)],
                ],
            )
            for rank in range(worker_count)
        ]
        started = time.monotonic()
        run_commands(commands, output / "logs")
        write_json(
            output / f"{stage}.timing.json",
            {
                "seconds": time.monotonic() - started,
                "workers": worker_count,
                "devices": list(devices),
                "requests": len(plans),
            },
        )
    records = [read_json(output / "E" / f"{i:05d}.json")["record"] for i in range(len(plans))]
    write_rows(output / "structures.jsonl", records)
    write_json(
        output / "inference.json",
        {
            "requests": len(plans),
            "successful_outputs": sum(row["success"] for row in records),
            "physical_inputs_during_selection": False,
            "output_sha256": fingerprint(records),
            "maximum_editor_calls": max(
                read_json(output / "E" / f"{i:05d}.json")["forward_calls"] for i in range(len(plans))
            ),
        },
    )
    return records


def evaluate_output(config, run, *, devices=("cuda:0",), workers_per_device=4, nu_workers=4, cache=None):
    from .physics import label_records
    from .evaluation import score_records

    run = Path(run)
    records = read_rows(run / "structures.jsonl")
    labels = label_records(
        records,
        config.assets.chgnet,
        run / "evaluation/physics",
        devices=devices,
        workers_per_device=workers_per_device,
        cache=Path(cache) / "physics" if cache else None,
    )
    return score_records(
        records,
        labels,
        config.assets.hull_cache,
        config.assets.novelty_reference,
        run / "evaluation",
        cache=Path(cache) / "scoring" if cache else None,
        workers=nu_workers,
    )


def training_feedback(config, run, *, devices=("cuda:0",), workers_per_device=4, nu_workers=4, cache=None):
    """Label each unique endpoint once, then score each candidate panel in source order."""
    from .physics import label_records
    from .evaluation import score_records
    from .feedback import compile_sources, generator_target_records
    from .models import build_dynamic_lightweight_constraints
    from crystal_dlm.r03_physics_transfer import build_repair_constraints
    from transformers import AutoTokenizer

    run = Path(run)
    plans = read_rows(run / "plans.jsonl")
    generated = [read_json(run / "G" / f"{i:05d}.json") for i in range(len(plans))]
    current = [read_json(run / "F" / f"{i:05d}.json") for i in range(len(plans))]
    edited = [read_json(run / "E" / f"{i:05d}.json") for i in range(len(plans))]
    panels = {
        "G": [value["record"] for value in generated],
        "current": [value["record"] for value in current],
    }
    for rank in range(config.inference.editor_candidates):
        panels[f"candidate_{rank}"] = [
            next((c["record"] for c in value["candidates"] if c["rank"] == rank), current[i]["record"])
            for i, value in enumerate(edited)
        ]
    ordered = [record for panel in panels.values() for record in panel]
    labels = label_records(
        ordered,
        config.assets.chgnet,
        run / "feedback/physics",
        devices=devices,
        workers_per_device=workers_per_device,
        cache=Path(cache) / "physics" if cache else None,
    )
    scores, offset = {}, 0
    for name, panel in panels.items():
        scored, report = score_records(
            panel,
            labels[offset : offset + len(panel)],
            config.assets.hull_cache,
            config.assets.novelty_reference,
            run / "feedback" / name,
            cache=Path(cache) / "scoring" if cache else None,
            workers=nu_workers,
        )
        scores[name] = scored
        offset += len(panel)
    tokenizer = AutoTokenizer.from_pretrained(config.assets.generator, trust_remote_code=True)
    targets = generator_target_records(plans, generated, current, edited, scores, tokenizer)
    write_rows(run / "feedback/G_teacher/structures.jsonl", targets)
    target_labels = label_records(
        targets,
        config.assets.chgnet,
        run / "feedback/G_teacher/physics",
        devices=devices,
        workers_per_device=workers_per_device,
        cache=Path(cache) / "physics" if cache else None,
    )
    scores["G_teacher"], _ = score_records(
        targets,
        target_labels,
        config.assets.hull_cache,
        config.assets.novelty_reference,
        run / "feedback/G_teacher",
        cache=Path(cache) / "scoring" if cache else None,
        workers=nu_workers,
    )
    g_rows, e_rows, value_rows = compile_sources(
        plans,
        generated,
        current,
        edited,
        scores,
        tokenizer=tokenizer,
        support=build_repair_constraints(tokenizer),
        generator_targets=targets,
    )
    for name, rows in (("G", g_rows), ("E", e_rows), ("value", value_rows)):
        write_rows(run / "feedback" / f"{name}.jsonl", rows)
    write_json(
        run / "feedback/dataset.json",
        {
            "source_Plans": len(plans),
            "G_rows": len(g_rows),
            "E_rows": len(e_rows),
            "value_rows": len(value_rows),
            "evaluation_sources_used": False,
        },
    )
    return run / "feedback"


def self_improve(
    config, output, *, devices=("cuda:0",), refiner_workers=1, workers_per_device=4, nu_workers=4
):
    output = Path(output).resolve()
    source = resolve_plan_source(config, output, devices)
    main, selection = load_plans(
        source,
        requests=config.inference.requests,
        legal_only=config.inference.legal_only,
        seed=config.inference.seed,
    )
    train, _ = load_plans(config.training.plans, legal_only=True, seed=config.training.seed)
    eval_compositions = {composition_key(row["plan_state"]) for row in main if row["body_eligible"]}
    if any(row.get("provenance", {}).get("usage_role") != "train" for row in train):
        raise ValueError("Self-improvement requires explicit TRAIN source provenance")
    overlap = eval_compositions & {composition_key(row["plan_state"]) for row in train}
    if overlap:
        raise ValueError(
            "TRAIN contains evaluation compositions; prepare a composition-disjoint training file"
        )
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "train_plans.jsonl", train)
    write_rows(output / "evaluation_plans.jsonl", main)
    write_json(
        output / "source_split.json",
        {
            "train_sources": len(train),
            "evaluation_sources": len(main),
            "composition_overlap": 0,
            "train_sha256": fingerprint(train),
            "evaluation_sha256": fingerprint(main),
        },
    )
    current_config = deepcopy(config)
    history = []
    for round_index in range(config.training.rounds + 1):
        if round_index:
            train_run = output / f"round_{round_index}/train"
            run_inference(
                current_config, train_run, devices=devices, refiner_workers=refiner_workers, plans=train
            )
            feedback = training_feedback(
                current_config,
                train_run,
                devices=devices,
                workers_per_device=workers_per_device,
                nu_workers=nu_workers,
                cache=output / "cache",
            )
            settings = deepcopy(current_config)
            settings.training.seed = config.training.seed + round_index
            settings_path = output / f"round_{round_index}/training_config.json"
            write_json(settings_path, settings.to_dict())
            checkpoints = output / f"round_{round_index}/models"
            commands = [
                (
                    branch,
                    [
                        "train",
                        branch,
                        "--config",
                        settings_path,
                        "--data",
                        feedback / f"{branch}.jsonl",
                        "--output",
                        checkpoints / branch,
                        "--device",
                        devices[index % len(devices)],
                    ],
                )
                for index, branch in enumerate(("G", "E"))
            ]
            if len(devices) > 1:
                run_commands(commands, checkpoints / "logs")
            else:
                for command in commands:
                    run_commands([command], checkpoints / "logs")
            settings.assets.generator = str(checkpoints / "G/checkpoint")
            settings.assets.editor = str(checkpoints / "E/checkpoint")
            write_json(settings_path, settings.to_dict())
            run_commands(
                [
                    (
                        "value",
                        [
                            "train",
                            "value",
                            "--config",
                            settings_path,
                            "--data",
                            feedback / "value.jsonl",
                            "--output",
                            checkpoints / "value",
                            "--device",
                            devices[0],
                        ],
                    )
                ],
                checkpoints / "logs",
            )
            settings.assets.value = str(checkpoints / "value/autonomous_value.pt")
            current_config = settings
        snapshot = output / f"S{round_index}"
        run_inference(current_config, snapshot, devices=devices, refiner_workers=refiner_workers, plans=main)
        _, report = evaluate_output(
            current_config,
            snapshot,
            devices=devices,
            workers_per_device=workers_per_device,
            nu_workers=nu_workers,
            cache=output / "cache",
        )
        history.append({"snapshot": f"S{round_index}", **report})
        write_json(output / "history.json", history)
    return history
