"""Stage execution over one saved Plan order, with independent GPU workers."""

from copy import deepcopy
import multiprocessing as mp
from pathlib import Path
from .config import run_root, asset, backend_config
from .io import read_json, read_rows, write_json, write_rows, fingerprint
from .names import MODULE_NAMES, module_key

STAGES = {"constructor": "raw", "periodic": "raw", "diffusion": "refined", "feedback": "edited"}


def stage_settings(config, stage, plans, output):
    stage = module_key(stage)
    if any((output / f"{previous}.settings.json").exists() for previous in MODULE_NAMES):
        raise ValueError("This output directory uses earlier stage metadata; choose a new output directory.")
    roles = {
        "constructor": ["dlm", "constructor"],
        "periodic": ["dlm", "constructor", "periodic"],
        "diffusion": ["constructor", "diffusion"],
        "feedback": ["dlm", "feedback", "verifier", "risk"],
    }[stage]
    assets = {}
    for role in roles:
        location = asset(config, role)
        p = Path(location)
        members = [p] if p.is_file() else sorted(p.glob("*")) if p.is_dir() else []
        assets[role] = {
            "location": location,
            "files": [(f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in members if f.is_file()],
        }
    definition = {"plans": fingerprint(plans), "models": assets, "settings": config[stage],
                  "dataset": config["dataset"]}
    if stage in ("constructor", "periodic"):
        other = "periodic" if stage == "constructor" else "constructor"
        if (output / f"{other}.settings.json").exists():
            raise ValueError("base constructor and periodic construction require separate output directories")
    previous = {"diffusion": "periodic", "feedback": "diffusion"}.get(stage)
    if stage == "diffusion" and (output / "constructor.settings.json").exists():
        previous = "constructor"
    if previous:
        definition["input_stage"] = read_json(output / f"{previous}.settings.json")
    if stage == "feedback":
        definition["evaluation"] = config["evaluation"]
    signature = fingerprint(definition)
    target = output / f"{stage}.settings.json"
    if target.exists() and read_json(target)["signature"] != signature:
        raise ValueError(f"{stage} configuration or inputs changed. Use a new --output directory.")
    write_json(target, {"signature": signature, "definition": definition})


def _worker(stage, config, plans, output, rank, world, device, protected):
    from .datasets import activate
    activate(config)
    from .device import setup_device

    device = setup_device(device, threads=config["runtime"]["threads"])
    import torch

    cfg = backend_config(config, stage=stage)
    root = Path(output)
    indices = [
        i for i in range(rank, len(plans), world) if not (root / STAGES[stage] / f"{i:06d}.json").exists()
    ]
    if not indices:
        return
    if stage in ("constructor", "periodic"):
        from ..periodic.generation import Constructor
        from ..periodic.trainer import load_head
        from .models import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(
            asset(config, "dlm"), asset(config, "constructor"), device, mean_resizing=False
        )
        model.eval().requires_grad_(False)
        head = load_head(asset(config, "periodic"), device) if stage == "periodic" else None
        generator = Constructor(model, tokenizer, cfg.inference, axis_head=head)
        for done, i in enumerate(indices, 1):
            value = generator.generate(plans[i])
            graph = value.pop("graph")
            value["graph_available"] = graph is not None
            if graph is not None:
                target = root / "graphs" / f"{i:06d}.pt"
                target.parent.mkdir(parents=True, exist_ok=True)
                torch.save(graph, target)
            write_json(root / "raw" / f"{i:06d}.json", value)
            print({"stage": stage, "worker": rank, "completed": done, "assigned": len(indices)}, flush=True)
    elif stage == "diffusion":
        from transformers import AutoTokenizer
        from ..diffusion.refinement import Refiner

        tokenizer = AutoTokenizer.from_pretrained(asset(config, "constructor"), trust_remote_code=True)
        refiner = Refiner(
            asset(config, "diffusion"),
            tokenizer,
            device,
            steps=config["diffusion"]["steps"],
            reuse_fixed_geometry=config["diffusion"]["reuse_fixed_geometry"],
        )
        for done, i in enumerate(indices, 1):
            generated = read_json(root / "raw" / f"{i:06d}.json")
            generated["graph"] = (
                torch.load(root / "graphs" / f"{i:06d}.pt", map_location="cpu", weights_only=False)
                if generated["graph_available"]
                else None
            )
            write_json(root / "refined" / f"{i:06d}.json", refiner.refine(plans[i], generated))
            print({"stage": stage, "worker": rank, "completed": done, "assigned": len(indices)}, flush=True)
    else:
        from ..feedback.editor import Editor
        from ..feedback.value import load_value
        from ..feedback.model import FeedbackReconstructor
        from ..feedback.revision import OnePassRepair
        from ..feedback.risk import RiskModel
        from .models import load_editor

        active = [i for i in indices if plans[i]["source_id"] not in protected]
        for i in indices:
            if plans[i]["source_id"] in protected:
                refined = read_json(root / "refined" / f"{i:06d}.json")
                write_json(
                    root / "edited" / f"{i:06d}.json",
                    {
                        "plan": plans[i],
                        "F": refined,
                        "E": {
                            "record": deepcopy(refined["record"]),
                            "selected_candidate": None,
                            "forward_calls": 0,
                            "candidates": [],
                        },
                        "decision": {"protected_sun": True},
                    },
                )
        if not active:
            return
        model, tokenizer = load_editor(asset(config, "dlm"), asset(config, "feedback"), device)
        value = load_value(asset(config, "verifier"))
        risk = RiskModel.load(asset(config, "risk"))
        settings = dict(
            config["feedback"]["revision"],
            editor_max_calls=config["feedback"]["max_calls"],
            temperature=config["feedback"]["temperature"],
        )
        feedback = FeedbackReconstructor(
            Editor(model, tokenizer, value, cfg.inference),
            OnePassRepair(model, tokenizer, value, risk, settings),
        )
        batch_size = config["feedback"]["batch_size"]
        for start in range(0, len(active), batch_size):
            selected = active[start : start + batch_size]
            refined = [read_json(root / "refined" / f"{i:06d}.json") for i in selected]
            bundles = feedback.edit_many([plans[i] for i in selected], refined)
            for i, bundle in zip(selected, bundles):
                bundle["decision"] = {"protected_sun": False}
                write_json(root / "edited" / f"{i:06d}.json", bundle)
            print(
                {
                    "stage": stage,
                    "worker": rank,
                    "completed": min(start + batch_size, len(active)),
                    "assigned": len(active),
                },
                flush=True,
            )


def sample(config, stage, *, plans=None, output=None):
    stage = module_key(stage)
    output = Path(output) if output else run_root(config) / "samples"
    if plans is not None:
        from ..data.plans import load_plans

        planned, _ = load_plans(plans, requests=config["sampling"]["requests"])
        if (output / "plans.jsonl").exists() and read_rows(output / "plans.jsonl") != planned:
            raise ValueError("Choose a new output directory for a different Plan sequence")
        write_rows(output / "plans.jsonl", planned)
    else:
        planned = read_rows(output / "plans.jsonl")
    stage_settings(config, stage, planned, output)
    protected = set()
    if stage == "feedback" and config["feedback"]["protect_sun"]:
        from ..evaluation.physics import record_key

        scores = read_rows(output / "evaluation/refined/scores.jsonl")
        refined = read_rows(output / "refined.jsonl")
        if len(scores) != len(refined) or any(
            s["source_id"] != r["source_id"] or s["record_key"] != record_key(r)
            for s, r in zip(scores, refined)
        ):
            raise ValueError("Evaluate the current refined structures before applying FeedbackReconstructor")
        protected = {s["source_id"] for s in scores if s["strict_sun"] is True}
    devices = config["runtime"]["devices"]
    count = (
        config["runtime"]["generation_workers_per_device"]
        if stage in ("constructor", "periodic")
        else config["runtime"]["refine_workers_per_device"]
        if stage == "diffusion"
        else 1
    )
    workers = [device for device in devices for _ in range(count)]
    context = mp.get_context("spawn")
    processes = []
    for rank, device in enumerate(workers):
        process = context.Process(
            target=_worker, args=(stage, config, planned, str(output), rank, len(workers), device, protected)
        )
        process.start()
        processes.append(process)
    try:
        for process in processes:
            process.join()
            if process.exitcode:
                raise RuntimeError(f"{stage} worker exited with code {process.exitcode}")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
    records = []
    bundles = []
    for i in range(len(planned)):
        value = read_json(output / STAGES[stage] / f"{i:06d}.json")
        records.append(value["E"]["record"] if stage == "feedback" else value["record"])
        if stage == "feedback":
            bundles.append(value)
    write_rows(output / f"{STAGES[stage]}.jsonl", records)
    if bundles:
        write_rows(output / "bundles.jsonl", bundles)
    return {"stage": stage, "requests": len(records), "records": str(output / f"{STAGES[stage]}.jsonl")}


def run(config, plans, *, output=None, start="periodic", end="evaluate", constructor="periodic"):
    from ..evaluation.hull import query
    from ..evaluation.workflow import evaluate

    start, end, constructor = map(module_key, (start, end, constructor))
    root = Path(output) if output else run_root(config) / "samples"
    from ..evaluation.direct import evaluate_direct
    stages = [constructor, "diffusion", "hull", "physics", "feedback", "evaluate"]
    if start == "periodic" and constructor == "constructor":
        start = "constructor"
    if start not in stages or end not in stages or stages.index(start) > stages.index(end):
        raise ValueError("Invalid stage range for this constructor")
    if start != constructor and plans:
        from ..data.plans import load_plans

        rows, _ = load_plans(plans, requests=config["sampling"]["requests"])
        if read_rows(root / "plans.jsonl") != rows:
            raise ValueError("Resume Plans differ from the saved request sequence")
    for stage in stages[stages.index(start) : stages.index(end) + 1]:
        if stage in STAGES:
            result = sample(config, stage, plans=plans if stage == constructor else None, output=root)
        elif stage == "hull":
            result = query(
                read_rows(root / "refined.jsonl"),
                run_root(config) / "hull",
                batch_size=config["evaluation"]["hull_batch_size"],
            )
        else:
            result = {}
            for endpoint in (["raw", "refined"] if stage == "physics" else ["edited"]):
                records = read_rows(root / f"{endpoint}.jsonl")
                _, direct = evaluate_direct(
                    records, root / "direct" / endpoint,
                    metrics=config["evaluation"]["direct"],
                    reference=run_root(config) / "data/structures/test.jsonl",
                    dataset=config["dataset"]["name"], workers=config["runtime"]["matching_workers"],
                    composition=config["evaluation"]["composition"],
                    coverage_cutoffs=config["evaluation"]["coverage_cutoffs"],
                    cache=run_root(config) / "cache/direct",
                )
                _, physical = evaluate(config, records, root / "evaluation" / endpoint)
                result[endpoint] = {"direct": direct, "sun": physical}
            if stage == "evaluate":
                from .finalize import finalize
                result["final"] = finalize(config, root, result["edited"])
        write_json(root / "progress.json", {"completed_stage": stage, "result": result})
    return result
