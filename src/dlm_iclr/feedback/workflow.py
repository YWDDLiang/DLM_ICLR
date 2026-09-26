"""Reusable stages for geometric warm-up, physical supervision and local revision."""

from ..runtime.config import run_root, asset, path, backend_config
from ..runtime.io import read_rows, read_json, write_rows, write_json


def require_feedback_inputs(config):
    """Check collected feedback inputs before starting model fitting."""
    root = run_root(config) / "feedback"
    files = [root / "collection/bundles.jsonl", root / "labels/current.jsonl"]
    files += [root / "labels" / f"candidate_{i}.jsonl" for i in range(config["feedback"]["candidates"])]
    missing = [str(p.relative_to(root)) for p in files if not p.is_file()]
    if not (root / "warmup/checkpoint").is_dir():
        missing.append("warmup/checkpoint/")
    if missing:
        raise FileNotFoundError(
            "Run scripts/05_collect_feedback.sh before feedback training (see docs/feedback_data.md). Missing: "
            + ", ".join(missing)
        )


def load_feedback_inputs(config):
    from ..evaluation.physics import record_key

    require_feedback_inputs(config)
    root = run_root(config) / "feedback"
    bundles = read_rows(root / "collection/bundles.jsonl")
    if not bundles:
        raise ValueError("Feedback data must contain at least one training source")
    groups = ["current", *[f"candidate_{i}" for i in range(config["feedback"]["candidates"])]]
    scores = {name: read_rows(root / "labels" / f"{name}.jsonl") for name in groups}
    if any(len(rows) != len(bundles) for rows in scores.values()):
        raise ValueError("Feedback label files must have one row per training source")
    seen = set()
    for index, bundle in enumerate(bundles):
        plan = bundle["plan"]
        source = plan["source_id"]
        if source in seen or plan.get("provenance", {}).get("usage_role") != "train":
            raise ValueError("Feedback sources must be unique TRAIN records")
        seen.add(source)
        before = bundle["F"]["record"]
        candidates = {c["rank"]: c for c in bundle["E"]["candidates"]}
        if len(candidates) != len(bundle["E"]["candidates"]) or any(
            type(rank) is not int or not 0 <= rank < config["feedback"]["candidates"] for rank in candidates
        ):
            raise ValueError("Feedback candidate ranks must be unique and within the configured count")
        records = {"current": before}
        for rank in range(config["feedback"]["candidates"]):
            records[f"candidate_{rank}"] = candidates[rank]["record"] if rank in candidates else dict(
                before, success=False, reason="candidate_not_generated"
            )
        for name, record in records.items():
            score = scores[name][index]
            if record["source_id"] != source or score["source_id"] != source or score["record_key"] != record_key(record):
                raise ValueError(f"Feedback label does not match source and geometry: {name}, row {index}")
    return bundles, scores


def collect(config, *, device="cuda:0"):
    import torch
    from ..data.plans import load_plans
    from ..runtime.pipeline import sample
    from ..runtime.models import load_editor
    from ..runtime.device import setup_device
    from .editor import Editor
    from .value import ValueNetwork

    root = run_root(config)
    out = root / "feedback/collection"
    source = config["feedback"]["training"].get("plans", "@run/data/plans/train.jsonl")
    source = (
        str(root / source[5:])
        if source.startswith("@run/")
        else source[7:]
        if source.startswith("preset:")
        else str(path(config, source))
    )
    plans, _ = load_plans(source)
    limit = config["feedback"]["training"].get("sources", 1000)
    plans = plans[:limit] if limit else plans
    if any(p.get("provenance", {}).get("usage_role") != "train" for p in plans):
        raise ValueError("FeedbackReconstructor collection needs training Plan sources")
    write_rows(out / "plans.jsonl", plans)
    sample(config, "periodic", output=out)
    sample(config, "diffusion", output=out)
    model, tokenizer = load_editor(
        asset(config, "dlm"),
        str(root / "feedback/warmup/checkpoint"),
        setup_device(device, threads=config["runtime"]["threads"]),
    )
    quality = model.quality_head.state_dict()
    weight = quality["layers.0.weight"]
    bias = quality["layers.0.bias"]
    value = ValueNetwork(weight.shape[1], weight.shape[0], 9)
    with torch.no_grad():
        value.hidden.weight.copy_(weight.cpu())
        value.hidden.bias.copy_(bias.cpu())
        value.head.weight.zero_()
        value.head.bias.zero_()
    editor = Editor(model, tokenizer, value.eval(), backend_config(config).inference)
    bundles = []
    width = config["feedback"]["batch_size"]
    for start in range(0, len(plans), width):
        cache = out / "batches" / f"{start:06d}.json"
        if cache.exists():
            produced = read_json(cache)
        else:
            indices = range(start, min(start + width, len(plans)))
            refined = [read_json(out / "refined" / f"{i:06d}.json") for i in indices]
            edited, _ = editor.edit(plans[start : start + width], refined)
            produced = [
                {"plan": plans[i], "G": read_json(out / "raw" / f"{i:06d}.json"), "F": f, "E": e}
                for i, f, e in zip(indices, refined, edited)
            ]
            write_json(cache, produced)
        bundles.extend(produced)
        print({"stage": "feedback-collect", "completed": len(bundles), "requests": len(plans)}, flush=True)
    write_rows(out / "bundles.jsonl", bundles)
    return {"requests": len(bundles), "bundles": str(out / "bundles.jsonl")}


def label(config):
    from ..evaluation.hull import query
    from ..evaluation.workflow import evaluate, ensure_chgnet
    from ..evaluation.physics import label_records

    root = run_root(config)
    out = root / "feedback/labels"
    bundles = read_rows(root / "feedback/collection/bundles.jsonl")
    groups = {"current": [b["F"]["record"] for b in bundles]}
    for rank in range(config["feedback"]["candidates"]):
        groups[f"candidate_{rank}"] = [
            next(
                (c["record"] for c in b["E"]["candidates"] if c["rank"] == rank),
                dict(b["F"]["record"], success=False, reason="candidate_not_generated"),
            )
            for b in bundles
        ]
    query(groups["current"], root / "hull", batch_size=config["evaluation"]["hull_batch_size"])
    flat = [r for records in groups.values() for r in records]
    labels = label_records(
        flat,
        ensure_chgnet(config),
        out / "physics",
        devices=config["runtime"]["devices"],
        workers_per_device=config["runtime"]["physics_workers_per_device"],
        cache=root / "cache/physics",
        protocol={
            "max_steps": config["evaluation"]["relaxation_steps"],
            "fmax": config["evaluation"]["fmax"],
            "stress_tolerance_GPa": config["evaluation"]["stress_tolerance_GPa"],
        },
    )
    offset = 0
    for name, records in groups.items():
        scores, _ = evaluate(
            config, records, out / name, labels=labels[offset : offset + len(records)], training=True
        )
        write_rows(out / f"{name}.jsonl", scores)
        offset += len(records)
    return {"sources": len(bundles), "physical_endpoints": len(flat)}


def compile_data(config):
    from .feedback import compile_sources
    from .teacher import build

    root = run_root(config)
    bundles, scores = load_feedback_inputs(config)
    editor, value = compile_sources(
        [b["plan"] for b in bundles],
        [b["F"] for b in bundles],
        [b["E"] for b in bundles],
        scores,
    )
    write_rows(root / "feedback/data/reconstruction.jsonl", editor)
    write_rows(root / "feedback/data/verifier.jsonl", value)
    recipe = config["feedback"]["training"]
    light, report = build(
        editor,
        bundles,
        scores,
        seed=recipe["seed"],
        beta=recipe.get("refit_beta", 0.1),
        fraction=recipe.get("refit_fraction", 0.25),
    )
    write_rows(root / "feedback/data/refit.jsonl", light)
    write_json(root / "feedback/data/teacher.json", report)
    return {"editor_sources": len(editor), "value_pairs": len(value), "refit": report}


def fit_risk(config):
    from .risk import features, target, fit_risk

    root = run_root(config)
    bundles, scores = load_feedback_inputs(config)
    x = []
    y = []
    sources = []
    for i, bundle in enumerate(bundles):
        before = bundle["F"]["record"]
        source = bundle["plan"]["source_id"]
        endpoints = [(before, [], scores["current"][i])] + [
            (c["record"], c["trace"]["action"]["positions"], scores[f"candidate_{c['rank']}"][i])
            for c in bundle["E"]["candidates"]
            if c["commit"]["applied"]
        ]
        for after, positions, score in endpoints:
            outcome = target(score)
            if outcome is not None:
                x.append(features(before, after, positions))
                y.append(outcome)
                sources.append(source)
    model = fit_risk(x, y, sources, seed=config["feedback"]["training"]["risk_seed"])
    write_json(root / "feedback/risk/model.json", model)
    return {"rows": len(y), "sources": len(set(sources)), "metrics": model["metrics"]}


def stage(config, name, *, device="cuda:0", resume=False):
    root = run_root(config)
    cfg = backend_config(config)
    if name == "warmup":
        from .warmup import train

        return train(config, device=device, resume=resume)
    if name == "collect":
        return collect(config, device=device)
    if name == "label":
        return label(config)
    if name == "compile":
        return compile_data(config)
    if name == "risk":
        return fit_risk(config)
    if name in ("reconstruction", "refit", "verifier"):
        from ..runtime.device import setup_device

        device = setup_device(device, threads=config["runtime"]["threads"])
    if name == "verifier":
        from .value_training import train_value

        target = train_value(
            cfg,
            root / "feedback/data/verifier.jsonl",
            root / "feedback/verifier",
            editor_checkpoint=str(root / "feedback/refit/checkpoint"),
            device=device,
        )
        return {"checkpoint": str(target)}
    from .training import train_actor

    if name == "reconstruction":
        target = train_actor(
            "E",
            cfg,
            root / "feedback/data/reconstruction.jsonl",
            root / "feedback/reconstruction",
            checkpoint=str(root / "feedback/warmup/checkpoint"),
            device=device,
            resume=resume,
        )
    elif name == "refit":
        cfg.training.epochs = config["feedback"]["training"]["refit_epochs"]
        cfg.training.editor_content_lr = config["feedback"]["training"]["refit_content_lr"]
        target = train_actor(
            "E",
            cfg,
            root / "feedback/data/refit.jsonl",
            root / "feedback/refit",
            checkpoint=str(root / "feedback/reconstruction/checkpoint"),
            device=device,
            parameter_scope="lora_heads",
            resume=resume,
        )
    else:
        raise ValueError(f"Unknown FeedbackReconstructor training stage: {name}")
    return {"checkpoint": str(target)}


def train(config, *, device="cuda:0", resume=False, only=None):
    stages = [only] if only else ["warmup", "collect", "label", "compile", "reconstruction", "refit", "verifier", "risk"]
    root = run_root(config)
    for name in stages:
        receipt = root / "feedback/stages" / f"{name}.json"
        if resume and receipt.exists():
            continue
        result = stage(config, name, device=device, resume=resume)
        write_json(receipt, result)
    return {"checkpoint": asset(config, "feedback"), "stages": stages}
