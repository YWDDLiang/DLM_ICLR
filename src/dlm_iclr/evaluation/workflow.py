"""Direct, stability and ordered set metrics for saved crystal records."""

import math
from pathlib import Path
from ..runtime.config import asset, run_root, backend_config
from ..runtime.io import write_json, write_rows


def qualify(scores, *, stable_threshold=0.0, metastable_threshold=0.1):
    from .._core.exact_sun_nu import conjunction

    result = []
    for score in scores:
        row = dict(score)
        if row["terminal_status"] in ("generation_failure", "invalid_raw", "invalid_terminal"):
            stable = meta = False
        elif row["terminal_verified"] and all(
            isinstance(row.get(k), (float, int)) and math.isfinite(row[k])
            for k in ("e_above_hull_eV_atom", "terminal_energy_eV_atom", "hull_energy_eV_atom")
        ):
            stable = row["e_above_hull_eV_atom"] <= stable_threshold
            meta = row["e_above_hull_eV_atom"] <= metastable_threshold
        else:
            stable = meta = None
        novel, unique = row.get("novel"), row.get("unique_representative")
        row.update(
            strict_stable=stable,
            meta_stable=meta,
            strict_sun=conjunction(stable, novel, unique),
            meta_sun=conjunction(meta, novel, unique),
        )
        result.append(row)
    return result


def summary(scores):
    from .._core.exact_sun_nu import conjunction

    fields = {
        "Stable": "strict_stable",
        "MetaStable": "meta_stable",
        "SUN": "strict_sun",
        "MSUN": "meta_sun",
        "V": "valid",
        "U": "unique_representative",
        "N": "novel",
        "VUN": "vun",
        "comp_valid": "comp_valid",
        "struct_valid": "struct_valid",
    }
    for row in scores:
        row["valid"] = conjunction(row["comp_valid"], row["struct_valid"])
        row["vun"] = conjunction(row["valid"], row.get("novel"), row.get("unique_representative"))
    metrics = {}
    for label, field in fields.items():
        passed = sum(r.get(field) is True for r in scores)
        pending = sum(r.get(field) is None for r in scores)
        metrics[label] = {
            "passed": passed,
            "pending": pending,
            "requests": len(scores),
            "count_bounds": [passed, passed + pending],
            "percent_bounds": [100 * passed / len(scores), 100 * (passed + pending) / len(scores)],
        }
    return {"requests": len(scores), "metrics": metrics}


def ensure_chgnet(config):
    checkpoint = asset(config, "chgnet")
    if checkpoint:
        return checkpoint
    root = run_root(config) / "assets/chgnet_0.3.0.pt"
    if not root.exists():
        import torch
        from chgnet.model.model import CHGNet

        root.parent.mkdir(parents=True, exist_ok=True)
        model = CHGNet.load(model_name="0.3.0", use_device="cpu")
        torch.save({"model": model.as_dict()}, root)
    return str(root)


def evaluate(config, records, output, *, labels=None, training=False):
    from .inputs import normalize_records
    from .physics import label_records
    from .sun import score_records
    from .composition import composition_validity
    from .._core.continuous_keep_edit import structure_of

    records = normalize_records(records)
    root = run_root(config)
    output = Path(output)
    cfg = backend_config(config)
    if labels is None:
        labels = label_records(
            records,
            ensure_chgnet(config),
            output / "physics",
            devices=config["runtime"]["devices"],
            workers_per_device=config["runtime"]["physics_workers_per_device"],
            cache=root / "cache/physics",
            protocol={
                "max_steps": config["evaluation"]["relaxation_steps"],
                "fmax": config["evaluation"]["fmax"],
                "stress_tolerance_GPa": config["evaluation"]["stress_tolerance_GPa"],
            },
        )
    scores, report = score_records(
        records,
        labels,
        cfg.assets.hull_cache,
        cfg.assets.novelty_reference,
        output / "scoring",
        cache=root / "cache/matching",
        workers=config["runtime"]["matching_workers"],
        all_predicates=not training,
    )
    if training:
        return scores, report
    reference = (report.get("novelty_evaluation") or {}).get("training_identity", {})
    if reference.get("parse_failures"):
        for row in scores:
            if row.get("novel") is True:
                row["novel"] = None
    scores = qualify(
        scores,
        stable_threshold=config["evaluation"]["stable_threshold"],
        metastable_threshold=config["evaluation"]["metastable_threshold"],
    )
    for row, record in zip(scores, records):
        if row["reconstructed"] and config["evaluation"]["composition"] == "smact3_mixed":
            row["comp_valid"] = composition_validity(structure_of(record).composition.formula)["comp_valid"]
    result = summary(scores)
    result["protocol"] = config["evaluation"]
    write_rows(output / "scores.jsonl", scores)
    write_json(output / "summary.json", result)
    return scores, result
