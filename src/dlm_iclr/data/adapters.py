"""Convert CSV, JSONL or CIF directories through a single crystal interface."""

import csv
from pathlib import Path
from .plans import composition_key
from ..runtime.io import fingerprint, read_rows, write_rows, write_json
from ..runtime.config import path, run_root, asset


def iter_structures(source, fields=None):
    source = Path(source)
    fields = fields or {}
    if source.is_dir():
        rows = ({"id": p.stem, "cif": p.read_text(encoding="utf-8")} for p in sorted(source.glob("*.cif")))
    elif source.suffix.lower() == ".csv":
        with source.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = read_rows(source)
    for index, original in enumerate(rows):
        row = dict(original)
        row.update({key: original[value] for key, value in fields.items() if value in original})
        yield index, str(row.get("id") or row.get("material_id") or index), row


def align_sites(arrays, plan):
    order = [
        i for element in plan["elements"] for i, species in enumerate(arrays["species"]) if species == element
    ]
    return dict(
        arrays,
        species=[arrays["species"][i] for i in order],
        frac_coords=[arrays["frac_coords"][i] for i in order],
    ), order


def prepare(config):
    from transformers import AutoTokenizer
    from pymatgen.core import Structure
    from .._core.dynamic_crystal import (
        build_special_tokens,
        structure_to_dynamic_answer,
        parse_dynamic_answer,
        arrays_to_dynamic_answer,
    )
    from .._core.fixed_slot import metadata_from_csv_row
    from .._core.r5_plan_state import plan_state_from_arrays, build_body_prompt
    from ..planner.prepare import build_records_for_plan

    root = run_root(config) / "data"
    tokenizer = AutoTokenizer.from_pretrained(asset(config, "dlm"), trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": build_special_tokens()})
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.save_pretrained(root / "tokenizer")
    planner_tokenizer = AutoTokenizer.from_pretrained(asset(config, "planner_base"), trust_remote_code=True)
    dataset = config["dataset"]
    statistics = {"dataset": dataset["name"], "splits": {}}
    max_prompt = max_answer = 0
    for split, source in dataset["splits"].items():
        structures, bodies, planner_rows, plans, failures = [], [], [], [], []
        for index, identifier, row in iter_structures(path(config, source), dataset["fields"]):
            source_id = f"{dataset['name']}:{split}:{identifier}:{index}"
            try:
                crystal = (
                    Structure.from_dict(row["structure"])
                    if row.get("structure")
                    else Structure.from_str(row["cif"], fmt="cif")
                )
                if len(crystal) > dataset["max_atoms"]:
                    raise ValueError("Structure exceeds configured atom count")
                answer, diagnostic = structure_to_dynamic_answer(crystal)
                if diagnostic.length_clips or diagnostic.angle_clips or diagnostic.coord_clips:
                    raise ValueError("Structure exceeds the coordinate vocabulary")
                arrays = parse_dynamic_answer(answer, strict=True)
                metadata = metadata_from_csv_row(row)
                plan = plan_state_from_arrays(arrays, metadata=metadata)
                # Planner conditions use precisely the original quantized source view.
                planner_rows.extend(
                    build_records_for_plan(
                        split=split,
                        row_idx=index,
                        plan_state=plan,
                        metadata=metadata,
                        tokenizer=planner_tokenizer,
                        prompt_style="h1_rich_plan_v1",
                        include_sample_id=False,
                        sample_types=["direct_plan"],
                        weights={"direct_plan": 1.0},
                    )
                )
                arrays, order = align_sites(arrays, plan)
                answer, _ = arrays_to_dynamic_answer(
                    arrays["lengths"], arrays["angles"], arrays["species"], arrays["frac_coords"]
                )
                prompt = build_body_prompt(plan).rstrip() + "\n"
                ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
                prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                provenance = {
                    "dataset_origin": dataset["name"],
                    "original_split": split,
                    "material_id": identifier,
                    "source_row_index": index,
                    "site_permutation": order,
                    "usage_role": "train" if split == "train" else "evaluation",
                }
                structures.append(
                    {
                        "source_id": source_id,
                        "dataset": dataset["name"],
                        "split": split,
                        "structure": crystal.as_dict(),
                        "body_prompt": prompt,
                        "body_token_ids": ids,
                        "plan_state": plan,
                        "provenance": provenance,
                    }
                )
                bodies.append(
                    {
                        "source_id": source_id,
                        "source_split": split,
                        "prompt": build_body_prompt(plan),
                        "answer": answer,
                        "num_atoms": len(crystal),
                        "sample_weight": 1.0,
                    }
                )
                planned = {
                    "schema": "crystal_plan_v1",
                    "source_id": source_id,
                    "ordinal": len(plans),
                    "original_ordinal": index,
                    "body_eligible": True,
                    "body_prompt": prompt,
                    "plan_state": plan,
                    "provenance": provenance,
                }
                for field, stage in [("body_noise_seed", "G"), ("refiner_noise_seed", "F")]:
                    planned[field] = int(fingerprint([config["b0"]["seed"], source_id, stage])[:15], 16)
                plans.append(planned)
                max_prompt, max_answer = max(max_prompt, len(prefix)), max(max_answer, len(ids))
            except (ValueError, KeyError, TypeError) as error:
                failures.append({"source_id": source_id, "reason": str(error)})
            if (index + 1) % 1000 == 0:
                print({"split": split, "processed": index + 1, "prepared": len(structures)}, flush=True)
        for folder, values in [
            ("structures", structures),
            ("b0", bodies),
            ("planner", planner_rows),
            ("plans", plans),
            ("failures", failures),
        ]:
            write_rows(root / folder / f"{split}.jsonl", values)
        statistics["splits"][split] = {
            "prepared": len(structures),
            "failed": len(failures),
            "compositions": len({composition_key(p["plan_state"]) for p in plans}),
        }
    statistics["max_length"] = min(768, max(256, max_prompt + max_answer + 48))
    statistics["observed_max_length"] = max_prompt + max_answer
    write_json(root / "statistics.json", statistics)
    return statistics
