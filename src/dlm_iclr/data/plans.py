"""Plan presets, source identities, and composition-disjoint dataset preparation."""

from __future__ import annotations

from dlm_iclr.runtime.capacity import MAX_ATOMS, MIN_ATOMS
from collections import Counter
from functools import reduce
import math
from pathlib import Path
from dlm_iclr._core.fixed_slot import SYMBOL_TO_Z, FixedSlotConfig
from dlm_iclr._core.plan_schema import build_body_prompt
from dlm_iclr.runtime.io import fingerprint, read_rows, write_json, write_rows

PRESETS = ("mp20_default",)
PRESET_FILES = {name: f"plans/{name}.jsonl" for name in PRESETS}


def composition_key(plan):
    counts = Counter()
    for element, count in zip(plan["elements"], plan["counts"], strict=True):
        counts[element] += int(count)
    divisor = reduce(math.gcd, counts.values())
    return "|".join(f"{element}:{count // divisor}" for element, count in sorted(counts.items()))


def validate_plan(row):
    if row.get("body_eligible") is False:
        return row.get("ineligible_reason") or "planner_failure"
    plan = row.get("plan_state")
    if not isinstance(plan, dict):
        return "missing_plan_state"
    n, elements, counts = plan.get("N"), plan.get("elements"), plan.get("counts")
    if type(n) is not int or not MIN_ATOMS <= n <= MAX_ATOMS:
        return f"atom_count_outside_{MIN_ATOMS}_to_{MAX_ATOMS}"
    if not isinstance(elements, list) or not isinstance(counts, list) or len(elements) != len(counts):
        return "invalid_element_counts"
    if not elements or len(set(elements)) != len(elements):
        return "empty_or_duplicate_elements"
    if any(
        element not in SYMBOL_TO_Z or SYMBOL_TO_Z[element] > FixedSlotConfig().max_atomic_number
        for element in elements
    ):
        return "element_outside_trained_vocabulary"
    if any(type(count) is not int or count < 1 for count in counts) or sum(counts) != n:
        return "counts_do_not_match_N"
    if plan.get("rich_field_valid") is False or plan.get("plan_end_marker_present") is False:
        return "invalid_rich_plan"
    required = {
        "anion_framework",
        "charge_bucket",
        "lattice_system",
        "spacegroup_bucket",
        "volume_per_atom_bin",
    }
    if not required.issubset(plan):
        return "missing_rich_plan_fields:" + ",".join(sorted(required - set(plan)))
    expected = build_body_prompt(plan).rstrip() + "\n"
    if row.get("body_prompt") is not None and row["body_prompt"] != expected:
        return "body_prompt_differs_from_plan"
    return None


def axis_schedule(plan):
    n = plan["N"]
    result = [[0, *[7 + 4 * site for site in range(n)]], [1, 2, 3, 4, 5, 6]]
    groups, offset = [], 0
    for count in plan["counts"]:
        groups.append(range(offset, offset + count))
        offset += count
    for axis in range(3):
        result.extend([[8 + 4 * site + axis for site in group] for group in groups])
    return result


def load_plans(source, *, requests=None, legal_only=False, seed=17):
    source = str(source)
    if source.startswith("preset:"):
        source = source[7:]
    path = (
        Path(__file__).parent / "presets" / PRESET_FILES[source] if source in PRESET_FILES else Path(source)
    )
    rows = read_rows(path)
    normalized, excluded, seen = [], [], set()
    for index, original in enumerate(rows):
        row = dict(original)
        row.setdefault("source_id", row.get("ancestor_id", f"file:{fingerprint(original)[:20]}"))
        if row["source_id"] in seen:
            raise ValueError(f"Duplicate source_id: {row['source_id']}")
        seen.add(row["source_id"])
        row.setdefault("original_ordinal", index)
        reason = validate_plan(row)
        row.update(body_eligible=reason is None, ineligible_reason=reason)
        if reason is None:
            row["body_prompt"] = (
                row.get("body_prompt") or build_body_prompt(row["plan_state"]).rstrip() + "\n"
            )
        for name in ("body_noise_seed", "refiner_noise_seed"):
            row.setdefault(name, int(fingerprint([seed, row["source_id"], name])[:16], 16) & ((1 << 63) - 1))
            if type(row[name]) is not int or not 0 <= row[name] < 2**63:
                raise ValueError(f"{name} must be an exact nonnegative 63-bit integer")
        if reason and legal_only:
            excluded.append(
                {"source_id": row["source_id"], "original_ordinal": row["original_ordinal"], "reason": reason}
            )
        else:
            normalized.append(row)
    if requests is not None:
        if not 1 <= requests <= len(normalized):
            raise ValueError(f"Requested {requests} Plans; source contains {len(normalized)} eligible rows")
        normalized = normalized[:requests]
    for ordinal, row in enumerate(normalized):
        row["ordinal"] = ordinal
    return normalized, {
        "source": source,
        "source_rows": len(rows),
        "requests": len(normalized),
        "legal_only": legal_only,
        "excluded_invalid_plans": excluded,
        "selection": "original_order_before_structure_generation",
    }
