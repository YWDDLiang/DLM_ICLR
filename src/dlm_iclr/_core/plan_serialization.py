from __future__ import annotations

from dlm_iclr.runtime.capacity import MAX_ATOMS
from collections import Counter
import re
from typing import Any, Dict, Mapping, Sequence
from dlm_iclr._core.fixed_slot import SYMBOL_TO_Z
from dlm_iclr._core.plan_schema import (
    ALLOWED_LATTICE_SYSTEMS,
    ALLOWED_SPACEGROUP_BUCKETS,
    CHARGE_BUCKET_TO_CODE,
    PLAN_STATE_VERSION,
    anion_framework_from_symbols,
    prototype_key,
)

PLAN_FORMAT = "formula_text"
FORMULA_END_PLAN_FORMAT = "formula_end_v1"
SEMANTIC_PLAN_FORMAT = "semantic_formula_v1"
RICH_PLAN_FORMAT = "rich_plan_v1"
PLAN_STYLES = (
    PLAN_FORMAT,
    FORMULA_END_PLAN_FORMAT,
    SEMANTIC_PLAN_FORMAT,
    RICH_PLAN_FORMAT,
)
PLAN_FIELDS = ("formula",)
RICH_PLAN_FIELDS = ("formula", "anion", "charge", "lattice", "spacegroup", "volume")
PLAN_END_FIELD = "end"
PLAN_END_VALUE = "plan"
RICH_ANION_FRAMEWORKS = {
    "oxide",
    "sulfide",
    "chalcogenide",
    "halide",
    "nitride",
    "phosphide_or_phosphate",
    "other",
}
METAL_SYMBOLS = {
    "Li",
    "Be",
    "Na",
    "Mg",
    "Al",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
}
HALOGEN_SYMBOLS = {"F", "Cl", "Br", "I"}
CHALCOGEN_SYMBOLS = {"S", "Se", "Te"}
PNICTIDE_SYMBOLS = {"N", "P", "As", "Sb", "Bi"}
CARBIDE_BORIDE_SYMBOLS = {"B", "C", "Si", "Ge"}


def normalize_plan_style(plan_style: str | None = None) -> str:
    style = PLAN_FORMAT if plan_style is None else str(plan_style).strip()
    if style not in PLAN_STYLES:
        raise ValueError(f"unknown formula plan style {style!r}; expected one of {PLAN_STYLES}")
    return style


def _canonical_symbol_counts(symbols: Sequence[str], counts: Sequence[int]) -> tuple[list[str], list[int]]:
    counter: Counter[str] = Counter()
    for symbol, count in zip(symbols, counts):
        symbol = str(symbol).strip()
        if symbol not in SYMBOL_TO_Z:
            raise ValueError(f"unsupported element symbol {symbol!r}")
        count_value = int(count)
        if count_value <= 0:
            raise ValueError(f"element count for {symbol} must be positive, got {count_value}")
        counter[symbol] += count_value
    ordered = sorted(counter, key=lambda item: SYMBOL_TO_Z[item])
    return (ordered, [int(counter[symbol]) for symbol in ordered])


def formula_from_symbol_counts(symbols: Sequence[str], counts: Sequence[int]) -> str:
    parts: list[str] = []
    for symbol, count in zip(symbols, counts):
        count = int(count)
        parts.append(str(symbol) if count == 1 else f"{symbol}{count}")
    return "".join(parts)


def symbol_counts_from_formula(formula: str) -> tuple[list[str], list[int]]:
    """Parse a flat integer-count formula and canonicalize it by atomic number."""
    compact = re.sub("\\s+", "", str(formula))
    if not compact:
        raise ValueError("composition plan formula is empty")
    tokens = re.findall("([A-Z][a-z]?)(\\d*)", compact)
    if not tokens:
        raise ValueError(f"composition plan formula {compact!r} contains no element symbols")
    reconstructed = "".join((symbol + count for symbol, count in tokens))
    if reconstructed != compact:
        raise ValueError(f"composition plan formula {compact!r} is not a flat integer-count formula")
    symbols: list[str] = []
    counts: list[int] = []
    for symbol, count_text in tokens:
        count = int(count_text) if count_text else 1
        symbols.append(symbol)
        counts.append(count)
    return _canonical_symbol_counts(symbols, counts)


def composition_plan_from_state(plan_state: Mapping[str, Any]) -> Dict[str, Any]:
    symbols = [str(symbol) for symbol in plan_state.get("elements") or []]
    counts = [int(value) for value in plan_state.get("counts") or []]
    symbols, counts = _canonical_symbol_counts(symbols, counts)
    num_atoms = int(sum(counts))
    formula = formula_from_symbol_counts(symbols, counts)
    if int(plan_state.get("N", num_atoms)) != num_atoms:
        raise ValueError(f"plan_state N {plan_state.get('N')} does not match counts sum {num_atoms}")
    return {"formula": formula, "elements": symbols, "counts": counts, "N": num_atoms}


def arity_label(num_elements: int) -> str:
    if num_elements <= 1:
        return "unary"
    if num_elements == 2:
        return "binary"
    if num_elements == 3:
        return "ternary"
    if num_elements == 4:
        return "quaternary"
    return "multi"


def size_label(num_atoms: int) -> str:
    atoms = int(num_atoms)
    if atoms <= 3:
        return "tiny"
    if atoms <= 6:
        return "small"
    if atoms <= 10:
        return "medium"
    if atoms <= 16:
        return "large"
    return "xlarge"


def family_label(elements: Sequence[str]) -> str:
    symbols = {str(symbol) for symbol in elements}
    if not symbols:
        return "other"
    if all((symbol in METAL_SYMBOLS for symbol in symbols)):
        return "intermetallic"
    has_oxygen = "O" in symbols
    has_halogen = bool(symbols & HALOGEN_SYMBOLS)
    has_chalcogen = bool(symbols & CHALCOGEN_SYMBOLS)
    has_pnictide = bool(symbols & PNICTIDE_SYMBOLS)
    has_carbide_boride = bool(symbols & CARBIDE_BORIDE_SYMBOLS)
    group_count = sum(
        (bool(value) for value in (has_oxygen, has_halogen, has_chalcogen, has_pnictide, has_carbide_boride))
    )
    if has_oxygen and group_count == 1:
        return "oxide"
    if has_oxygen and has_halogen and (group_count == 2):
        return "oxyhalide"
    if has_oxygen and has_chalcogen and (group_count == 2):
        return "oxychalcogenide"
    if has_oxygen and group_count > 1:
        return "mixed_anion"
    if has_halogen and group_count == 1:
        return "halide"
    if has_chalcogen and group_count == 1:
        return "chalcogenide"
    if has_pnictide and group_count == 1:
        return "pnictide"
    if has_carbide_boride and group_count == 1:
        return "carbide_boride"
    if group_count > 1:
        return "mixed_anion"
    return "other"


def semantic_fields_from_plan(plan_state: Mapping[str, Any]) -> Dict[str, str]:
    plan = composition_plan_from_state(plan_state)
    return {
        "family": family_label(plan["elements"]),
        "arity": arity_label(len(plan["elements"])),
        "size": size_label(int(plan["N"])),
    }


def semantic_consistency_from_plan(plan_state: Mapping[str, Any]) -> Dict[str, bool | None]:
    expected = semantic_fields_from_plan(plan_state)
    result: Dict[str, bool | None] = {}
    for key in ("family", "arity", "size"):
        generated = plan_state.get(f"generated_{key}")
        result[f"{key}_match_formula"] = None if generated is None else str(generated) == expected[key]
    return result


def _normalize_rich_anion(value: Any) -> str:
    normalized = _normalize_generated_label(str(value))
    if normalized not in RICH_ANION_FRAMEWORKS:
        raise ValueError(f"invalid rich-plan anion field {value!r}")
    return normalized


def _normalize_rich_charge(value: Any) -> str:
    normalized = _normalize_generated_label(str(value))
    if normalized not in CHARGE_BUCKET_TO_CODE:
        raise ValueError(f"invalid rich-plan charge field {value!r}")
    return normalized


def _normalize_rich_lattice(value: Any) -> str:
    normalized = _normalize_generated_label(str(value))
    if normalized not in ALLOWED_LATTICE_SYSTEMS:
        raise ValueError(f"invalid rich-plan lattice field {value!r}")
    return normalized


def _normalize_rich_spacegroup(value: Any) -> str:
    normalized = _normalize_generated_label(str(value))
    if normalized not in ALLOWED_SPACEGROUP_BUCKETS:
        raise ValueError(f"invalid rich-plan spacegroup field {value!r}")
    return normalized


def _normalize_rich_volume(value: Any) -> str:
    normalized = _normalize_generated_label(str(value))
    if not re.fullmatch("volpa_\\d{3}_\\d{3}", normalized):
        raise ValueError(f"invalid rich-plan volume field {value!r}")
    return normalized


def rich_fields_from_plan_state(plan_state: Mapping[str, Any]) -> Dict[str, str]:
    return {
        "anion": _normalize_rich_anion(plan_state.get("anion_framework", "other")),
        "charge": _normalize_rich_charge(plan_state.get("charge_bucket", "validator_unavailable")),
        "lattice": _normalize_rich_lattice(plan_state.get("lattice_system", "triclinic")),
        "spacegroup": _normalize_rich_spacegroup(plan_state.get("spacegroup_bucket", "sg_001_002")),
        "volume": _normalize_rich_volume(plan_state.get("volume_per_atom_bin", "volpa_000_004")),
    }


def format_composition_plan(plan_state: Mapping[str, Any], *, plan_style: str | None = None) -> str:
    plan = composition_plan_from_state(plan_state)
    style = normalize_plan_style(plan_style)
    if style == FORMULA_END_PLAN_FORMAT:
        return f"formula: {plan['formula']}\n{PLAN_END_FIELD}: {PLAN_END_VALUE}"
    if style == SEMANTIC_PLAN_FORMAT:
        semantic = semantic_fields_from_plan(plan)
        return "\n".join(
            [
                f"family: {semantic['family']}",
                f"arity: {semantic['arity']}",
                f"size: {semantic['size']}",
                f"formula: {plan['formula']}",
            ]
        )
    if style == RICH_PLAN_FORMAT:
        rich = rich_fields_from_plan_state(plan_state)
        return "\n".join(
            [
                f"formula: {plan['formula']}",
                f"anion: {rich['anion']}",
                f"charge: {rich['charge']}",
                f"lattice: {rich['lattice']}",
                f"spacegroup: {rich['spacegroup']}",
                f"volume: {rich['volume']}",
                f"{PLAN_END_FIELD}: {PLAN_END_VALUE}",
            ]
        )
    return f"formula: {plan['formula']}"


def _strip_special_tail(text: str) -> str:
    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n")
    for marker in ("<|endoftext|>", "</s>", "<s>"):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0]
    return cleaned


def _normalize_generated_label(value: str) -> str:
    cleaned = str(value).split("<", 1)[0].strip().lower()
    match = re.match("([a-z][a-z0-9_+-]*)", cleaned)
    return match.group(1).replace("-", "_") if match else cleaned


def has_plan_end_marker(text: str) -> bool:
    cleaned = _strip_special_tail(text)
    return re.search(f"(?im)^\\s*{PLAN_END_FIELD}\\s*:\\s*{PLAN_END_VALUE}\\s*$", cleaned) is not None


def has_plan_tail_after_end_marker(text: str) -> bool:
    cleaned = _strip_special_tail(text)
    match = re.search(f"(?im)^\\s*{PLAN_END_FIELD}\\s*:\\s*{PLAN_END_VALUE}\\s*$", cleaned)
    if match is None:
        return False
    tail = cleaned[match.end() :].strip()
    if not tail:
        return False
    return bool(re.search("(?im)^\\s*body\\s*:\\s*$|<N_\\d{3}>|<LA_\\d{3}>|<E_[A-Z][a-z]?>", tail))


def parse_composition_plan(
    text: str, *, max_atoms: int = MAX_ATOMS, plan_style: str | None = None
) -> Dict[str, Any]:
    """Parse a text plan into a minimal plan_state dict.

    ``counts`` and ``N`` are intentionally derived by Python from ``formula``.
    This avoids making the model emit redundant arithmetic fields that can
    contradict each other during de novo sampling. DN4 semantic fields are
    diagnostics only; they never override the formula-derived composition. Planner
    rich fields are generated conditioning fields; they condition the body
    executor, but never override formula-derived composition.
    """
    cleaned = _strip_special_tail(text)
    requested_style = normalize_plan_style(plan_style) if plan_style is not None else None
    fields: Dict[str, str] = {}
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        if re.fullmatch("(?i)plan\\s*:", line):
            continue
        if re.fullmatch("(?i)body\\s*:", line):
            break
        match = re.match(
            "(?i)^(family|arity|size|formula|elements|counts|n|anion|charge|lattice|spacegroup|volume|end)\\s*:\\s*(.+?)\\s*$",
            line,
        )
        if match:
            key = match.group(1).lower()
            fields[key] = match.group(2).strip()
            if key == PLAN_END_FIELD and _normalize_generated_label(fields[key]) == PLAN_END_VALUE:
                break
            continue
        if fields and (not set(PLAN_FIELDS).issubset(fields)):
            continue
    missing = [field for field in PLAN_FIELDS if field.lower() not in fields]
    if missing:
        raise ValueError(f"composition plan missing fields: {','.join(missing)}")
    if requested_style == FORMULA_END_PLAN_FORMAT:
        marker_value = _normalize_generated_label(fields.get(PLAN_END_FIELD, ""))
        if marker_value != PLAN_END_VALUE:
            raise ValueError("composition plan missing required end: plan marker")
    if requested_style == RICH_PLAN_FORMAT:
        missing_rich = [field for field in RICH_PLAN_FIELDS if field not in fields]
        if missing_rich:
            raise ValueError(f"rich composition plan missing fields: {','.join(missing_rich)}")
        marker_value = _normalize_generated_label(fields.get(PLAN_END_FIELD, ""))
        if marker_value != PLAN_END_VALUE:
            raise ValueError("rich composition plan missing required end: plan marker")
    formula_value = fields["formula"].split("<", 1)[0].strip()
    if not formula_value:
        raise ValueError("composition plan formula is empty")
    generated_formula = formula_value.split()[0]
    symbols, counts = symbol_counts_from_formula(generated_formula)
    num_atoms = int(sum(counts))
    if not 1 <= num_atoms <= int(max_atoms):
        raise ValueError(f"composition plan N {num_atoms} outside 1..{max_atoms}")
    formula = formula_from_symbol_counts(symbols, counts)
    has_rich_fields = any((key in fields for key in ("anion", "charge", "lattice", "spacegroup", "volume")))
    inferred_style = (
        FORMULA_END_PLAN_FORMAT
        if _normalize_generated_label(fields.get(PLAN_END_FIELD, "")) == PLAN_END_VALUE
        and (not has_rich_fields)
        else RICH_PLAN_FORMAT
        if has_rich_fields
        else SEMANTIC_PLAN_FORMAT
        if any((key in fields for key in ("family", "arity", "size")))
        else PLAN_FORMAT
    )
    output_style = requested_style or inferred_style
    plan: Dict[str, Any] = {
        "plan_state_version": PLAN_STATE_VERSION,
        "N": num_atoms,
        "elements": symbols,
        "counts": counts,
        "formula": formula,
        "reduced_formula": formula,
        "charge_bucket": "unknown",
        "oxidation_candidates": "unknown",
        "anion_framework": "unknown",
        "lattice_system": "unknown",
        "spacegroup_bucket": "sg_unknown",
        "volume_per_atom_bin": "volpa_unknown",
        "prototype_key": f"formula={formula}|N={num_atoms}",
        "plan_format": output_style,
        "plan_end_marker_present": _normalize_generated_label(fields.get(PLAN_END_FIELD, ""))
        == PLAN_END_VALUE,
        "derived_counts_from_formula": True,
        "derived_n_from_formula": True,
    }
    expected = semantic_fields_from_plan(plan)
    plan.update(expected)
    if output_style == RICH_PLAN_FORMAT or has_rich_fields:
        generated_rich = {
            "anion": _normalize_rich_anion(fields.get("anion", "")),
            "charge": _normalize_rich_charge(fields.get("charge", "")),
            "lattice": _normalize_rich_lattice(fields.get("lattice", "")),
            "spacegroup": _normalize_rich_spacegroup(fields.get("spacegroup", "")),
            "volume": _normalize_rich_volume(fields.get("volume", "")),
        }
        expected_anion = anion_framework_from_symbols(symbols)
        plan.update(
            {
                "anion_framework": generated_rich["anion"],
                "charge_bucket": generated_rich["charge"],
                "lattice_system": generated_rich["lattice"],
                "spacegroup_bucket": generated_rich["spacegroup"],
                "volume_per_atom_bin": generated_rich["volume"],
                "generated_rich_fields": dict(generated_rich),
                "expected_anion_framework": expected_anion,
                "anion_match_formula": generated_rich["anion"] == expected_anion,
                "rich_field_valid": True,
            }
        )
        plan["prototype_key"] = prototype_key(plan)
    generated_semantic: Dict[str, str] = {}
    for key in ("family", "arity", "size"):
        if key in fields:
            generated = _normalize_generated_label(fields[key])
            generated_semantic[key] = generated
            plan[f"generated_{key}"] = generated
            plan[f"expected_{key}"] = expected[key]
            plan[f"{key}_match_formula"] = generated == expected[key]
    plan["generated_semantic_fields"] = generated_semantic
    plan["semantic_consistency"] = semantic_consistency_from_plan(plan)
    return plan
