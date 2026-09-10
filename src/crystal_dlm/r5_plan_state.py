from __future__ import annotations

"Retained crystal DLM implementation; see docs/method.md for the public workflow."
import json
from typing import Any, Dict, Mapping, Sequence

PLAN_STATE_FIELDS = [
    "N",
    "elements",
    "counts",
    "formula",
    "reduced_formula",
    "charge_bucket",
    "oxidation_candidates",
    "anion_framework",
    "lattice_system",
    "spacegroup_bucket",
    "volume_per_atom_bin",
    "prototype_key",
]


def canonical_plan_state(plan: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: plan.get(key) for key in PLAN_STATE_FIELDS}


def plan_state_to_json(plan: Mapping[str, Any], *, canonical_only: bool = True) -> str:
    payload = canonical_plan_state(plan) if canonical_only else dict(plan)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_body_prompt(plan: Mapping[str, Any]) -> str:
    return f"Generate only the exact-length dynamic crystal body for this fixed plan_state. The first token must match N and the element multiset must match elements/counts.\nplan_state: {plan_state_to_json(plan)}\ndynamic_crystal_body:"


from collections import Counter
from dataclasses import dataclass
import math
from crystal_dlm.composition_validity import classify_smact_validity, element_symbols, reduced_composition
from crystal_dlm.fixed_slot import SYMBOL_TO_Z

PLAN_STATE_VERSION = "r5_plan_state_v1"
ALLOWED_LATTICE_SYSTEMS = {
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
}
ALLOWED_SPACEGROUP_BUCKETS = {
    "sg_001_002",
    "sg_003_015",
    "sg_016_074",
    "sg_075_142",
    "sg_143_167",
    "sg_168_194",
    "sg_195_230",
}
CHARGE_BUCKET_TO_CODE = {
    "neutral_plausible": "B_NEU",
    "single_element": "B_ONE",
    "all_metal": "B_MET",
    "charge_fail": "B_CHF",
    "pauling_fail": "B_PAU",
    "oxidation_missing": "B_OXM",
    "validator_unavailable": "B_UNK",
}


@dataclass(frozen=True)
class PlanValidation:
    valid_N: bool
    valid_formula: bool
    valid_counts: bool
    valid_elements: bool
    valid_generated_N: bool = True

    @property
    def valid(self) -> bool:
        return (
            self.valid_N
            and self.valid_generated_N
            and self.valid_formula
            and self.valid_counts
            and self.valid_elements
        )

    def to_dict(self) -> Dict[str, bool]:
        return {
            "valid": self.valid,
            "valid_N": self.valid_N,
            "valid_generated_N": self.valid_generated_N,
            "valid_formula": self.valid_formula,
            "valid_counts": self.valid_counts,
            "valid_elements": self.valid_elements,
        }


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _formula_from_symbols(symbols: Sequence[str], counts: Sequence[int]) -> str:
    parts: list[str] = []
    for symbol, count in zip(symbols, counts):
        count = int(count)
        parts.append(str(symbol) if count == 1 else f"{symbol}{count}")
    return "".join(parts)


def _full_composition_from_species(species: Sequence[str]) -> tuple[list[str], list[int]]:
    counter = Counter((str(symbol) for symbol in species))
    symbols = sorted(counter, key=lambda symbol: SYMBOL_TO_Z.get(symbol, 10000))
    return (symbols, [int(counter[symbol]) for symbol in symbols])


def _safe_smact_classification(symbols: Sequence[str], counts: Sequence[int]) -> Dict[str, Any]:
    elems = tuple((SYMBOL_TO_Z[symbol] for symbol in symbols if symbol in SYMBOL_TO_Z))
    if len(elems) != len(symbols):
        return {"valid": False, "reason": "unsupported_element"}
    try:
        return classify_smact_validity(elems, tuple((int(value) for value in counts)))
    except Exception as exc:
        return {"valid": None, "reason": "validator_unavailable", "validator_error": type(exc).__name__}


def charge_bucket_from_classification(classification: Mapping[str, Any]) -> str:
    reason = str(classification.get("reason", "unknown"))
    valid = classification.get("valid")
    if valid is True and reason == "charge_neutral_pauling_valid":
        return "neutral_plausible"
    if valid is True and reason == "single_element_shortcut":
        return "single_element"
    if valid is True and reason == "all_metal_shortcut":
        return "all_metal"
    if reason == "charge_neutrality_fail":
        return "charge_fail"
    if reason == "pauling_fail_or_ratio_rejected":
        return "pauling_fail"
    if reason == "oxidation_state_missing":
        return "oxidation_missing"
    if reason == "validator_unavailable":
        return "validator_unavailable"
    return reason


def anion_framework_from_symbols(symbols: Sequence[str]) -> str:
    symbol_set = set(symbols)
    if "O" in symbol_set:
        return "oxide"
    if "S" in symbol_set:
        return "sulfide"
    if "Se" in symbol_set or "Te" in symbol_set:
        return "chalcogenide"
    if symbol_set.intersection({"F", "Cl", "Br", "I"}):
        return "halide"
    if "N" in symbol_set:
        return "nitride"
    if "P" in symbol_set:
        return "phosphide_or_phosphate"
    return "other"


def lattice_system_from_lattice(lengths: Sequence[float], angles: Sequence[float], tol: float = 0.01) -> str:
    a, b, c = [float(value) for value in lengths]
    alpha, beta, gamma = [float(value) for value in angles]
    eq_ab = abs(a - b) <= tol
    eq_bc = abs(b - c) <= tol
    right = all((abs(value - 90.0) <= tol for value in (alpha, beta, gamma)))
    if eq_ab and eq_bc and right:
        return "cubic"
    if eq_ab and (not eq_bc) and right:
        return "tetragonal"
    if right:
        return "orthorhombic"
    if eq_ab and abs(alpha - 90.0) <= tol and (abs(beta - 90.0) <= tol) and (abs(gamma - 120.0) <= tol):
        return "hexagonal"
    if eq_ab and eq_bc and (abs(alpha - beta) <= tol) and (abs(beta - gamma) <= tol):
        return "trigonal"
    if sum((abs(value - 90.0) <= tol for value in (alpha, beta, gamma))) == 2:
        return "monoclinic"
    return "triclinic"


def lattice_volume(lengths: Sequence[float], angles: Sequence[float]) -> float:
    a, b, c = [float(value) for value in lengths]
    alpha, beta, gamma = [math.radians(float(value)) for value in angles]
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_g = math.cos(gamma)
    radicand = 1.0 + 2.0 * cos_a * cos_b * cos_g - cos_a * cos_a - cos_b * cos_b - cos_g * cos_g
    return a * b * c * math.sqrt(max(radicand, 0.0))


def volume_per_atom_bin(lengths: Sequence[float], angles: Sequence[float], num_atoms: int) -> str:
    if int(num_atoms) <= 0:
        return "volpa_unknown"
    value = lattice_volume(lengths, angles) / float(num_atoms)
    if not math.isfinite(value) or value <= 0:
        return "volpa_unknown"
    low = int(math.floor(value / 5.0) * 5)
    high = low + 4
    return f"volpa_{low:03d}_{high:03d}"


def spacegroup_bucket(metadata: Mapping[str, Any]) -> str:
    raw = metadata.get("spacegroup.number.conv", metadata.get("spacegroup.number"))
    value = _safe_float(raw)
    if value is None:
        return "sg_unknown"
    number = int(value)
    if number <= 2:
        return "sg_001_002"
    if number <= 15:
        return "sg_003_015"
    if number <= 74:
        return "sg_016_074"
    if number <= 142:
        return "sg_075_142"
    if number <= 167:
        return "sg_143_167"
    if number <= 194:
        return "sg_168_194"
    if number <= 230:
        return "sg_195_230"
    return "sg_unknown"


def prototype_key(plan: Mapping[str, Any]) -> str:
    reduced_formula = str(plan.get("reduced_formula", "unknown"))
    return "|".join(
        [
            f"formula={reduced_formula}",
            f"anion={plan.get('anion_framework', 'unknown')}",
            f"charge={plan.get('charge_bucket', 'unknown')}",
            f"lat={plan.get('lattice_system', 'unknown')}",
            f"sg={plan.get('spacegroup_bucket', 'sg_unknown')}",
            f"vol={plan.get('volume_per_atom_bin', 'volpa_unknown')}",
        ]
    )


def plan_state_from_arrays(
    arrays: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None
) -> Dict[str, Any]:
    metadata = dict(metadata or {})
    species = [str(symbol) for symbol in arrays["species"]]
    symbols, full_counts = _full_composition_from_species(species)
    atom_types = [SYMBOL_TO_Z[symbol] for symbol in species if symbol in SYMBOL_TO_Z]
    reduced_elems, reduced_counts = reduced_composition(atom_types)
    reduced_symbols = list(element_symbols(reduced_elems))
    classification = _safe_smact_classification(symbols, full_counts)
    oxidation = classification.get("oxidation_states")
    plan: Dict[str, Any] = {
        "plan_state_version": PLAN_STATE_VERSION,
        "N": int(arrays["num_atoms"]),
        "elements": symbols,
        "counts": full_counts,
        "formula": _formula_from_symbols(symbols, full_counts),
        "reduced_formula": _formula_from_symbols(reduced_symbols, reduced_counts),
        "charge_bucket": charge_bucket_from_classification(classification),
        "oxidation_candidates": "unknown" if oxidation is None else list(oxidation),
        "anion_framework": anion_framework_from_symbols(symbols),
        "lattice_system": lattice_system_from_lattice(arrays["lengths"], arrays["angles"]),
        "spacegroup_bucket": spacegroup_bucket(metadata),
        "volume_per_atom_bin": volume_per_atom_bin(
            arrays["lengths"], arrays["angles"], int(arrays["num_atoms"])
        ),
        "validator": classification,
    }
    plan["prototype_key"] = prototype_key(plan)
    if metadata:
        plan["metadata"] = {
            key: metadata[key]
            for key in (
                "material_id",
                "pretty_formula",
                "e_above_hull",
                "spacegroup.number",
                "spacegroup.number.conv",
            )
            if key in metadata
        }
    return plan


def validate_plan_state(plan: Mapping[str, Any], *, max_atoms: int = 20) -> PlanValidation:
    try:
        num_atoms = int(plan.get("N"))
    except Exception:
        num_atoms = -1
    generated_raw = plan.get("generated_N", num_atoms)
    try:
        generated_num_atoms = int(generated_raw)
    except Exception:
        generated_num_atoms = -1
    elements = plan.get("elements")
    counts = plan.get("counts")
    valid_elements = (
        isinstance(elements, list)
        and len(elements) > 0
        and all((isinstance(symbol, str) and symbol in SYMBOL_TO_Z for symbol in elements))
    )
    valid_counts = (
        isinstance(counts, list)
        and isinstance(elements, list)
        and (len(counts) == len(elements))
        and all((isinstance(count, int) and count > 0 for count in counts))
        and (sum((int(count) for count in counts)) == num_atoms)
    )
    formula = plan.get("formula")
    expected_formula = _formula_from_symbols(elements, counts) if valid_elements and valid_counts else None
    valid_n_range = 1 <= num_atoms <= int(max_atoms)
    valid_generated_n = generated_num_atoms == num_atoms
    return PlanValidation(
        valid_N=valid_n_range and valid_generated_n,
        valid_generated_N=valid_generated_n,
        valid_formula=isinstance(formula, str) and formula == expected_formula,
        valid_counts=valid_counts,
        valid_elements=valid_elements,
    )
