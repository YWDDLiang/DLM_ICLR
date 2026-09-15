"""CIF-lite modular text representation for CrysLLMGen-style DLM trials.

The representation deliberately avoids the crystal special-token vocabulary.
One adapter is trained on three module tasks:

    composition -> lattice -> sites

Later module prompts include the accepted text from earlier modules.  The
parser enforces that the site species multiset exactly matches the composition
header and treats fractional coordinates modulo the periodic cell.
"""

from __future__ import annotations

from collections import Counter
import random
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from dlm_iclr._core.fixed_slot import (
    SYMBOL_TO_Z,
    Z_TO_SYMBOL,
    arrays_to_structure,
    arrays_to_torch_payload,
    metadata_from_csv_row,
    write_json,
)
from dlm_iclr._core.lattice_geometry import lattice_angle_rad


class CifLiteError(ValueError):
    """Raised when CIF-lite text cannot be parsed into a crystal."""


CIF_LITE_PROMPT_VERSION = "cif_lite_modular_v2_strict_schema_prompt"

CANONICAL_CIF_LITE_COMPOSITION_PROMPT = (
    "Generate an MP-20 crystal composition block in exactly this schema.\n"
    "Use chemical symbols only, never element names, words, bullets, colons, "
    "percentages, decimals, markdown, or explanation.\n"
    "Each count is a positive integer and the total atom count is at most 20.\n"
    "Example:\n"
    "composition:\n"
    "Li 2\n"
    "V 1\n"
    "O 4\n"
    "end\n"
    "Now output one new composition block:"
)

CANONICAL_CIF_LITE_LATTICE_PROMPT = (
    "Given the composition block below, generate an MP-20 crystal lattice block "
    "in exactly this schema.\n"
    "Use only two numeric lines: three positive lengths with one decimal, then "
    "three angles as integers. Do not write explanation.\n"
    "Example:\n"
    "lattice:\n"
    "5.4 5.4 7.6\n"
    "90 90 120\n"
    "end\n"
    "Composition:"
)

CANONICAL_CIF_LITE_SITES_PROMPT = (
    "Given the composition and lattice blocks below, generate an MP-20 crystal "
    "sites block in exactly this schema.\n"
    "For each atom write one chemical symbol line and one fractional coordinate "
    "line with three decimals rounded to two places. The site symbols must match "
    "the composition multiset exactly. Do not write explanation.\n"
    "Example:\n"
    "sites:\n"
    "Li\n"
    "0.00 0.00 0.00\n"
    "O\n"
    "0.50 0.50 0.50\n"
    "end\n"
    "Conditioning blocks:"
)

CIF_LITE_COMPOSITION_PROMPT_POOL = [
    CANONICAL_CIF_LITE_COMPOSITION_PROMPT,
    (
        "Output exactly one MP-20 composition block. Only lines of the form "
        "`ElementSymbol integer_count` are allowed between composition: and end. "
        "No percentages, no full element names, no prose:"
    ),
    (
        "Write a crystal atom-count composition block using only chemical "
        "symbols and positive integer counts. End with the line end:"
    ),
]

CIF_LITE_LATTICE_PROMPT_POOL = [
    CANONICAL_CIF_LITE_LATTICE_PROMPT,
    (
        "Use this composition to output exactly one lattice block: three decimal "
        "lengths, three integer angles, then end. No prose:"
    ),
]

CIF_LITE_SITES_PROMPT_POOL = [
    CANONICAL_CIF_LITE_SITES_PROMPT,
    (
        "Use this composition and lattice to output exactly one sites block. "
        "Each site is a chemical symbol line followed by x y z fractional "
        "coordinates. No prose:"
    ),
]

CIF_LITE_PROMPT_POOL = {
    "composition": CIF_LITE_COMPOSITION_PROMPT_POOL,
    "lattice": CIF_LITE_LATTICE_PROMPT_POOL,
    "sites": CIF_LITE_SITES_PROMPT_POOL,
}

MODULE_TO_ID = {"composition": 1, "lattice": 2, "sites": 3}
ID_TO_MODULE = {value: key for key, value in MODULE_TO_ID.items()}
MAX_MP20_ATOMS = 20


def _clean_lines(text: str) -> List[str]:
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return [line.strip() for line in text.split("\n") if line.strip()]


def _is_end_line(line: str) -> bool:
    return line.strip().lower() == "end"


def _find_header(lines: Sequence[str], header: str) -> int | None:
    wanted = header.lower().rstrip(":")
    for idx, line in enumerate(lines):
        normalized = line.lower().rstrip(":")
        if normalized == wanted:
            return idx
    return None


def extract_module_lines(text: str, module: str, *, allow_missing_header: bool = True) -> List[str]:
    """Return content lines between ``module:`` and ``end``.

    Generation sometimes repeats prompt fragments before the module header.  We
    accept that by scanning for the header first; if it is missing, the parser
    can optionally treat the beginning of the decoded suffix as the module.
    """

    lines = _clean_lines(text)
    if not lines:
        raise CifLiteError(f"Empty {module} block")
    start_idx = _find_header(lines, f"{module}:")
    if start_idx is None:
        if not allow_missing_header:
            raise CifLiteError(f"Missing {module}: header")
        start_idx = -1
    content: List[str] = []
    for line in lines[start_idx + 1 :]:
        if _is_end_line(line):
            break
        lower = line.lower().rstrip(":")
        if lower in {"composition", "lattice", "sites"} and lower != module:
            break
        content.append(line)
    if not content:
        raise CifLiteError(f"No content in {module} block")
    return content


def truncate_module_text(text: str, module: str) -> str:
    """Keep the first complete module block from generated text."""

    lines = _clean_lines(text)
    start_idx = _find_header(lines, f"{module}:")
    if start_idx is None:
        start_idx = 0
        out = [f"{module}:"]
    else:
        out = []
    for line in lines[start_idx:]:
        out.append(line)
        if _is_end_line(line):
            break
    if not out or not _is_end_line(out[-1]):
        out.append("end")
    return "\n".join(out) + "\n"


def _symbol_key(symbol: str) -> int:
    if symbol not in SYMBOL_TO_Z:
        raise CifLiteError(f"Unsupported element symbol {symbol!r}")
    return int(SYMBOL_TO_Z[symbol])


def _species_symbol(value: Any) -> str:
    if hasattr(value, "symbol"):
        return str(value.symbol)
    if hasattr(value, "specie") and hasattr(value.specie, "symbol"):
        return str(value.specie.symbol)
    return str(value)


def format_composition_block(species: Sequence[str]) -> str:
    counts = Counter(str(symbol) for symbol in species)
    for symbol, count in counts.items():
        _symbol_key(symbol)
        if int(count) <= 0:
            raise CifLiteError(f"Invalid nonpositive count for {symbol}: {count}")
    lines = ["composition:"]
    for symbol in sorted(counts, key=_symbol_key):
        lines.append(f"{symbol} {int(counts[symbol])}")
    lines.append("end")
    return "\n".join(lines) + "\n"


def parse_composition_block(text: str) -> Dict[str, Any]:
    lines = extract_module_lines(text, "composition")
    counts: Counter[str] = Counter()
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            raise CifLiteError(f"Invalid composition line {line!r}")
        symbol = parts[0]
        _symbol_key(symbol)
        try:
            count = int(parts[1])
        except ValueError as exc:
            raise CifLiteError(f"Invalid composition count in {line!r}") from exc
        if count <= 0:
            raise CifLiteError(f"Composition count must be positive in {line!r}")
        counts[symbol] += count
    num_atoms = sum(counts.values())
    if not 1 <= num_atoms <= MAX_MP20_ATOMS:
        raise CifLiteError(f"Composition atom count {num_atoms} outside 1..{MAX_MP20_ATOMS}")
    species_multiset: List[str] = []
    atom_types: List[int] = []
    for symbol in sorted(counts, key=_symbol_key):
        species_multiset.extend([symbol] * int(counts[symbol]))
        atom_types.extend([SYMBOL_TO_Z[symbol]] * int(counts[symbol]))
    return {
        "counts": dict(sorted(counts.items(), key=lambda item: _symbol_key(item[0]))),
        "species_multiset": species_multiset,
        "atom_types_multiset": atom_types,
        "num_atoms": int(num_atoms),
        "block": format_composition_block(species_multiset),
    }


def format_lattice_block(lengths: Sequence[float], angles: Sequence[float]) -> str:
    if len(lengths) != 3 or len(angles) != 3:
        raise CifLiteError("Lattice requires 3 lengths and 3 angles")
    lines = [
        "lattice:",
        " ".join(f"{float(value):.1f}" for value in lengths),
        " ".join(str(int(round(float(value)))) for value in angles),
        "end",
    ]
    return "\n".join(lines) + "\n"


def parse_lattice_block(text: str, *, min_lattice_rad: float = 1e-4) -> Dict[str, Any]:
    lines = extract_module_lines(text, "lattice")
    if len(lines) < 2:
        raise CifLiteError("Lattice block requires length and angle lines")
    try:
        lengths = [float(value) for value in lines[0].split()]
        angles = [float(value) for value in lines[1].split()]
    except ValueError as exc:
        raise CifLiteError("Lattice lines must contain numeric values") from exc
    if len(lengths) != 3 or len(angles) != 3:
        raise CifLiteError("Lattice length and angle lines must each contain 3 numbers")
    if any(value <= 0.0 for value in lengths):
        raise CifLiteError(f"Lattice lengths must be positive: {lengths}")
    if any(value <= 0.0 or value >= 180.0 for value in angles):
        raise CifLiteError(f"Lattice angles must be in (0, 180): {angles}")
    if (
        lattice_angle_rad(int(round(angles[0])), int(round(angles[1])), int(round(angles[2])))
        <= min_lattice_rad
    ):
        raise CifLiteError(f"Illegal lattice angle triple: {angles}")
    return {
        "lengths": lengths,
        "angles": angles,
        "block": format_lattice_block(lengths, angles),
    }


def _wrap_frac(value: float) -> float:
    wrapped = float(value) % 1.0
    if abs(wrapped - 1.0) < 1e-8:
        return 0.0
    return wrapped


def format_sites_block(species: Sequence[str], frac_coords: Sequence[Sequence[float]]) -> str:
    if len(species) != len(frac_coords):
        raise CifLiteError("Species and coordinate counts do not match")
    lines = ["sites:"]
    for symbol, coord in zip(species, frac_coords):
        _symbol_key(str(symbol))
        if len(coord) != 3:
            raise CifLiteError(f"Site for {symbol} does not have 3 coordinates")
        lines.append(str(symbol))
        lines.append(" ".join(f"{_wrap_frac(float(value)):.2f}" for value in coord))
    lines.append("end")
    return "\n".join(lines) + "\n"


def parse_sites_block(text: str) -> Dict[str, Any]:
    lines = extract_module_lines(text, "sites")
    if len(lines) % 2 != 0:
        raise CifLiteError(f"Sites block must contain element/coord pairs, got {len(lines)} content lines")
    species: List[str] = []
    atom_types: List[int] = []
    frac_coords: List[List[float]] = []
    for idx in range(0, len(lines), 2):
        symbol = lines[idx].split()[0]
        _symbol_key(symbol)
        try:
            coord = [_wrap_frac(float(value)) for value in lines[idx + 1].split()]
        except ValueError as exc:
            raise CifLiteError(f"Invalid coordinate line {lines[idx + 1]!r}") from exc
        if len(coord) != 3:
            raise CifLiteError(f"Coordinate line must contain 3 values: {lines[idx + 1]!r}")
        species.append(symbol)
        atom_types.append(int(SYMBOL_TO_Z[symbol]))
        frac_coords.append(coord)
    if not 1 <= len(species) <= MAX_MP20_ATOMS:
        raise CifLiteError(f"Sites atom count {len(species)} outside 1..{MAX_MP20_ATOMS}")
    return {
        "species": species,
        "atom_types": atom_types,
        "frac_coords": frac_coords,
        "num_atoms": len(species),
        "block": format_sites_block(species, frac_coords),
    }


def pbc_coordinate_key(coord: Sequence[float], bins: int = 100) -> Tuple[int, int, int]:
    if len(coord) != 3:
        raise CifLiteError(f"Expected 3 fractional coordinates, got {len(coord)}")
    return tuple(int(round(_wrap_frac(float(value)) * bins)) % bins for value in coord)


def assert_no_pbc_duplicate(frac_coords: Sequence[Sequence[float]]) -> None:
    seen: Dict[Tuple[int, int, int], int] = {}
    for idx, coord in enumerate(frac_coords):
        key = pbc_coordinate_key(coord)
        if key in seen:
            raise CifLiteError(
                f"duplicate/PBC-equivalent fractional coordinate {key} for sites {seen[key]} and {idx}"
            )
        seen[key] = idx


def parse_cif_lite_modules(
    composition_text: str,
    lattice_text: str,
    sites_text: str,
    *,
    require_no_pbc_duplicate: bool = True,
) -> Dict[str, Any]:
    composition = parse_composition_block(composition_text)
    lattice = parse_lattice_block(lattice_text)
    sites = parse_sites_block(sites_text)
    if Counter(sites["species"]) != Counter(composition["species_multiset"]):
        raise CifLiteError(
            "Sites species multiset does not match composition: "
            f"composition={dict(Counter(composition['species_multiset']))}, "
            f"sites={dict(Counter(sites['species']))}"
        )
    if require_no_pbc_duplicate:
        assert_no_pbc_duplicate(sites["frac_coords"])
    return {
        "num_atoms": int(composition["num_atoms"]),
        "lengths": lattice["lengths"],
        "angles": lattice["angles"],
        "species": sites["species"],
        "atom_types": sites["atom_types"],
        "frac_coords": sites["frac_coords"],
        "composition": composition["counts"],
        "composition_block": composition["block"],
        "lattice_block": lattice["block"],
        "sites_block": sites["block"],
        "answer": composition["block"] + "\n" + lattice["block"] + "\n" + sites["block"],
    }


def parse_cif_lite_answer(text: str, *, require_no_pbc_duplicate: bool = True) -> Dict[str, Any]:
    return parse_cif_lite_modules(
        truncate_module_text(text, "composition"),
        truncate_module_text(text, "lattice"),
        truncate_module_text(text, "sites"),
        require_no_pbc_duplicate=require_no_pbc_duplicate,
    )


def composition_prompt() -> str:
    return CANONICAL_CIF_LITE_COMPOSITION_PROMPT


def lattice_prompt(composition_block: str) -> str:
    return CANONICAL_CIF_LITE_LATTICE_PROMPT + "\n\n" + composition_block.rstrip() + "\n"


def sites_prompt(composition_block: str, lattice_block: str) -> str:
    return (
        CANONICAL_CIF_LITE_SITES_PROMPT
        + "\n\n"
        + composition_block.rstrip()
        + "\n\n"
        + lattice_block.rstrip()
        + "\n"
    )


def build_module_prompt(
    module: str, *, composition_block: str | None = None, lattice_block: str | None = None
) -> str:
    if module == "composition":
        return composition_prompt()
    if module == "lattice":
        if composition_block is None:
            raise CifLiteError("lattice prompt requires composition_block")
        return lattice_prompt(composition_block)
    if module == "sites":
        if composition_block is None or lattice_block is None:
            raise CifLiteError("sites prompt requires composition_block and lattice_block")
        return sites_prompt(composition_block, lattice_block)
    raise CifLiteError(f"Unknown CIF-lite module {module!r}")


def _copy_structure_with_optional_shift(structure: Any, rng: random.Random | None, origin_shift: bool) -> Any:
    copied = structure.copy()
    if origin_shift:
        if rng is None:
            rng = random.Random()
        shift = [rng.random(), rng.random(), rng.random()]
        copied.translate_sites(
            indices=range(len(copied.sites)),
            vector=shift,
            frac_coords=True,
            to_unit_cell=True,
        )
    return copied


def structure_to_cif_lite_modules(
    structure: Any,
    *,
    rng: random.Random | None = None,
    origin_shift: bool = False,
    permute_sites: bool = False,
) -> Dict[str, Any]:
    structure = _copy_structure_with_optional_shift(structure, rng, origin_shift)
    species = [_species_symbol(site.specie) for site in structure.sites]
    frac_coords = structure.frac_coords.tolist()
    order = list(range(len(species)))
    if permute_sites:
        if rng is None:
            rng = random.Random()
        rng.shuffle(order)
    species_ordered = [species[idx] for idx in order]
    coords_ordered = [frac_coords[idx] for idx in order]
    composition = format_composition_block(species_ordered)
    lattice = format_lattice_block(structure.lattice.abc, structure.lattice.angles)
    sites = format_sites_block(species_ordered, coords_ordered)
    arrays = parse_cif_lite_modules(composition, lattice, sites)
    return {
        "composition": composition,
        "lattice": lattice,
        "sites": sites,
        "answer": arrays["answer"],
        "arrays": arrays,
    }


def module_record(
    *,
    module: str,
    modules: Mapping[str, Any],
    metadata: Mapping[str, Any],
    prompt: str,
    tokenizer=None,
) -> Dict[str, Any]:
    if module == "composition":
        answer = str(modules["composition"])
    elif module == "lattice":
        answer = str(modules["lattice"])
    elif module == "sites":
        answer = str(modules["sites"])
    else:
        raise CifLiteError(f"Unknown module {module!r}")
    prompt_text = prompt.rstrip() + "\n"
    text = prompt_text + answer
    prompt_length = None
    answer_model_length = None
    if tokenizer is not None:
        prompt_length = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        answer_model_length = len(tokenizer(answer, add_special_tokens=False)["input_ids"])
    return {
        "task": module,
        "module": module,
        "module_id": MODULE_TO_ID[module],
        "representation": "cif_lite_modular",
        "prompt": prompt,
        "answer": answer,
        "text": text,
        "prompt_length": prompt_length,
        "answer_model_length": answer_model_length,
        "num_atoms": int(modules["arrays"]["num_atoms"]),
        "metadata": dict(metadata),
    }


def atom_types_to_species(atom_types: Iterable[int]) -> List[str]:
    return [Z_TO_SYMBOL[int(value)] for value in atom_types]


__all__ = [
    "CANONICAL_CIF_LITE_COMPOSITION_PROMPT",
    "CANONICAL_CIF_LITE_LATTICE_PROMPT",
    "CANONICAL_CIF_LITE_SITES_PROMPT",
    "CIF_LITE_PROMPT_VERSION",
    "CIF_LITE_PROMPT_POOL",
    "CifLiteError",
    "ID_TO_MODULE",
    "MODULE_TO_ID",
    "arrays_to_structure",
    "arrays_to_torch_payload",
    "assert_no_pbc_duplicate",
    "atom_types_to_species",
    "build_module_prompt",
    "extract_module_lines",
    "format_composition_block",
    "format_lattice_block",
    "format_sites_block",
    "lattice_prompt",
    "metadata_from_csv_row",
    "module_record",
    "parse_cif_lite_answer",
    "parse_cif_lite_modules",
    "parse_composition_block",
    "parse_lattice_block",
    "parse_sites_block",
    "pbc_coordinate_key",
    "sites_prompt",
    "structure_to_cif_lite_modules",
    "truncate_module_text",
    "write_json",
]
