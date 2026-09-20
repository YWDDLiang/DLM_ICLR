"""Fixed-slot crystal representation used by the MP-20 DLM experiments.

The schema follows rawplan.md:

    <N_012>
    <LA_080> <LB_080> <LC_056> <AA_090> <AB_090> <AG_120>
    <S00> <E_Na> <X_033> <Y_066> <Z_000>
    ...
    <S19> <EMPTY> <X_PAD> <Y_PAD> <Z_PAD>

The default MP-20 answer is 107 semantic tokens: 1 atom-count token,
6 lattice tokens, and 20 atom slots with 5 tokens each. Some downstream
feasibility checks reuse the same schema with a larger max_atoms value.
"""

from __future__ import annotations
from dlm_iclr.runtime.capacity import MAX_ATOMS, MIN_ATOMS, DATASET_LABEL, ATOM_RANGE_TEXT, LENGTH_MAX_BIN

from dataclasses import dataclass, field
import json
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ANSWER_TOKEN_COUNT = 1 + 6 + MAX_ATOMS * 5
MASK_TOKEN_ID = 126336

CANONICAL_PROMPT = (
    "Below is a description of a bulk material. Generate a description of "
    "the lengths and angles of the lattice vectors and then the element type "
    "and coordinates for each atom within the lattice in fixed-slot format:"
)

PROMPT_POOL = [
    CANONICAL_PROMPT,
    (
        "Generate a bulk material by first describing the lengths and angles "
        "of the lattice vectors, followed by the element type and fractional "
        "coordinates for each atom in fixed-slot format:"
    ),
    (
        "Provide a fixed-slot crystal structure for a bulk material, including "
        "lattice lengths, lattice angles, atom species, and fractional coordinates:"
    ),
    (
        "Generate a valid crystal description in fixed-slot format. Include "
        "lattice lengths, lattice angles, element types, and fractional coordinates:"
    ),
]

CHEMICAL_SYMBOLS = [
    "X",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
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
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
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
    "Sb",
    "Te",
    "I",
    "Xe",
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
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
]

SYMBOL_TO_Z = {symbol: z for z, symbol in enumerate(CHEMICAL_SYMBOLS) if z > 0}
Z_TO_SYMBOL = {z: symbol for symbol, z in SYMBOL_TO_Z.items()}


@dataclass(frozen=True)
class FixedSlotConfig:
    """Configuration for MP-20 fixed-slot tokenization."""

    max_atoms: int = MAX_ATOMS
    length_step: float = 0.1
    length_min_bin: int = 0
    length_max_bin: int = LENGTH_MAX_BIN
    angle_min_bin: int = 1
    angle_max_bin: int = 179
    coord_min_bin: int = 0
    coord_max_bin: int = 100
    max_atomic_number: int = SYMBOL_TO_Z["Pu"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_atoms": self.max_atoms,
            "length_step": self.length_step,
            "length_min_bin": self.length_min_bin,
            "length_max_bin": self.length_max_bin,
            "angle_min_bin": self.angle_min_bin,
            "angle_max_bin": self.angle_max_bin,
            "coord_min_bin": self.coord_min_bin,
            "coord_max_bin": self.coord_max_bin,
            "max_atomic_number": self.max_atomic_number,
            "answer_token_count": answer_token_count(self),
        }


@dataclass
class EncodeDiagnostics:
    """Small counters collected while discretizing a structure."""

    length_clips: int = 0
    angle_clips: int = 0
    coord_clips: int = 0
    coord_wraps: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "length_clips": self.length_clips,
            "angle_clips": self.angle_clips,
            "coord_clips": self.coord_clips,
            "coord_wraps": self.coord_wraps,
            "notes": list(self.notes),
        }


class FixedSlotError(ValueError):
    """Raised when a fixed-slot answer cannot be parsed."""


SCHEMA_TOKEN_RE = re.compile(
    r"<(?:N_\d{3}|L[ABC]_\d{3,}|A[ABG]_\d{3}|S\d{2}|"
    r"E_[A-Z][a-z]?|[XYZ]_\d{3}|EMPTY|[XYZ]_PAD)>"
)
COUNT_RE = re.compile(r"^<N_(\d{3})>$")
LENGTH_RE = re.compile(r"^<L([ABC])_(\d{3,})>$")
ANGLE_RE = re.compile(r"^<A([ABG])_(\d{3})>$")
SLOT_RE = re.compile(r"^<S(\d{2})>$")
ELEMENT_RE = re.compile(r"^<E_([A-Z][a-z]?)>$")
COORD_RE = re.compile(r"^<([XYZ])_(\d{3})>$")
PAD_COORD_RE = re.compile(r"^<([XYZ])_PAD>$")


def answer_token_count(config: FixedSlotConfig = FixedSlotConfig()) -> int:
    """Return semantic answer length for a fixed-slot config."""

    return 1 + 6 + int(config.max_atoms) * 5


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _clip(value: int, low: int, high: int) -> Tuple[int, bool]:
    clipped = min(max(value, low), high)
    return clipped, clipped != value


def _format_bin(prefix: str, value: int) -> str:
    return f"<{prefix}_{value:03d}>"


def _parse_int_token(regex: re.Pattern[str], token: str, name: str) -> Tuple[str, int]:
    match = regex.match(token)
    if not match:
        raise FixedSlotError(f"Expected {name} token, got {token!r}")
    return match.group(1), int(match.group(2))


def tokenize_answer_text(text: str) -> List[str]:
    """Extract fixed-slot schema tokens from decoded model text."""

    return SCHEMA_TOKEN_RE.findall(text)


def build_special_tokens(config: FixedSlotConfig = FixedSlotConfig()) -> List[str]:
    """Return the complete crystal special-token vocabulary for this schema."""

    tokens: List[str] = []
    tokens.extend(f"<N_{i:03d}>" for i in range(1, config.max_atoms + 1))
    for prefix in ("LA", "LB", "LC"):
        tokens.extend(f"<{prefix}_{i:03d}>" for i in range(config.length_min_bin, config.length_max_bin + 1))
    for prefix in ("AA", "AB", "AG"):
        tokens.extend(f"<{prefix}_{i:03d}>" for i in range(config.angle_min_bin, config.angle_max_bin + 1))
    tokens.extend(f"<S{i:02d}>" for i in range(config.max_atoms))
    tokens.extend(f"<E_{Z_TO_SYMBOL[z]}>" for z in range(1, config.max_atomic_number + 1))
    for prefix in ("X", "Y", "Z"):
        tokens.extend(f"<{prefix}_{i:03d}>" for i in range(config.coord_min_bin, config.coord_max_bin + 1))
    tokens.extend(["<EMPTY>", "<X_PAD>", "<Y_PAD>", "<Z_PAD>"])
    return tokens


def normalize_fractional_coord(value: float) -> Tuple[float, bool]:
    """Map a fractional coordinate into [0, 1)."""

    wrapped = value % 1.0
    changed = not math.isclose(wrapped, value, abs_tol=1e-10)
    return wrapped, changed


def arrays_to_tokens(
    lengths: Sequence[float],
    angles: Sequence[float],
    species: Sequence[str],
    frac_coords: Sequence[Sequence[float]],
    config: FixedSlotConfig = FixedSlotConfig(),
) -> Tuple[List[str], EncodeDiagnostics]:
    """Convert crystal arrays into fixed-slot tokens."""

    diagnostics = EncodeDiagnostics()

    if len(lengths) != 3:
        raise FixedSlotError(f"Expected 3 lattice lengths, got {len(lengths)}")
    if len(angles) != 3:
        raise FixedSlotError(f"Expected 3 lattice angles, got {len(angles)}")
    if len(species) != len(frac_coords):
        raise FixedSlotError("Species and coordinate counts do not match")
    if not 1 <= len(species) <= config.max_atoms:
        raise FixedSlotError(f"Atom count {len(species)} outside 1..{config.max_atoms}")

    tokens = [f"<N_{len(species):03d}>"]

    for prefix, length in zip(("LA", "LB", "LC"), lengths):
        raw_bin = _round_half_up(float(length) / config.length_step)
        bin_value, clipped = _clip(raw_bin, config.length_min_bin, config.length_max_bin)
        diagnostics.length_clips += int(clipped)
        tokens.append(_format_bin(prefix, bin_value))

    for prefix, angle in zip(("AA", "AB", "AG"), angles):
        raw_bin = _round_half_up(float(angle))
        bin_value, clipped = _clip(raw_bin, config.angle_min_bin, config.angle_max_bin)
        diagnostics.angle_clips += int(clipped)
        tokens.append(_format_bin(prefix, bin_value))

    for slot_index in range(config.max_atoms):
        tokens.append(f"<S{slot_index:02d}>")
        if slot_index >= len(species):
            tokens.extend(["<EMPTY>", "<X_PAD>", "<Y_PAD>", "<Z_PAD>"])
            continue

        symbol = str(species[slot_index])
        if symbol not in SYMBOL_TO_Z:
            raise FixedSlotError(f"Unsupported element symbol {symbol!r}")
        if SYMBOL_TO_Z[symbol] > config.max_atomic_number:
            raise FixedSlotError(f"Element {symbol} exceeds configured max Z")
        tokens.append(f"<E_{symbol}>")

        coord = frac_coords[slot_index]
        if len(coord) != 3:
            raise FixedSlotError(f"Atom slot {slot_index} does not have 3 coords")
        for prefix, value in zip(("X", "Y", "Z"), coord):
            wrapped, changed = normalize_fractional_coord(float(value))
            diagnostics.coord_wraps += int(changed)
            raw_bin = _round_half_up(wrapped * config.coord_max_bin)
            bin_value, clipped = _clip(raw_bin, config.coord_min_bin, config.coord_max_bin)
            diagnostics.coord_clips += int(clipped)
            tokens.append(_format_bin(prefix, bin_value))

    expected_tokens = answer_token_count(config)
    if len(tokens) != expected_tokens:
        raise FixedSlotError(
            f"Internal schema error: produced {len(tokens)} tokens, expected {expected_tokens}"
        )
    return tokens, diagnostics


def tokens_to_arrays(
    tokens: Sequence[str],
    config: FixedSlotConfig = FixedSlotConfig(),
    strict: bool = True,
) -> Dict[str, Any]:
    """Parse fixed-slot tokens into crystal arrays."""

    token_list = list(tokens)
    expected_tokens = answer_token_count(config)
    if strict and len(token_list) != expected_tokens:
        raise FixedSlotError(f"Expected {expected_tokens} tokens, got {len(token_list)}")
    if len(token_list) < expected_tokens:
        raise FixedSlotError(f"Expected at least {expected_tokens} tokens, got {len(token_list)}")
    token_list = token_list[:expected_tokens]

    count_match = COUNT_RE.match(token_list[0])
    if not count_match:
        raise FixedSlotError(f"Invalid atom-count token {token_list[0]!r}")
    num_atoms = int(count_match.group(1))
    if not 1 <= num_atoms <= config.max_atoms:
        raise FixedSlotError(f"Atom count {num_atoms} outside schema range")

    lengths: List[float] = []
    for expected_axis, token in zip(("A", "B", "C"), token_list[1:4]):
        axis, bin_value = _parse_int_token(LENGTH_RE, token, "length")
        if axis != expected_axis:
            raise FixedSlotError(f"Expected length axis {expected_axis}, got {axis}")
        if not config.length_min_bin <= bin_value <= config.length_max_bin:
            raise FixedSlotError(f"Length bin {bin_value} outside range")
        lengths.append(bin_value * config.length_step)

    angles: List[float] = []
    for expected_axis, token in zip(("A", "B", "G"), token_list[4:7]):
        axis, bin_value = _parse_int_token(ANGLE_RE, token, "angle")
        if axis != expected_axis:
            raise FixedSlotError(f"Expected angle axis {expected_axis}, got {axis}")
        if not config.angle_min_bin <= bin_value <= config.angle_max_bin:
            raise FixedSlotError(f"Angle bin {bin_value} outside range")
        angles.append(float(bin_value))

    species: List[str] = []
    atom_types: List[int] = []
    frac_coords: List[List[float]] = []
    position = 7
    for slot_index in range(config.max_atoms):
        slot_token = token_list[position]
        slot_match = SLOT_RE.match(slot_token)
        if not slot_match:
            raise FixedSlotError(f"Invalid slot token {slot_token!r}")
        actual_slot = int(slot_match.group(1))
        if actual_slot != slot_index:
            raise FixedSlotError(f"Expected slot {slot_index:02d}, got {actual_slot:02d}")
        position += 1

        field_tokens = token_list[position : position + 4]
        position += 4

        if slot_index >= num_atoms:
            if field_tokens != ["<EMPTY>", "<X_PAD>", "<Y_PAD>", "<Z_PAD>"]:
                raise FixedSlotError(f"Slot {slot_index:02d} should be empty, got {field_tokens}")
            continue

        element_match = ELEMENT_RE.match(field_tokens[0])
        if not element_match:
            raise FixedSlotError(f"Slot {slot_index:02d} expected element token, got {field_tokens[0]!r}")
        symbol = element_match.group(1)
        if symbol not in SYMBOL_TO_Z:
            raise FixedSlotError(f"Unsupported element token {field_tokens[0]!r}")
        atomic_number = SYMBOL_TO_Z[symbol]
        if atomic_number > config.max_atomic_number:
            raise FixedSlotError(f"Element {symbol} exceeds configured max Z")

        coord_values: List[float] = []
        for expected_axis, coord_token in zip(("X", "Y", "Z"), field_tokens[1:]):
            coord_match = COORD_RE.match(coord_token)
            if not coord_match:
                raise FixedSlotError(f"Slot {slot_index:02d} expected coordinate token, got {coord_token!r}")
            axis = coord_match.group(1)
            bin_value = int(coord_match.group(2))
            if axis != expected_axis:
                raise FixedSlotError(f"Expected coord axis {expected_axis}, got {axis}")
            if not config.coord_min_bin <= bin_value <= config.coord_max_bin:
                raise FixedSlotError(f"Coordinate bin {bin_value} outside range")
            coord_values.append(bin_value / config.coord_max_bin)

        species.append(symbol)
        atom_types.append(atomic_number)
        frac_coords.append(coord_values)

    return {
        "num_atoms": num_atoms,
        "lengths": lengths,
        "angles": angles,
        "species": species,
        "atom_types": atom_types,
        "frac_coords": frac_coords,
        "tokens": token_list,
        "answer": " ".join(token_list),
    }


def parse_fixed_slot_answer(
    text: str,
    config: FixedSlotConfig = FixedSlotConfig(),
    strict: bool = False,
) -> Dict[str, Any]:
    """Parse a decoded model answer.

    When strict is False, extra decoded text is ignored and the first 107
    fixed-slot-looking tokens are parsed. This is useful for model samples that
    may include prompt echoes or trailing text.
    """

    tokens = tokenize_answer_text(text)
    return tokens_to_arrays(tokens, config=config, strict=strict)


def arrays_to_answer(
    lengths: Sequence[float],
    angles: Sequence[float],
    species: Sequence[str],
    frac_coords: Sequence[Sequence[float]],
    config: FixedSlotConfig = FixedSlotConfig(),
    separator: str = " ",
) -> Tuple[str, EncodeDiagnostics]:
    tokens, diagnostics = arrays_to_tokens(
        lengths=lengths,
        angles=angles,
        species=species,
        frac_coords=frac_coords,
        config=config,
    )
    return separator.join(tokens), diagnostics


def structure_to_answer(
    structure: Any,
    config: FixedSlotConfig = FixedSlotConfig(),
    separator: str = " ",
) -> Tuple[str, EncodeDiagnostics]:
    """Convert a pymatgen Structure to a fixed-slot answer."""

    lengths = list(structure.lattice.abc)
    angles = list(structure.lattice.angles)
    species = [site.specie.symbol for site in structure.sites]
    frac_coords = structure.frac_coords.tolist()
    return arrays_to_answer(lengths, angles, species, frac_coords, config=config, separator=separator)


def arrays_to_structure(arrays: Mapping[str, Any]) -> Any:
    """Build a pymatgen Structure from parsed arrays."""

    try:
        from pymatgen.core import Lattice, Structure
    except ImportError as exc:
        raise RuntimeError("pymatgen is required to build a Structure") from exc

    return Structure(
        lattice=Lattice.from_parameters(*(list(arrays["lengths"]) + list(arrays["angles"]))),
        species=list(arrays["species"]),
        coords=list(arrays["frac_coords"]),
        coords_are_cartesian=False,
    )


def answer_to_structure(answer: str, config: FixedSlotConfig = FixedSlotConfig()) -> Any:
    return arrays_to_structure(parse_fixed_slot_answer(answer, config=config))


def arrays_to_torch_payload(arrays_list: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Convert parsed arrays into the .pt layout expected by compute_metrics.py."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to create a .pt payload") from exc

    frac_rows: List[List[float]] = []
    atom_rows: List[int] = []
    lengths_rows: List[Sequence[float]] = []
    angles_rows: List[Sequence[float]] = []
    num_atoms: List[int] = []
    for arrays in arrays_list:
        frac_rows.extend(arrays["frac_coords"])
        atom_rows.extend(arrays["atom_types"])
        lengths_rows.append(arrays["lengths"])
        angles_rows.append(arrays["angles"])
        num_atoms.append(int(arrays["num_atoms"]))

    return {
        "frac_coords": torch.tensor(frac_rows, dtype=torch.float32),
        "atom_types": torch.tensor(atom_rows, dtype=torch.long),
        "lengths": torch.tensor(lengths_rows, dtype=torch.float32),
        "angles": torch.tensor(angles_rows, dtype=torch.float32),
        "num_atoms": torch.tensor(num_atoms, dtype=torch.long),
    }


def metadata_from_csv_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Preserve MP-20 metadata for future conditional work."""

    keys = [
        "material_id",
        "formation_energy_per_atom",
        "band_gap",
        "pretty_formula",
        "e_above_hull",
        "elements",
        "spacegroup.number",
        "spacegroup.number.conv",
    ]
    metadata: Dict[str, Any] = {}
    for key in keys:
        if key in row and row[key] not in ("", None):
            metadata[key] = _coerce_json_value(row[key])
    return metadata


def _coerce_json_value(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    text = str(value)
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        return text


def write_json(path: str, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
