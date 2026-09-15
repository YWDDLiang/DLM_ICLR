"""Dynamic-length crystal representation for MP-20 conservative trials.

The dynamic-v1 answer removes fixed empty slots while keeping the existing
crystal token vocabulary:

    <N_004><LA_041><LB_041><LC_042><AA_090><AB_090><AG_120>
    <E_Li><X_000><Y_000><Z_000>...

The semantic length is ``7 + 4 * N`` and never contains ``<Sxx>``, ``<EMPTY>``,
or pad-coordinate tokens.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from dlm_iclr._core.fixed_slot import (
    ANGLE_RE,
    COORD_RE,
    COUNT_RE,
    ELEMENT_RE,
    LENGTH_RE,
    SYMBOL_TO_Z,
    FixedSlotConfig,
    FixedSlotError,
    _clip,
    _format_bin,
    _parse_int_token,
    _round_half_up,
    arrays_to_structure,
    arrays_to_torch_payload,
    build_special_tokens,
    metadata_from_csv_row,
    normalize_fractional_coord,
    tokenize_answer_text,
    write_json,
    Z_TO_SYMBOL,
    EncodeDiagnostics,
)


DYNAMIC_MAX_ANSWER_TOKEN_COUNT = 7 + 4 * FixedSlotConfig().max_atoms

CANONICAL_DYNAMIC_PROMPT = (
    "Below is a description of a bulk material. Generate a compact crystal "
    "description with atom count, lattice lengths, lattice angles, then exactly "
    "that many element and fractional coordinate entries:"
)

DYNAMIC_PROMPT_POOL = [
    CANONICAL_DYNAMIC_PROMPT,
    (
        "Generate a bulk material as compact crystal tokens: atom count, "
        "lattice parameters, and one element plus fractional coordinate triplet "
        "for each atom:"
    ),
    (
        "Provide a dynamic-length crystal structure for a bulk material, "
        "including lattice lengths, lattice angles, atom species, and "
        "fractional coordinates:"
    ),
]


def dynamic_answer_token_count(num_atoms: int) -> int:
    return 7 + 4 * int(num_atoms)


def dynamic_max_answer_token_count(config: FixedSlotConfig = FixedSlotConfig()) -> int:
    return 7 + 4 * int(config.max_atoms)


def arrays_to_dynamic_tokens(
    lengths: Sequence[float],
    angles: Sequence[float],
    species: Sequence[str],
    frac_coords: Sequence[Sequence[float]],
    config: FixedSlotConfig = FixedSlotConfig(),
) -> Tuple[List[str], EncodeDiagnostics]:
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

    for slot_index, symbol in enumerate(species):
        symbol = str(symbol)
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
    return tokens, diagnostics


def dynamic_tokens_to_arrays(
    tokens: Sequence[str],
    config: FixedSlotConfig = FixedSlotConfig(),
    strict: bool = True,
) -> Dict[str, Any]:
    token_list = list(tokens)
    if not token_list:
        raise FixedSlotError("Expected dynamic-v1 tokens, got none")
    count_match = COUNT_RE.match(token_list[0])
    if not count_match:
        raise FixedSlotError(f"Invalid atom-count token {token_list[0]!r}")
    num_atoms = int(count_match.group(1))
    if not 1 <= num_atoms <= config.max_atoms:
        raise FixedSlotError(f"Atom count {num_atoms} outside schema range")
    expected_tokens = dynamic_answer_token_count(num_atoms)
    if strict and len(token_list) != expected_tokens:
        raise FixedSlotError(f"Expected {expected_tokens} dynamic tokens, got {len(token_list)}")
    if len(token_list) < expected_tokens:
        raise FixedSlotError(f"Expected at least {expected_tokens} dynamic tokens, got {len(token_list)}")
    token_list = token_list[:expected_tokens]
    forbidden = [
        token
        for token in token_list
        if token.startswith("<S") or token in {"<EMPTY>", "<X_PAD>", "<Y_PAD>", "<Z_PAD>"}
    ]
    if forbidden:
        raise FixedSlotError(f"Dynamic-v1 answer contains fixed-slot-only tokens: {forbidden[:5]}")

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
    for atom_index in range(num_atoms):
        field_tokens = token_list[position : position + 4]
        position += 4
        element_match = ELEMENT_RE.match(field_tokens[0])
        if not element_match:
            raise FixedSlotError(f"Atom {atom_index:02d} expected element token, got {field_tokens[0]!r}")
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
                raise FixedSlotError(f"Atom {atom_index:02d} expected coordinate token, got {coord_token!r}")
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
        "answer": "".join(token_list),
    }


def parse_dynamic_answer(
    text: str,
    config: FixedSlotConfig = FixedSlotConfig(),
    strict: bool = False,
) -> Dict[str, Any]:
    return dynamic_tokens_to_arrays(tokenize_answer_text(text), config=config, strict=strict)


def arrays_to_dynamic_answer(
    lengths: Sequence[float],
    angles: Sequence[float],
    species: Sequence[str],
    frac_coords: Sequence[Sequence[float]],
    config: FixedSlotConfig = FixedSlotConfig(),
    separator: str = "",
) -> Tuple[str, EncodeDiagnostics]:
    tokens, diagnostics = arrays_to_dynamic_tokens(lengths, angles, species, frac_coords, config=config)
    return separator.join(tokens), diagnostics


def structure_to_dynamic_answer(
    structure: Any,
    config: FixedSlotConfig = FixedSlotConfig(),
    separator: str = "",
) -> Tuple[str, EncodeDiagnostics]:
    lengths = list(structure.lattice.abc)
    angles = list(structure.lattice.angles)
    species = [site.specie.symbol for site in structure.sites]
    frac_coords = structure.frac_coords.tolist()
    return arrays_to_dynamic_answer(lengths, angles, species, frac_coords, config=config, separator=separator)


__all__ = [
    "CANONICAL_DYNAMIC_PROMPT",
    "DYNAMIC_MAX_ANSWER_TOKEN_COUNT",
    "DYNAMIC_PROMPT_POOL",
    "Z_TO_SYMBOL",
    "arrays_to_dynamic_answer",
    "arrays_to_dynamic_tokens",
    "arrays_to_structure",
    "arrays_to_torch_payload",
    "build_special_tokens",
    "dynamic_answer_token_count",
    "dynamic_max_answer_token_count",
    "dynamic_tokens_to_arrays",
    "metadata_from_csv_row",
    "parse_dynamic_answer",
    "structure_to_dynamic_answer",
    "write_json",
]
