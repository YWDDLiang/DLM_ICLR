"""Generation-position schedules for fixed-slot crystal DLM sampling."""

from __future__ import annotations


def _shift_groups(groups: list[list[int]], offset: int) -> list[list[int]]:
    if int(offset) == 0:
        return groups
    return [[int(position) + int(offset) for position in group] for group in groups]


def n_elements_coords_lattice_schedule(max_atoms: int = 20, offset: int = 0) -> list[list[int]]:
    """Return generation-position groups for N -> elements -> coords/lattice."""

    element_positions = [8 + 5 * slot_index for slot_index in range(max_atoms)]
    lattice_positions = list(range(1, 7))
    coord_positions: list[int] = []
    for slot_index in range(max_atoms):
        base = 7 + slot_index * 5
        coord_positions.extend([base + 2, base + 3, base + 4])
    return _shift_groups([[0], element_positions, lattice_positions + coord_positions], offset)


def n_elements_sequential_rest_schedule(max_atoms: int = 20, offset: int = 0) -> list[list[int]]:
    """Return N -> all elements -> remaining positions one-by-one.

    This keeps the useful composition bias from choosing active element tokens
    as a group, while preserving the sequential coordinate order needed by the
    PBC-aware duplicate-coordinate mask.
    """

    element_positions = [8 + 5 * slot_index for slot_index in range(max_atoms)]
    rest_positions = list(range(1, 7))
    for slot_index in range(max_atoms):
        base = 7 + slot_index * 5
        rest_positions.extend([base + 2, base + 3, base + 4])
    return _shift_groups([[0], element_positions] + [[position] for position in rest_positions], offset)
