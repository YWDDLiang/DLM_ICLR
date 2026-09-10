from __future__ import annotations

"Retained crystal DLM implementation; see docs/method.md for the public workflow."
from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Mapping, Tuple

MASK_TOKEN_ID = 126336
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

    max_atoms: int = 20
    length_step: float = 0.1
    length_min_bin: int = 0
    length_max_bin: int = 500
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
    "<(?:N_\\d{3}|L[ABC]_\\d{3}|A[ABG]_\\d{3}|S\\d{2}|E_[A-Z][a-z]?|[XYZ]_\\d{3}|EMPTY|[XYZ]_PAD)>"
)
COUNT_RE = re.compile("^<N_(\\d{3})>$")
LENGTH_RE = re.compile("^<L([ABC])_(\\d{3})>$")
ANGLE_RE = re.compile("^<A([ABG])_(\\d{3})>$")
ELEMENT_RE = re.compile("^<E_([A-Z][a-z]?)>$")
COORD_RE = re.compile("^<([XYZ])_(\\d{3})>$")


def answer_token_count(config: FixedSlotConfig = FixedSlotConfig()) -> int:
    """Return semantic answer length for a fixed-slot config."""
    return 1 + 6 + int(config.max_atoms) * 5


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _clip(value: int, low: int, high: int) -> Tuple[int, bool]:
    clipped = min(max(value, low), high)
    return (clipped, clipped != value)


def _format_bin(prefix: str, value: int) -> str:
    return f"<{prefix}_{value:03d}>"


def _parse_int_token(regex: re.Pattern[str], token: str, name: str) -> Tuple[str, int]:
    match = regex.match(token)
    if not match:
        raise FixedSlotError(f"Expected {name} token, got {token!r}")
    return (match.group(1), int(match.group(2)))


def tokenize_answer_text(text: str) -> List[str]:
    """Extract fixed-slot schema tokens from decoded model text."""
    return SCHEMA_TOKEN_RE.findall(text)


def normalize_fractional_coord(value: float) -> Tuple[float, bool]:
    """Map a fractional coordinate into [0, 1)."""
    wrapped = value % 1.0
    changed = not math.isclose(wrapped, value, abs_tol=1e-10)
    return (wrapped, changed)


def arrays_to_structure(arrays: Mapping[str, Any]) -> Any:
    """Build a pymatgen Structure from parsed arrays."""
    try:
        from pymatgen.core import Lattice, Structure
    except ImportError as exc:
        raise RuntimeError("pymatgen is required to build a Structure") from exc
    return Structure(
        lattice=Lattice.from_parameters(*list(arrays["lengths"]) + list(arrays["angles"])),
        species=list(arrays["species"]),
        coords=list(arrays["frac_coords"]),
        coords_are_cartesian=False,
    )
