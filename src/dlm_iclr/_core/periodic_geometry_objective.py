"""Retained crystal DLM implementation; see docs/method.md for the public workflow."""

from __future__ import annotations
import re
from typing import Any


_FAMILY_PATTERNS = {
    "length": re.compile(r"^<L([ABC])_(\d{3,})>$"),
    "angle": re.compile(r"^<A([ABG])_(\d{3})>$"),
    "coord": re.compile(r"^<([XYZ])_(\d{3})>$"),
}


def build_geometry_token_support(tokenizer: Any) -> dict[str, dict[str, dict[str, list[float] | list[int]]]]:
    """Build a JSON-serializable legal-token/value table from a tokenizer."""

    support: dict[str, dict[str, dict[str, list[Any]]]] = {
        "length": {axis: {"ids": [], "values": []} for axis in "ABC"},
        "angle": {axis: {"ids": [], "values": []} for axis in "ABG"},
        "coord": {axis: {"ids": [], "values": []} for axis in "XYZ"},
    }
    for token, token_id in tokenizer.get_vocab().items():
        for family, pattern in _FAMILY_PATTERNS.items():
            match = pattern.fullmatch(str(token))
            if match is None:
                continue
            axis, raw_value = match.groups()
            value = int(raw_value)
            if family == "length":
                value = value * 0.1
            elif family == "coord":
                value = value / 100.0
            support[family][axis]["ids"].append(int(token_id))
            support[family][axis]["values"].append(float(value))
            break
    for family in support.values():
        for axis, table in family.items():
            pairs = sorted(zip(table["ids"], table["values"]))
            if not pairs:
                raise ValueError(f"tokenizer has no geometry support for axis {axis}")
            table["ids"] = [pair[0] for pair in pairs]
            table["values"] = [pair[1] for pair in pairs]
    return support
