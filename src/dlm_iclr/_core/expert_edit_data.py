"""Retained crystal DLM implementation; see docs/method.md for the public workflow."""

from __future__ import annotations
from dlm_iclr.runtime.capacity import MAX_ATOMS
from dataclasses import asdict
import itertools
import math
import numpy as np
from dlm_iclr._core.dynamic_crystal import arrays_to_dynamic_tokens, parse_dynamic_answer
from dlm_iclr._core.fixed_slot import FixedSlotConfig


GEOMETRY_PROTOCOL = {
    "minimum_distance_A": 0.5,
    "minimum_volume_A3": 0.1,
    "image_bound": "reciprocal_column_norm_complete_for_contact_cutoff",
    "max_pair_images": 4_000_000,
    "tolerance_A": 1e-8,
}


def lattice_from_parameters(lengths, angles):
    lengths, angles = np.asarray(lengths, dtype=float), np.asarray(angles, dtype=float)
    if lengths.shape != (3,) or angles.shape != (3,) or not np.isfinite(np.r_[lengths, angles]).all():
        raise ValueError("invalid lattice values")
    if (lengths <= 0).any() or (angles <= 0).any() or (angles >= 180).any():
        raise ValueError("lattice parameters outside physical domain")
    alpha, beta, gamma = np.deg2rad(angles)
    a, b, c = lengths
    sg = math.sin(gamma)
    cx, cy = c * math.cos(beta), c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sg
    square = c * c - cx * cx - cy * cy
    if sg <= 1e-10 or square <= 1e-10:
        raise ValueError("nonpositive lattice metric")
    return np.asarray([[a, 0.0, 0.0], [b * math.cos(gamma), b * sg, 0.0], [cx, cy, math.sqrt(square)]])


def certify_geometry(arrays, *, cutoff=0.5, max_pair_images=4_000_000):
    """Certify all periodic contacts below cutoff with an adaptive exact box.

    For a wrapped fractional difference d, any contact r=(d+n)L shorter
    than cutoff has |n_i| <= .5 + cutoff*||L^-1[:,i]||. Bounded chunks avoid
    a large all-image tensor; exceeding the explicit cap means unverified.
    """
    try:
        lattice = lattice_from_parameters(arrays["lengths"], arrays["angles"])
        coords = np.asarray(arrays["frac_coords"], dtype=float)
        n = len(arrays["species"])
        if not 1 <= n <= MAX_ATOMS or coords.shape != (n, 3) or not np.isfinite(coords).all():
            raise ValueError("invalid coordinate dimensions")
        volume = float(abs(np.linalg.det(lattice)))
        if volume < GEOMETRY_PROTOCOL["minimum_volume_A3"]:
            raise ValueError("volume below target admission")
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as error:
        return {"valid": False, "certified": True, "reason": str(error)}
    radii = np.ceil(0.5 + cutoff * np.linalg.norm(np.linalg.inv(lattice), axis=0) + 1e-12).astype(int)
    count = math.prod(int(2 * radius + 1) for radius in radii)
    if n * n * count > max_pair_images:
        return {
            "valid": None,
            "certified": False,
            "reason": "periodic_certificate_budget",
            "images": count,
            "volume_A3": volume,
        }
    delta = coords[:, None] - coords[None, :]
    delta -= np.round(delta)
    shifts = itertools.product(*(range(-int(r), int(r) + 1) for r in radii))
    minimum = float("inf")
    while True:
        block = list(itertools.islice(shifts, 256))
        if not block:
            break
        offsets = np.asarray(block, dtype=float)
        values = np.linalg.norm((delta[:, :, None, :] + offsets) @ lattice, axis=-1)
        zero = np.flatnonzero((offsets == 0).all(-1))
        if len(zero):
            values[np.arange(n), np.arange(n), int(zero[0])] = np.inf
        minimum = min(minimum, float(values.min()))
        if minimum < cutoff - GEOMETRY_PROTOCOL["tolerance_A"]:
            return {
                "valid": False,
                "certified": True,
                "reason": "periodic_short_contact",
                "witness_distance_A": minimum,
                "images": count,
                "volume_A3": volume,
            }
    return {
        "valid": True,
        "certified": True,
        "reason": None,
        "images": count,
        "searched_minimum_A": minimum,
        "volume_A3": volume,
    }


def quantize_arrays(arrays, vocabulary):
    tokens, diagnostics = arrays_to_dynamic_tokens(
        arrays["lengths"],
        arrays["angles"],
        arrays["species"],
        arrays["frac_coords"],
        config=FixedSlotConfig(),
    )
    if diagnostics.length_clips or diagnostics.angle_clips or diagnostics.coord_clips:
        raise ValueError("expert target clipped by B0 quantization")
    tokens = [
        token.replace("_100>", "_000>") if token.startswith(("<X_", "<Y_", "<Z_")) else token
        for token in tokens
    ]
    try:
        ids = [int(vocabulary[token]) for token in tokens]
    except KeyError as error:
        raise ValueError("target token is outside the preserved B0 vocabulary") from error
    decoded = parse_dynamic_answer("".join(tokens), strict=True)
    return ids, decoded, asdict(diagnostics)


def decode_body(ids, inverse_vocabulary):
    return parse_dynamic_answer("".join(inverse_vocabulary[int(i)] for i in ids), strict=True)


def canonical_body(ids, vocabulary, inverse_vocabulary=None):
    inverse = inverse_vocabulary or {int(value): key for key, value in vocabulary.items()}
    result = []
    for value in ids:
        token = inverse[int(value)]
        if token.startswith(("<X_", "<Y_", "<Z_")) and token.endswith("_100>"):
            token = token[:-4] + "000>"
        result.append(int(vocabulary[token]))
    return result


def numeric_positions(n):
    return list(range(1, 7)) + [8 + 4 * i + axis for i in range(n) for axis in range(3)]


def fixed_composition(old, target, n):
    positions = [0] + [7 + 4 * i for i in range(n)]
    return len(old) == len(target) == 7 + 4 * n and all(old[i] == target[i] for i in positions)


def arrays_from_structure(structure):
    lattice = structure["lattice"]
    sites = structure["sites"]
    species = []
    for site in sites:
        if len(site["species"]) != 1 or abs(float(site["species"][0].get("occu", 1)) - 1) > 1e-8:
            raise ValueError("disordered terminal is outside the fixed-species edit contract")
        species.append(site["species"][0]["element"])
    return {
        "lengths": [lattice[k] for k in ("a", "b", "c")],
        "angles": [lattice[k] for k in ("alpha", "beta", "gamma")],
        "species": species,
        "frac_coords": [site["abc"] for site in sites],
    }
