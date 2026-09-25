"""Retained crystal DLM implementation; see docs/reproduction.md for the public workflow."""

from __future__ import annotations
import math
import numpy as np


TERMINAL_VERIFICATION_PROTOCOL = {
    "fresh_terminal_energy": True,
    "periodic_coordinate_wrap": True,
    "rigid_fractional_shift": [0.137, 0.271, 0.419],
    "energy_tolerance_eV_atom": 0.001,
}


COMMON_RELAXATION_PROTOCOL = {
    "model": "CHGNet-0.3.0",
    "optimizer": "FIRE",
    "relax_cell": True,
    "ase_filter": "FrechetCellFilter",
    "fmax": 0.1,
    "scalar_pressure": 0.0,
    "constant_volume": False,
    "hydrostatic_strain": False,
    "cell_mask": "all_six",
    "fire_dt": 0.1,
    "fire_maxstep": 0.2,
    "stress_tolerance_GPa": 0.5,
    "max_steps": 500,
}


LABEL_GEOMETRY_PROTOCOL = {
    "minimum_periodic_distance_A": 0.5,
    "distance_tolerance_A": 1e-8,
    "minimum_volume_A3": 1e-10,
    "lattice_basis": "LLL_unimodular",
    "contact_enumeration": "complete_reciprocal_bound",
    "max_pair_images": 4_000_000,
    "unimodular_rebuild_from_original": True,
    "numeric_error_budget_A": 1e-9,
}


def compare_energies(stored, fresh, *, tolerance=0.001):
    values = [float(stored), *map(float, fresh)]
    if len(fresh) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError("three finite terminal representation energies are required")
    delta = fresh[0] - stored
    spread = max(fresh) - min(fresh)
    return {
        "stored_energy_matches_fresh": abs(delta) <= tolerance,
        "periodic_representation_consistent": spread <= tolerance,
        "fresh_minus_stored_eV_atom": delta,
        "representation_spread_eV_atom": spread,
        "status": "consistent" if abs(delta) <= tolerance and spread <= tolerance else "inconsistent",
    }


def check_terminal_energy(model, original, stored_energy):
    from pymatgen.core import Structure

    wrapped = Structure(original.lattice, original.species, np.mod(original.frac_coords, 1.0))
    shifted = Structure(
        original.lattice, original.species, np.mod(original.frac_coords + [0.137, 0.271, 0.419], 1.0)
    )
    scores = []

    def array(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=float)

    for name, structure in (
        ("stored_geometry", original),
        ("wrapped_geometry", wrapped),
        ("shifted_geometry", shifted),
    ):
        result = model.predict_structure(structure, task="efs")
        if isinstance(result, list):
            result = result[0]
        energy = float(result["e"])
        forces, stress = array(result["f"]), array(result["s"])
        if not np.isfinite(forces).all() or not np.isfinite(stress).all():
            raise ValueError("nonfinite fresh force or stress")
        scores.append(
            {
                "representation": name,
                "energy_eV_atom": energy,
                "force_max_eV_A": float(np.linalg.norm(forces, axis=-1).max()),
                "stress_max_GPa": float(np.abs(stress).max()),
            }
        )
    return {
        "stored_terminal_energy_eV_atom": stored_energy,
        "scores": scores,
        **compare_energies(stored_energy, [r["energy_eV_atom"] for r in scores]),
    }
