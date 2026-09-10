"""Retained crystal DLM implementation; see docs/method.md for the public workflow."""

from __future__ import annotations


import itertools


import math


import os


from typing import Any


import numpy as np


from crystal_dlm.terminal_energy_consistency import LABEL_GEOMETRY_PROTOCOL, check_terminal_energy


EV_A3_TO_GPA = 160.21766208


_OPT_STATUS: dict[str, Any] = {}


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def finite_scalar(value):
    values = array(value).reshape(-1)
    if len(values) != 1 or not np.isfinite(values).all():
        raise ValueError("energy must be one finite scalar")
    return float(values[0])


def force_and_stress(forces, stress, *, stress_unit):
    forces, stress = array(forces), array(stress)
    if forces.ndim != 2 or forces.shape[-1] != 3 or not len(forces):
        raise ValueError("forces must have shape [N,3]")
    if stress.shape not in ((3, 3), (6,)):
        raise ValueError("stress must be a matrix or Voigt six-vector")
    if not np.isfinite(forces).all() or not np.isfinite(stress).all():
        raise ValueError("nonfinite force/stress")
    if stress_unit == "eV/A3":
        stress = stress * EV_A3_TO_GPA
    elif stress_unit != "GPa":
        raise ValueError("unknown stress unit")
    norms = np.linalg.norm(forces, axis=-1)
    return {
        "force_max_eV_A": float(norms.max()),
        "force_rms_eV_A": float(np.sqrt(np.mean(norms**2))),
        "stress_max_GPa": float(np.abs(stress).max()),
        "stress_GPa": stress.tolist(),
    }


def structure_from_record(record):
    if record.get("structure") is not None:
        from pymatgen.core import Structure

        return Structure.from_dict(record["structure"])
    from crystal_dlm.dynamic_crystal import parse_dynamic_answer, arrays_to_structure

    return arrays_to_structure(parse_dynamic_answer(record["body"], strict=True))


class GeometryCertificationUnavailable(RuntimeError):
    pass


class InvalidPeriodicGeometry(ValueError):
    pass


def _validate_structure_geometry(structure):
    lattice, coords = array(structure.lattice.matrix), array(structure.frac_coords)
    if lattice.shape != (3, 3) or coords.shape != (int(structure.num_sites), 3):
        raise InvalidPeriodicGeometry("invalid periodic geometry dimensions")
    volume = float(abs(np.linalg.det(lattice)))
    if (
        not np.isfinite(lattice).all()
        or not np.isfinite(coords).all()
        or not math.isfinite(volume)
        or volume <= 1e-10
    ):
        raise InvalidPeriodicGeometry("nonfinite or degenerate periodic structure")
    from pymatgen.core import Lattice

    try:
        reduced = np.asarray(Lattice(lattice).get_lll_reduced_lattice().matrix)
        transform = reduced @ np.linalg.inv(lattice)
        rounded = np.rint(transform)
        # Any nonzero integer image is already a valid short-contact witness;
        # it does not require the full coordinate change to be well conditioned.
        if np.isfinite(rounded).all() and np.max(np.abs(rounded)) <= 1_000_000_000:
            images_from_original = rounded @ lattice
            component_error = (np.abs(rounded) @ np.abs(lattice)) * np.finfo(float).eps * 16
            upper_lengths = np.linalg.norm(images_from_original, axis=-1) * (
                1 + 4 * np.finfo(float).eps
            ) + np.linalg.norm(component_error, axis=-1)
            if bool(((upper_lengths < 0.5 - 1e-8) & np.any(rounded != 0, axis=-1)).any()):
                raise InvalidPeriodicGeometry("periodic geometry violates the common 0.5 Angstrom support")
        if (
            not np.isfinite(transform).all()
            or np.max(np.abs(rounded)) > 1_000_000_000
            or not np.allclose(transform, rounded, rtol=0, atol=1e-7)
        ):
            raise GeometryCertificationUnavailable("LLL basis change is not certified integral")
        a, b, c, d, e, f, g, h, i = map(int, rounded.reshape(-1))
        determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        if abs(determinant) != 1:
            raise GeometryCertificationUnavailable("LLL basis change is not unimodular")
        # Use the original represented crystal and the certified integer map,
        # never the approximately returned LLL vectors as a replacement cell.
        reduced = rounded @ lattice
        integer_inverse = (
            np.asarray(
                [
                    [e * i - f * h, c * h - b * i, b * f - c * e],
                    [f * g - d * i, a * i - c * g, c * d - a * f],
                    [d * h - e * g, b * g - a * h, a * e - b * d],
                ],
                dtype=float,
            )
            * determinant
        )
        numeric_bound = max(
            float(np.max(np.abs(rounded) @ np.abs(lattice))),
            float(np.max(np.abs(integer_inverse))) * float(np.linalg.norm(reduced, axis=-1).max()),
        )
        if numeric_bound * np.finfo(float).eps * 32 > 1e-9:
            raise GeometryCertificationUnavailable(
                "integer basis transformation exceeds the numerical error budget"
            )
        inverse = np.linalg.inv(reduced)
        reduced_coords = np.mod(coords, 1.0) @ integer_inverse
        radii_float = np.ceil(0.5 + 0.5 * np.linalg.norm(inverse, axis=0) + 1e-12)
        if not np.isfinite(radii_float).all() or np.max(radii_float) > 1_000_000:
            raise GeometryCertificationUnavailable("periodic contact bound is not numerically established")
    except (GeometryCertificationUnavailable, InvalidPeriodicGeometry):
        raise
    except Exception as error:
        raise GeometryCertificationUnavailable("periodic contact reduction could not be certified") from error
    minimum = float(np.linalg.norm(reduced, axis=-1).min())
    if minimum < 0.5 - 1e-8:
        raise InvalidPeriodicGeometry("periodic geometry violates the common 0.5 Angstrom support")
    radii = radii_float.astype(int)
    images = math.prod(2 * int(radius) + 1 for radius in radii)
    if len(coords) ** 2 * images > LABEL_GEOMETRY_PROTOCOL["max_pair_images"]:
        raise GeometryCertificationUnavailable(
            "periodic contact certification exceeded its explicit work budget"
        )
    delta = reduced_coords[:, None] - reduced_coords[None, :]
    delta -= np.round(delta)
    shifts = itertools.product(*(range(-int(radius), int(radius) + 1) for radius in radii))
    while True:
        chunk = list(itertools.islice(shifts, 256))
        if not chunk:
            break
        offsets = np.asarray(chunk, dtype=float)
        distances = np.linalg.norm((delta[:, :, None, :] + offsets) @ reduced, axis=-1)
        if not np.isfinite(distances).all():
            raise GeometryCertificationUnavailable("periodic distance computation is nonfinite")
        zero = np.flatnonzero((offsets == 0).all(-1))
        if len(zero):
            distances[np.arange(len(coords)), np.arange(len(coords)), int(zero[0])] = np.inf
        minimum = min(minimum, float(distances.min()))
        if minimum < 0.5 - 1e-8:
            raise InvalidPeriodicGeometry("periodic geometry violates the common 0.5 Angstrom support")
    return minimum


def validate_structure_geometry(structure):
    try:
        return _validate_structure_geometry(structure)
    except (InvalidPeriodicGeometry, GeometryCertificationUnavailable):
        raise
    except Exception as error:
        raise GeometryCertificationUnavailable("periodic certification could not complete") from error


def label_record(
    record,
    *,
    model,
    optimizer,
    structure_factory=structure_from_record,
    fmax=0.1,
    stress_tolerance=0.5,
    max_steps=500,
    optimizer_status=None,
    terminal_energy_checker=check_terminal_energy,
):
    result = {
        key: record.get(key)
        for key in ("trajectory_id", "group_id", "source_row_idx", "source_split", "endpoint")
    }
    result.update(
        raw_energy=None,
        terminal_energy=None,
        gap=None,
        verified=False,
        status="unknown",
        error=None,
        raw=None,
        terminal=None,
        actual_steps=None,
        optimizer_converged=None,
        final_structure=None,
    )
    if not record.get("success", False):
        result["status"] = "generation_failure"
        result["error"] = (record.get("trace") or {}).get("failure")
        return result
    try:
        structure = structure_factory(record)
        count = int(structure.num_sites)
        if count < 1:
            raise ValueError("empty structure")
        result["raw_min_distance_A"] = validate_structure_geometry(structure)
    except GeometryCertificationUnavailable as error:
        result.update(status="worker_error", error=f"{type(error).__name__}: {error}")
        return result
    except Exception as error:
        result.update(status="invalid_raw", error=f"{type(error).__name__}: {error}")
        return result
    try:
        raw = model.predict_structure(structure, task="efs")
        if isinstance(raw, list):
            raw = raw[0]
        result["raw_energy"] = finite_scalar(raw["e"])  # CHGNet predict returns eV/atom.
        result["raw"] = force_and_stress(raw["f"], raw["s"], stress_unit="GPa")
        if optimizer_status is not None:
            optimizer_status.clear()
        relaxed = optimizer.relax(
            structure,
            fmax=fmax,
            steps=max_steps,
            relax_cell=True,
            ase_filter="FrechetCellFilter",
            verbose=False,
        )
        final = relaxed["final_structure"]
        trajectory = relaxed["trajectory"]
        if int(final.num_sites) != count or final.composition != structure.composition:
            raise ValueError("relaxation changed the fixed composition")
        geometry_valid = True
        try:
            result["terminal_min_distance_A"] = validate_structure_geometry(final)
        except ValueError as error:
            geometry_valid = False
            result["terminal_geometry_error"] = str(error)
        energies = list(trajectory.energies)
        if not energies:
            raise ValueError("missing relaxation energy trajectory")
        result["terminal_energy"] = finite_scalar(energies[-1]) / count
        result["terminal"] = force_and_stress(
            trajectory.forces[-1], trajectory.stresses[-1], stress_unit="eV/A3"
        )
        result["gap"] = result["raw_energy"] - result["terminal_energy"]
        first_energy = finite_scalar(energies[0]) / count
        result["raw_vs_trajectory_first_delta"] = result["raw_energy"] - first_energy
        status = optimizer_status or relaxed.get("optimizer_status") or {}
        result["actual_steps"] = status.get("steps")
        result["optimizer_converged"] = status.get("converged")
        result["trajectory_frames"] = len(energies)
        result["relaxation_trajectory"] = {
            "energies_eV_atom": [finite_scalar(e) / count for e in energies],
            "forces_eV_A": [array(f).tolist() for f in trajectory.forces],
            "stresses_GPa": [(array(s) * EV_A3_TO_GPA).tolist() for s in trajectory.stresses],
            "cells_A": [array(v).tolist() for v in getattr(trajectory, "cells", [])],
            "positions_A": [array(v).tolist() for v in getattr(trajectory, "atom_positions", [])],
        }
        result["final_structure"] = final.as_dict()
        physical = (
            result["terminal"]["force_max_eV_A"] <= fmax + 1e-8
            and result["terminal"]["stress_max_GPa"] <= stress_tolerance + 1e-8
        )
        same_energy = abs(result["raw_vs_trajectory_first_delta"]) <= 0.001
        monotone = result["gap"] >= -0.001
        # Missing optimizer status is explicit, never synthesized as a success.
        stop_verified = result["optimizer_converged"] is True
        preliminary_verified = bool(
            geometry_valid and physical and same_energy and monotone and stop_verified
        )
        if preliminary_verified:
            result["terminal_consistency"] = terminal_energy_checker(model, final, result["terminal_energy"])
        result["verified"] = bool(
            preliminary_verified and result["terminal_consistency"]["status"] == "consistent"
        )
        if not same_energy:
            result["status"] = "energy_protocol_mismatch"
        elif not monotone:
            result["status"] = "relaxation_energy_increased"
        elif not geometry_valid:
            result["status"] = "invalid_terminal"
        elif not physical:
            result["status"] = "not_converged"
        elif not stop_verified:
            result["status"] = "optimizer_stop_unverified"
        elif result["terminal_consistency"]["status"] != "consistent":
            result["status"] = "terminal_consistency_unverified"
        else:
            result["status"] = "verified"
    except GeometryCertificationUnavailable as error:
        result.update(status="worker_error", verified=False, error=f"{type(error).__name__}: {error}")
    except InvalidPeriodicGeometry as error:
        result.update(status="invalid_terminal", verified=False, error=f"{type(error).__name__}: {error}")
    except Exception as error:
        result.update(status="evaluation_error", verified=False, error=f"{type(error).__name__}: {error}")
    return result


def recorded_fire_class(FIRE):
    """Construct the pinned optimizer independently of GPU/model loading."""

    class RecordedFIRE(FIRE):
        def converged(self, *args, **kwargs):
            native = super().converged(*args, **kwargs)
            # ASE 3.28 routes both run() and this compatibility method through
            # gradient_converged(); older ASE calls converged() directly.
            return native if hasattr(FIRE, "gradient_converged") else self.physical_convergence(native)

        def gradient_converged(self, gradient):
            return self.physical_convergence(super().gradient_converged(gradient))

        def physical_convergence(self, native):
            if os.environ.get("RSI_JOINT_PHYSICAL_STOP") != "1":
                return native
            from crystal_dlm.ranked_feedback import joint_stop_status

            atoms = self.atoms.atoms if hasattr(self.atoms, "atoms") else self.atoms
            physical = joint_stop_status(
                atoms.get_forces(apply_constraint=False),
                np.asarray(atoms.get_stress(voigt=False, apply_constraint=False)) * EV_A3_TO_GPA,
                fmax=float(self.fmax),
                stress_tolerance=float(os.environ.get("RSI_STRESS_TOLERANCE", ".5")),
            )
            _OPT_STATUS.update(filter_converged=bool(native), **physical)
            # Catch a collapsing cell without spending the remainder of 1000 steps.
            if self.nsteps % 25 == 0 or physical["physical_converged"]:
                from pymatgen.io.ase import AseAtomsAdaptor

                validate_structure_geometry(AseAtomsAdaptor.get_structure(atoms))
            return physical["physical_converged"]

        def run(self, *args, **kwargs):
            outcome = super().run(*args, **kwargs)
            _OPT_STATUS.update(steps=int(self.nsteps), converged=None if outcome is None else bool(outcome))
            return outcome

    return RecordedFIRE
