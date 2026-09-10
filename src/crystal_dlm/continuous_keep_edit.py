"""Retained crystal DLM implementation; see docs/method.md for the public workflow."""

from __future__ import annotations
import copy
import numpy as np


def structure_of(record):
    from pymatgen.core import Structure
    from crystal_dlm.dynamic_crystal import parse_dynamic_answer, arrays_to_structure

    if record.get("structure") is not None:
        return Structure.from_dict(record["structure"])
    if record.get("body"):
        return arrays_to_structure(parse_dynamic_answer(record["body"], strict=True))
    raise ValueError("no_structure_or_token_body")


def geometry(record):
    from crystal_dlm.expert_edit_data import arrays_from_structure, certify_geometry

    try:
        return certify_geometry(arrays_from_structure(structure_of(record).as_dict()))
    except (ValueError, TypeError, KeyError, FloatingPointError) as error:
        return dict(valid=False, certified=True, reason=str(error))


def commit_patch(current_record, old_tokens, new_tokens, inverse, *, editable=True):
    """Unchanged numeric fields retain their exact stored continuous values."""
    from pymatgen.core import Lattice
    from crystal_dlm.expert_edit_data import decode_body, fixed_composition

    if not editable:
        return copy.deepcopy(current_record), dict(applied=False, reason="no_matching_token_view", changed=[])
    if list(old_tokens) == list(new_tokens):
        return copy.deepcopy(current_record), dict(applied=False, reason="KEEP", changed=[])
    try:
        old = decode_body(old_tokens, inverse)
        new = decode_body(new_tokens, inverse)
        n = len(old["species"])
        if not fixed_composition(old_tokens, new_tokens, n):
            raise ValueError("fixed_atom_count_or_species_changed")
        native = structure_of(current_record)
        if [str(s.specie) for s in native] != old["species"]:
            raise ValueError("continuous_and_token_site_order_differ")
        changed = []
        for i, (a, b) in enumerate(zip(old_tokens, new_tokens, strict=True)):
            if a == b:
                continue
            if i >= 8:
                site, axis = divmod(i - 8, 4)
                if (
                    axis <= 2
                    and float(old["frac_coords"][site][axis]) % 1.0
                    == float(new["frac_coords"][site][axis]) % 1.0
                ):
                    continue
            changed.append(i)
        if not changed:
            return copy.deepcopy(current_record), dict(
                applied=False, reason="periodic_equivalent_KEEP", changed=[]
            )
        structure = copy.deepcopy(current_record.get("structure") or native.as_dict())
        if current_record.get("structure") is None:
            structure["lattice"]["pbc"] = list(structure["lattice"]["pbc"])
        lattice_changed = any(1 <= i <= 6 for i in changed)
        if lattice_changed:
            lengths, angles = list(native.lattice.abc), list(native.lattice.angles)
            for i in changed:
                if 1 <= i <= 3:
                    lengths[i - 1] = new["lengths"][i - 1]
                if 4 <= i <= 6:
                    angles[i - 4] = new["angles"][i - 4]
            lattice = Lattice.from_parameters(*lengths, *angles)
            structure["lattice"] = dict(
                matrix=lattice.matrix.tolist(),
                pbc=list(lattice.pbc),
                a=lattice.a,
                b=lattice.b,
                c=lattice.c,
                alpha=lattice.alpha,
                beta=lattice.beta,
                gamma=lattice.gamma,
                volume=lattice.volume,
            )
        matrix = np.asarray(structure["lattice"]["matrix"], dtype=float)
        changed_sites = set()
        for position in changed:
            if position >= 8:
                site, axis = divmod(position - 8, 4)
                if axis > 2:
                    raise ValueError("non_numeric_edit")
                structure["sites"][site]["abc"][axis] = float(new["frac_coords"][site][axis]) % 1.0
                changed_sites.add(site)
        if not lattice_changed:
            displacement = np.asarray([site["abc"] for site in structure["sites"]]) - native.frac_coords
            displacement -= np.round(displacement)
            relative = displacement - displacement[0]
            relative -= np.round(relative)
            if np.max(np.abs(relative)) <= 1e-12:
                return copy.deepcopy(current_record), dict(
                    applied=False, reason="rigid_translation_KEEP", changed=changed
                )
        for site in range(n):
            if lattice_changed or site in changed_sites:
                structure["sites"][site]["xyz"] = (
                    np.asarray(structure["sites"][site]["abc"]) @ matrix
                ).tolist()
        result = copy.deepcopy(current_record)
        result.update(structure=structure, body=None, success=True, reason=None)
        result.pop("body_token_ids", None)
        check = geometry(result)
        if check["valid"] is not True:
            return copy.deepcopy(current_record), dict(
                applied=False, reason="hybrid_geometry_revert", changed=changed, geometry=check
            )
        return result, dict(
            applied=True,
            reason="changed_numeric_fields_only",
            changed=changed,
            lattice_changed=lattice_changed,
            changed_sites=sorted(changed_sites),
            geometry=check,
        )
    except (ValueError, KeyError, TypeError, IndexError, FloatingPointError) as error:
        return copy.deepcopy(current_record), dict(
            applied=False, reason="invalid_patch_revert:" + str(error), changed=[]
        )
