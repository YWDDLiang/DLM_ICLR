"""Retained crystal DLM implementation; see docs/reproduction.md for the public workflow."""

import numpy as np


import torch


from pymatgen.core.structure import Structure


from pymatgen.core.lattice import Lattice


from pymatgen.analysis.graphs import StructureGraph


from pymatgen.analysis import local_env


CrystalNN = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)


def build_crystal(crystal_str, niggli=True, primitive=False):
    """Build crystal from cif string."""
    crystal = Structure.from_str(crystal_str, fmt="cif")

    if primitive:
        crystal = crystal.get_primitive_structure()

    if niggli:
        crystal = crystal.get_reduced_structure()

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    # match is gaurantteed because cif only uses lattice params & frac_coords
    # assert canonical_crystal.matches(crystal)
    return canonical_crystal


def build_crystal_graph(crystal, graph_method="crystalnn"):
    """ """

    if graph_method == "crystalnn":
        try:
            crystal_graph = StructureGraph.with_local_env_strategy(crystal, CrystalNN)
        except:
            crystalNN_tmp = local_env.CrystalNN(
                distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False, search_cutoff=10
            )
            crystal_graph = StructureGraph.with_local_env_strategy(crystal, crystalNN_tmp)
    elif graph_method == "none":
        pass
    else:
        raise NotImplementedError

    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(crystal.lattice.matrix, lattice_params_to_matrix(*lengths, *angles))

    edge_indices, to_jimages = [], []
    if graph_method != "none":
        for i, j, to_jimage in crystal_graph.graph.edges(data="to_jimage"):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    return frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms


def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)


def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


def process_one(crystal_str, niggli, primitive, graph_method, use_space_group=False, tol=0.01):
    crystal = build_crystal(crystal_str, niggli=niggli, primitive=primitive)
    frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms = build_crystal_graph(
        crystal, graph_method
    )

    num_atoms = torch.tensor([num_atoms])
    frac_coords = torch.tensor(frac_coords)
    lengths = torch.tensor(lengths)
    angles = torch.tensor(angles)
    data_dict = {
        "n_atom": num_atoms,
        "x_coord": frac_coords,
        "a_type": atom_types,
        "length": lengths.view(1, 3),
        "angle": angles.view(1, 3),
        "edge_indices": edge_indices,
        "to_jimages": to_jimages,
    }
    return frac_coords, atom_types, lengths, angles, num_atoms, edge_indices, to_jimages, data_dict
