"""Choose an equivalent token teacher representation near a retained draft.

Only an integer-bin periodic translation and a same-element site permutation
are allowed. The teacher lattice and all relative geometry are preserved. The
result still receives its own physical measurement before use as supervision.
This is offline target registration, never retrieval during inference.
"""
from collections import Counter

import numpy as np
from pymatgen.core import Lattice
from pymatgen.util.coord import pbc_shortest_vectors
from scipy.optimize import linear_sum_assignment


def align_teacher_to_anchor(teacher_tokens, anchor_tokens, tables):
    teacher = list(teacher_tokens); anchor = list(anchor_tokens)
    if len(teacher) != len(anchor) or len(teacher) < 11 or (len(teacher)-7) % 4:
        raise ValueError('Two complete crystals with matching atom counts required')
    n = (len(teacher)-7)//4
    reverse = {f: {token: value for value, token in values.items()} for f, values in tables.items()}
    species = np.asarray([teacher[7+4*i] for i in range(n)])
    anchor_species = np.asarray([anchor[7+4*i] for i in range(n)])
    if teacher[0] != anchor[0] or Counter(species) != Counter(anchor_species):
        raise ValueError('Teacher and anchor chemical inventories differ')
    try:
        lengths = [reverse[f][teacher[i]]*.1 for i, f in enumerate(('LA','LB','LC'), 1)]
        angles = [reverse[f][teacher[i]] for i, f in enumerate(('AA','AB','AG'), 4)]
        t = np.asarray([[reverse[f][teacher[8+4*i+a]] % 100 for a,f in enumerate('XYZ')] for i in range(n)], dtype=int)
        r = np.asarray([[reverse[f][anchor[8+4*i+a]] % 100 for a,f in enumerate('XYZ')] for i in range(n)], dtype=int)
    except KeyError as error:
        raise ValueError('Complete typed lattice and coordinates required') from error
    lattice = Lattice.from_parameters(*lengths, *angles)
    mismatch = anchor_species[:,None] != species[None,:]
    starts = {(0,0,0)}
    starts.update(tuple(((r[i]-t[j]) % 100).tolist()) for i in range(n) for j in range(n) if not mismatch[i,j])
    best = None; inverse = np.linalg.inv(lattice.matrix)
    for start in sorted(starts):
        shift = np.asarray(start, dtype=int); seen = set()
        for _ in range(6):
            key = tuple(shift.tolist())
            if key in seen: break
            seen.add(key)
            vectors, d2 = pbc_shortest_vectors(lattice, r/100, ((t+shift) % 100)/100, return_d2=True)
            ii, jj = linear_sum_assignment(np.where(mismatch, 1e30, d2))
            candidate = (float(d2[ii,jj].mean()), key, tuple(jj.tolist()))
            if best is None or candidate < best: best = candidate
            correction = vectors[ii,jj].mean(axis=0) @ inverse
            shift = np.rint(shift-correction*100).astype(int) % 100
    score, shift, order = best
    coords = (t[list(order)]+np.asarray(shift)) % 100
    result = teacher.copy()
    for i,j in enumerate(order):
        if species[j] != anchor_species[i]: raise AssertionError('Registration crossed species')
        result[7+4*i] = teacher[7+4*j]
        for a,f in enumerate('XYZ'): result[8+4*i+a] = tables[f][int(coords[i,a])]
    if result[:7] != teacher[:7]: raise AssertionError('Registration changed teacher lattice')
    # Verify relative periodic geometry in exact integer arithmetic, before any
    # floating-point structure conversion or independent physical remeasurement.
    if not np.array_equal((coords[:,None]-coords[None,:]) % 100,
                          (t[list(order),None]-t[np.asarray(order)[None,:]]) % 100):
        raise AssertionError('Registration changed relative geometry')
    return result, {'source_site_for_output_site': list(order), 'translation_bins': list(shift),
                    'coordinate_bins': 100, 'teacher_lattice_preserved': True,
                    'exact_periodic_relative_geometry_preserved': True,
                    'anchor_coordinate_RMS_A_in_teacher_cell': float(np.sqrt(score)),
                    'alignment_is_approximate': True,
                    'physical_remeasurement_required': True,
                    'not_an_inference_lookup': True}
