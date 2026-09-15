"""Retained crystal DLM implementation; see docs/method.md for the public workflow."""

from itertools import product
import torch
from dlm_iclr._core.fixed_slot import MASK_TOKEN_ID
from dlm_iclr._core.llada_generation import _lattice_matrix_from_token_ids
from dlm_iclr._core.post_refine_contract import derived_seed


def failed_sites(body, n, failure):
    sites = {(int(p) - 8) // 4 for p in failure.get("failure_positions", []) if 8 <= int(p) < len(body)}
    if not sites:
        sites = {i for i in range(n) if MASK_TOKEN_ID in body[8 + 4 * i : 11 + 4 * i]}
    return sorted(sites or set(range(n)))


def neighbor_sites(body, n, centers, constraints, radius=2.0):
    """All nearby sites plus four nearest; unknown Z is projected out."""
    cell = _lattice_matrix_from_token_ids(torch.tensor(body), prompt_length=0, constraints=constraints)
    if cell is None or not bool(torch.isfinite(cell).all()):
        return list(range(n))
    cell = cell.double()
    selected = set(centers)
    maps = constraints["coord_token_to_bin"]
    coords = [[maps[a].get(int(body[8 + 4 * i + j])) for j, a in enumerate("XYZ")] for i in range(n)]
    shifts = torch.tensor(list(product(range(-2, 3), repeat=3)), dtype=torch.double)
    for center in centers:
        distances = []
        for other in range(n):
            if other == center:
                continue
            missing = [j for j in range(3) if coords[center][j] is None or coords[other][j] is None]
            delta = torch.tensor(
                [
                    (coords[center][j] - coords[other][j]) / 100.0 if j not in missing else 0.0
                    for j in range(3)
                ],
                dtype=torch.double,
            )
            vectors = (shifts + delta) @ cell
            if missing:
                basis = cell[missing]
                vectors = vectors - (vectors @ torch.linalg.pinv(basis)) @ basis
            distances.append((float(torch.linalg.vector_norm(vectors, dim=1).min()), other))
        distances.sort()
        selected.update(i for distance, i in distances if distance <= radius)
        selected.update(i for _, i in distances[:4])
    return sorted(selected)


def reopen(body, n, sites, *, cell=False, z_only=False):
    result = list(body)
    positions = (list(range(1, 7)) if cell else []) + [
        8 + 4 * i + j for i in sites for j in ([2] if z_only else range(3))
    ]
    for position in positions:
        result[position] = MASK_TOKEN_ID
    return result, positions


def construct_cascade(
    model,
    tokenizer,
    task,
    runtime,
    *,
    construct,
    constraints,
    repair_constraints,
    geometry_api,
    complete_geometry,
    adaptive_lattice=False,
):
    n = int(task["plan_state"]["N"])
    initial = None
    episodes = []
    stages = ["draft", "failed_XYZ", "neighbor_XYZ", "all_numeric", "final_Z_relaxed"]
    centers = []
    opened = []
    lattice_recoveries = 0
    gamma_last = False
    for stage_index, stage in enumerate(stages):
        relaxed = stage == "final_Z_relaxed"
        seed = (
            None
            if stage_index == 0
            else derived_seed(str(task["body_noise_seed"]), "geometry_recovery_" + stage, 1)
        )
        try:
            body, metadata = construct(
                model,
                tokenizer,
                [task],
                runtime,
                constraints=constraints,
                geometry_api=geometry_api,
                initial_body=initial,
                noise_seed_override=seed,
                relax_final_z=relaxed,
                **({"lattice_gamma_last": True} if gamma_last else {}),
            )
            support = complete_geometry(body[0].tolist())
            if support["supported"] or (relaxed and support.get("reason") == "native_pair_below_0.5A"):
                episodes.append(
                    {
                        "stage": stage,
                        "seed": seed,
                        "opened_positions": opened,
                        "completed": True,
                        "complete_geometry": support,
                        "distance_hard_limit_relaxed": relaxed,
                    }
                )
                metadata.update(
                    complete_geometry=support,
                    construction_recovery={
                        "schema": "DLM_geometry_three_stage_v1",
                        "episodes": episodes,
                        "recoveries_used": stage_index,
                        "final_Z_relaxed": relaxed,
                        "Plan_replacement_or_resampling": False,
                        "relaxed_generation_is_not_physical_validity": True,
                        "neighbor_radius_A": 2.0,
                        "adaptive_lattice_recovery": adaptive_lattice,
                        "lattice_recoveries": lattice_recoveries,
                        "original_exact_duplicate_guard_retained_for_refiner_graph": True,
                    },
                )
                return body, metadata
            error = geometry_api.GeometryNoLegalSupport(
                {"reason": support["reason"], "complete_geometry": support}, body, 0
            )
        except geometry_api.GeometryNoLegalSupport as caught:
            error = caught
        partial = error.partial_canvas[0, error.prompt_length :].tolist()
        episodes.append(
            {
                "stage": stage,
                "seed": seed,
                "opened_positions": opened,
                "completed": False,
                "failure": error.to_dict(),
            }
        )
        if stage_index == 4:
            error.details["recovery_episodes"] = episodes
            raise error
        centers = sorted(set(centers) | set(failed_sites(partial, n, error.details)))
        if adaptive_lattice and "lattice" in error.details.get("reason", ""):
            # Gamma can have been revealed before alpha/beta. Reopening XYZ
            # cannot repair that cell. First resample gamma conditional on the
            # other angles; if needed resample the cell with gamma revealed last.
            lattice_recoveries += 1
            initial, opened = reopen(partial, n, range(n), cell=lattice_recoveries > 1)
            initial[6] = MASK_TOKEN_ID
            opened = sorted(set(opened) | {6})
            gamma_last = True
            stages[stage_index + 1] = "conditional_gamma" if lattice_recoveries == 1 else "conditional_cell"
        elif stage_index == 0:
            initial, opened = reopen(partial, n, centers)
        elif stage_index == 1:
            sites = neighbor_sites(partial, n, centers, repair_constraints)
            initial, opened = reopen(partial, n, sites)
        elif stage_index == 2:
            initial, opened = reopen(partial, n, range(n), cell=True)
        else:
            # Only the last coordinate group loses distance support. Lattice,
            # element identities and completed X/Y remain exactly as generated.
            initial, opened = reopen(partial, n, range(n), z_only=True)
