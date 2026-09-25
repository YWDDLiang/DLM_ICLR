"""Retained crystal DLM implementation; see docs/reproduction.md for the public workflow."""

from __future__ import annotations

from dlm_iclr.runtime.capacity import MAX_ATOMS
import torch
from dlm_iclr._core.lattice_geometry import lattice_angle_rad
from dlm_iclr._core.periodic_geometry_ops import minimum_image_distances


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def _validate_generation_position_groups(
    generation_position_groups: list[list[int]],
    gen_length: int,
) -> list[list[int]]:
    normalized: list[list[int]] = []
    seen: set[int] = set()
    for group_idx, group in enumerate(generation_position_groups):
        if not group:
            raise ValueError(f"Generation schedule group {group_idx} is empty")
        normalized_group: list[int] = []
        for position in group:
            position = int(position)
            if not 0 <= position < gen_length:
                raise ValueError(f"Generation position {position} outside 0..{gen_length - 1}")
            if position in seen:
                raise ValueError(f"Generation position {position} appears in multiple schedule groups")
            seen.add(position)
            normalized_group.append(position)
        normalized.append(normalized_group)
    missing = sorted(set(range(int(gen_length))) - seen)
    if missing:
        preview = ",".join(str(value) for value in missing[:12])
        raise ValueError(
            f"Generation schedule does not cover every position; missing {len(missing)} positions ({preview})"
        )
    return normalized


def _build_token_mask(token_ids: list[int], vocab_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((vocab_size,), dtype=torch.bool, device=device)
    if token_ids:
        mask[torch.tensor(token_ids, dtype=torch.long, device=device)] = True
    return mask


def _prepare_atom_count_grammar(
    atom_count_grammar: dict | None,
    vocab_size: int,
    device: torch.device,
) -> dict | None:
    if atom_count_grammar is None:
        return None
    representation = str(atom_count_grammar.get("representation", "fixed_slot"))
    coord_token_ids = atom_count_grammar["coord_token_ids"]
    prepared = {
        "representation": representation,
        "body_offset": int(atom_count_grammar.get("body_offset", 0)),
        "max_atoms": int(atom_count_grammar.get("max_atoms", MAX_ATOMS)),
        "count_token_to_n": {
            int(token_id): int(num_atoms)
            for token_id, num_atoms in atom_count_grammar["count_token_to_n"].items()
        },
        "element_mask": _build_token_mask(atom_count_grammar["element_token_ids"], vocab_size, device),
        "coord_masks": {
            axis: _build_token_mask([int(item) for item in ids], vocab_size, device)
            for axis, ids in coord_token_ids.items()
        },
    }
    if representation == "dynamic_v1":
        prepared["eos_mask"] = _build_token_mask(
            [int(atom_count_grammar["eos_token_id"])],
            vocab_size,
            device,
        )
    else:
        pad_coord_token_ids = atom_count_grammar.get("pad_coord_token_ids", {})
        prepared["empty_mask"] = _build_token_mask(
            [int(atom_count_grammar["empty_token_id"])],
            vocab_size,
            device,
        )
        prepared["pad_coord_masks"] = {
            axis: _build_token_mask([int(token_id)], vocab_size, device)
            for axis, token_id in pad_coord_token_ids.items()
        }
    return prepared


def _apply_atom_count_grammar_mask(
    generation_logits: torch.Tensor,
    x: torch.Tensor,
    prompt_length: int,
    grammar: dict | None,
) -> None:
    if grammar is None:
        return
    min_value = torch.finfo(generation_logits.dtype).min
    max_atoms = int(grammar["max_atoms"])
    body_offset = int(grammar.get("body_offset", 0))
    count_token_to_n = grammar["count_token_to_n"]
    axes = ("X", "Y", "Z")
    if grammar.get("representation") == "dynamic_v1":
        for batch_idx in range(generation_logits.shape[0]):
            count_token_id = int(x[batch_idx, prompt_length + body_offset].detach().item())
            num_atoms = count_token_to_n.get(count_token_id)
            if num_atoms is None:
                continue
            for slot_index in range(max_atoms):
                base = body_offset + 7 + slot_index * 4
                if base + 3 >= generation_logits.shape[1]:
                    break
                if slot_index < num_atoms:
                    generation_logits[batch_idx, base].masked_fill_(~grammar["element_mask"], min_value)
                    for offset, axis in enumerate(axes, start=1):
                        generation_logits[batch_idx, base + offset].masked_fill_(
                            ~grammar["coord_masks"][axis],
                            min_value,
                        )
                else:
                    for offset in range(4):
                        generation_logits[batch_idx, base + offset].masked_fill_(
                            ~grammar["eos_mask"],
                            min_value,
                        )
        return
    for batch_idx in range(generation_logits.shape[0]):
        count_token_id = int(x[batch_idx, prompt_length + body_offset].detach().item())
        num_atoms = count_token_to_n.get(count_token_id)
        if num_atoms is None:
            continue
        for slot_index in range(max_atoms):
            base = body_offset + 7 + slot_index * 5
            if base + 4 >= generation_logits.shape[1]:
                break
            if slot_index < num_atoms:
                generation_logits[batch_idx, base + 1].masked_fill_(~grammar["element_mask"], min_value)
                for offset, axis in enumerate(axes, start=2):
                    generation_logits[batch_idx, base + offset].masked_fill_(
                        ~grammar["coord_masks"][axis],
                        min_value,
                    )
            else:
                generation_logits[batch_idx, base + 1].masked_fill_(~grammar["empty_mask"], min_value)
                for offset, axis in enumerate(axes, start=2):
                    generation_logits[batch_idx, base + offset].masked_fill_(
                        ~grammar["pad_coord_masks"][axis],
                        min_value,
                    )


def _apply_schema_masks(
    logits: torch.Tensor,
    x: torch.Tensor,
    prompt_length: int,
    gen_length: int,
    allowed_mask: torch.Tensor | None,
    prepared_atom_count_grammar: dict | None,
) -> None:
    generation_logits = logits[:, prompt_length : prompt_length + gen_length, :]
    if allowed_mask is not None:
        generation_logits.masked_fill_(~allowed_mask.unsqueeze(0), torch.finfo(logits.dtype).min)
    _apply_atom_count_grammar_mask(
        generation_logits,
        x,
        prompt_length,
        prepared_atom_count_grammar,
    )


def _mask_zero_lengths(
    generation_logits: torch.Tensor,
    constraints: dict,
    min_value: float,
) -> None:
    zero_length_token_ids_by_position = constraints.get("zero_length_token_ids_by_position", {})
    for rel_pos, token_id in zero_length_token_ids_by_position.items():
        rel_pos = int(rel_pos)
        if 0 <= rel_pos < generation_logits.shape[1]:
            generation_logits[:, rel_pos, int(token_id)] = min_value


def _apply_lattice_volume_mask(
    generation_logits: torch.Tensor,
    x: torch.Tensor,
    prompt_length: int,
    constraints: dict,
    min_value: float,
) -> None:
    angle_token_to_bin = constraints.get("angle_token_to_bin", {})
    gamma_bin_to_token_id = constraints.get("gamma_bin_to_token_id", {})
    min_lattice_rad = float(constraints.get("min_lattice_rad", 1e-4))
    body_offset = int(constraints.get("body_offset", 0))
    if body_offset + 6 >= generation_logits.shape[1]:
        return
    alpha_map = angle_token_to_bin.get("AA", {})
    beta_map = angle_token_to_bin.get("AB", {})
    for batch_idx in range(generation_logits.shape[0]):
        alpha_token = int(x[batch_idx, prompt_length + body_offset + 4].detach().item())
        beta_token = int(x[batch_idx, prompt_length + body_offset + 5].detach().item())
        alpha = alpha_map.get(alpha_token)
        beta = beta_map.get(beta_token)
        if alpha is None or beta is None:
            continue
        legal_count = 0
        for gamma, token_id in gamma_bin_to_token_id.items():
            if lattice_angle_rad(int(alpha), int(beta), int(gamma)) <= min_lattice_rad:
                generation_logits[batch_idx, body_offset + 6, int(token_id)] = min_value
            else:
                legal_count += 1
        if legal_count == 0:
            for token_id in gamma_bin_to_token_id.values():
                generation_logits[batch_idx, body_offset + 6, int(token_id)] = min_value


def _apply_duplicate_coordinate_mask(
    generation_logits: torch.Tensor,
    x: torch.Tensor,
    prompt_length: int,
    constraints: dict,
    min_value: float,
) -> None:
    coord_token_to_bin = constraints.get("coord_token_to_bin", {})
    z_bin_to_token_id = constraints.get("z_bin_to_token_id", {})
    count_token_to_n = constraints.get("count_token_to_n", {})
    x_token_to_bin = coord_token_to_bin.get("X", {})
    y_token_to_bin = coord_token_to_bin.get("Y", {})
    z_token_to_bin = coord_token_to_bin.get("Z", {})
    max_atoms = int(constraints.get("max_atoms", MAX_ATOMS))
    coord_period = constraints.get("coord_period")
    body_offset = int(constraints.get("body_offset", 0))

    def coord_key(value: int) -> int:
        if coord_period is None:
            return int(value)
        period = int(coord_period)
        if period <= 0:
            return int(value)
        return int(value) % period

    representation = str(constraints.get("representation", "fixed_slot"))
    for batch_idx in range(generation_logits.shape[0]):
        count_token_id = int(x[batch_idx, prompt_length + body_offset].detach().item())
        num_atoms = int(count_token_to_n.get(count_token_id, 0))
        if num_atoms <= 0:
            continue
        num_atoms = min(num_atoms, max_atoms)
        slots: list[tuple[int, tuple[int, int] | None, int | None]] = []
        for slot_index in range(num_atoms):
            if representation == "dynamic_v1":
                base = body_offset + 7 + slot_index * 4
                x_offset, y_offset, z_offset = 1, 2, 3
            else:
                base = body_offset + 7 + slot_index * 5
                x_offset, y_offset, z_offset = 2, 3, 4
            if base + z_offset >= generation_logits.shape[1]:
                break
            x_token = int(x[batch_idx, prompt_length + base + x_offset].detach().item())
            y_token = int(x[batch_idx, prompt_length + base + y_offset].detach().item())
            z_token = int(x[batch_idx, prompt_length + base + z_offset].detach().item())
            x_bin = x_token_to_bin.get(x_token)
            y_bin = y_token_to_bin.get(y_token)
            if x_bin is None or y_bin is None:
                slots.append((base + z_offset, None, None))
                continue
            xy = (coord_key(int(x_bin)), coord_key(int(y_bin)))
            z_bin = z_token_to_bin.get(z_token)
            slots.append((base + z_offset, xy, None if z_bin is None else coord_key(int(z_bin))))

        seen_by_xy: dict[tuple[int, int], set[int]] = {}
        for _z_position, xy, z_bin in slots:
            if xy is None or z_bin is None:
                continue
            seen_by_xy.setdefault(xy, set()).add(z_bin)

        for z_position, xy, z_bin in slots:
            if xy is None:
                continue
            banned_z_bins = set(seen_by_xy.get(xy, set()))
            if z_bin is not None:
                banned_z_bins.discard(z_bin)
            if not banned_z_bins:
                continue
            for candidate_z_bin, token_id in z_bin_to_token_id.items():
                if coord_key(int(candidate_z_bin)) in banned_z_bins:
                    generation_logits[batch_idx, z_position, int(token_id)] = min_value


def _lattice_matrix_from_token_ids(
    x: torch.Tensor,
    *,
    prompt_length: int,
    constraints: dict,
) -> torch.Tensor | None:
    body_offset = int(constraints.get("body_offset", 0))
    length_maps = constraints.get("length_token_to_bin", {})
    angle_maps = constraints.get("angle_token_to_bin", {})
    length_step = float(constraints.get("length_step", 0.1))
    length_values: list[float] = []
    for offset, prefix in zip((1, 2, 3), ("LA", "LB", "LC"), strict=True):
        token = int(x[prompt_length + body_offset + offset].detach().item())
        value = length_maps.get(prefix, {}).get(token)
        if value is None:
            return None
        length_values.append(float(value) * length_step)
    angle_values: list[float] = []
    for offset, prefix in zip((4, 5, 6), ("AA", "AB", "AG"), strict=True):
        token = int(x[prompt_length + body_offset + offset].detach().item())
        value = angle_maps.get(prefix, {}).get(token)
        if value is None:
            return None
        angle_values.append(float(value))
    a, b, c = length_values
    if min(a, b, c) <= 0.0:
        return None
    alpha, beta, gamma = [
        torch.deg2rad(torch.tensor(value, dtype=torch.float64, device=x.device)) for value in angle_values
    ]
    sin_gamma = torch.sin(gamma)
    if float(torch.abs(sin_gamma).detach().item()) < 1.0e-8:
        return None
    c_x = c * torch.cos(beta)
    c_y = c * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / sin_gamma
    c_z_squared = c * c - c_x * c_x - c_y * c_y
    if float(c_z_squared.detach().item()) <= 1.0e-10:
        return None
    zero = torch.zeros((), dtype=torch.float64, device=x.device)
    return torch.stack(
        [
            torch.stack((torch.as_tensor(a, dtype=torch.float64, device=x.device), zero, zero)),
            torch.stack(
                (
                    torch.as_tensor(b, dtype=torch.float64, device=x.device) * torch.cos(gamma),
                    torch.as_tensor(b, dtype=torch.float64, device=x.device) * sin_gamma,
                    zero,
                )
            ),
            torch.stack((c_x, c_y, torch.sqrt(c_z_squared))),
        ]
    )


def _aggregate_periodic_coordinate_alias_logits(
    generation_logits: torch.Tensor,
    constraints: dict,
    min_value: float,
    active_generation_mask: torch.Tensor | None,
) -> None:
    aliases = constraints.get("coordinate_alias_token_ids", {})
    body_offset = int(constraints.get("body_offset", 0))
    max_atoms = int(constraints.get("max_atoms", MAX_ATOMS))
    for slot in range(max_atoms):
        base = body_offset + 7 + 4 * slot
        for component, axis in enumerate(("X", "Y", "Z"), start=1):
            position = base + component
            if position >= generation_logits.shape[1] or axis not in aliases:
                continue
            canonical_id, alias_id = (int(value) for value in aliases[axis])
            for row in range(generation_logits.shape[0]):
                if active_generation_mask is not None and not bool(
                    active_generation_mask[row, position].detach().item()
                ):
                    continue
                canonical = generation_logits[row, position, canonical_id].clone()
                alias = generation_logits[row, position, alias_id].clone()
                generation_logits[row, position, canonical_id] = torch.logaddexp(canonical, alias)
                generation_logits[row, position, alias_id] = min_value


def _apply_pbc_min_distance_mask(
    generation_logits: torch.Tensor,
    x: torch.Tensor,
    prompt_length: int,
    constraints: dict,
    min_value: float,
    active_generation_mask: torch.Tensor | None,
    mask_id: int,
) -> set[tuple[int, int]]:
    no_legal_completion: set[tuple[int, int]] = set()
    coord_maps = constraints.get("coord_token_to_bin", {})
    coord_ids = constraints.get("coord_bin_to_token_id", {})
    count_token_to_n = constraints.get("count_token_to_n", {})
    threshold = float(constraints.get("pbc_min_distance_A", 0.5))
    image_radius = int(constraints.get("pbc_image_radius", 2))
    body_offset = int(constraints.get("body_offset", 0))
    period = int(constraints.get("coord_period", 100))
    if period <= 0:
        raise ValueError("coordinate period must be positive")
    for row in range(generation_logits.shape[0]):
        count_token = int(x[row, prompt_length + body_offset].detach().item())
        num_atoms = int(count_token_to_n.get(count_token, 0))
        if num_atoms <= 1:
            continue
        lattice = _lattice_matrix_from_token_ids(x[row], prompt_length=prompt_length, constraints=constraints)
        if lattice is None:
            continue
        for slot in range(num_atoms):
            z_position = body_offset + 10 + 4 * slot
            if z_position >= generation_logits.shape[1]:
                break
            if active_generation_mask is not None and not bool(
                active_generation_mask[row, z_position].detach().item()
            ):
                continue
            if int(x[row, prompt_length + z_position].detach().item()) != int(mask_id):
                continue
            x_position = body_offset + 8 + 4 * slot
            y_position = body_offset + 9 + 4 * slot
            x_bin = coord_maps.get("X", {}).get(int(x[row, prompt_length + x_position].detach().item()))
            y_bin = coord_maps.get("Y", {}).get(int(x[row, prompt_length + y_position].detach().item()))
            if x_bin is None or y_bin is None:
                continue
            other: list[list[float]] = []
            for prior in range(num_atoms):
                if prior == slot:
                    continue
                base = body_offset + 8 + 4 * prior
                bins = [
                    coord_maps[axis].get(int(x[row, prompt_length + base + axis_index].detach().item()))
                    for axis_index, axis in enumerate(("X", "Y", "Z"))
                ]
                if any(value is None for value in bins):
                    continue
                other.append([float(value) / period for value in bins])
            if not other:
                continue
            candidate_bins = sorted(int(value) for value in coord_ids.get("Z", {}))
            candidates = torch.tensor(
                [
                    [float(x_bin) / period, float(y_bin) / period, float(value) / period]
                    for value in candidate_bins
                ],
                dtype=torch.float64,
                device=x.device,
            )
            existing = torch.tensor(other, dtype=torch.float64, device=x.device)
            deltas = candidates.unsqueeze(1) - existing.unsqueeze(0)
            distances = minimum_image_distances(deltas, lattice, image_radius=image_radius)
            legal = distances.min(dim=1).values >= threshold
            if not bool(legal.any().item()):
                # Do not create an empty action set; the caller can record the
                # unresolved risk and retain the model distribution.
                no_legal_completion.add((int(row), int(z_position)))
                continue
            for candidate_bin, allowed in zip(candidate_bins, legal, strict=True):
                if not bool(allowed.detach().item()):
                    token_id = int(coord_ids["Z"][candidate_bin])
                    generation_logits[row, z_position, token_id] = min_value
    return no_legal_completion


def _apply_lightweight_decoding_masks(
    logits: torch.Tensor,
    x: torch.Tensor,
    prompt_length: int,
    gen_length: int,
    constraints: dict | None,
    active_generation_mask: torch.Tensor | None = None,
    mask_id: int = 126336,
) -> dict[str, set[tuple[int, int]]]:
    report: dict[str, set[tuple[int, int]]] = {"pbc_no_legal_completion": set()}
    if not constraints:
        return report
    generation_logits = logits[:, prompt_length : prompt_length + gen_length, :]
    min_value = torch.finfo(generation_logits.dtype).min
    if active_generation_mask is not None and active_generation_mask.shape != generation_logits.shape[:2]:
        raise ValueError("active_generation_mask must match [batch, generation length]")
    if constraints.get("canonicalize_periodic_alias"):
        _aggregate_periodic_coordinate_alias_logits(
            generation_logits,
            constraints,
            min_value,
            active_generation_mask,
        )
    if constraints.get("lattice_volume_mask"):
        _mask_zero_lengths(generation_logits, constraints, min_value)
        _apply_lattice_volume_mask(generation_logits, x, prompt_length, constraints, min_value)
    if constraints.get("duplicate_coordinate_mask"):
        _apply_duplicate_coordinate_mask(generation_logits, x, prompt_length, constraints, min_value)
    if constraints.get("pbc_min_distance_mask"):
        report["pbc_no_legal_completion"] = _apply_pbc_min_distance_mask(
            generation_logits,
            x,
            prompt_length,
            constraints,
            min_value,
            active_generation_mask,
            int(mask_id),
        )
    return report


def _model_logits(
    model,
    x: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prompt_index: torch.Tensor,
    cfg_scale: float,
    mask_id: int,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if cfg_scale > 0.0:
        un_x = x.clone()
        un_x[prompt_index] = mask_id
        model_input = torch.cat([x, un_x], dim=0)
        if attention_mask is None:
            attention_mask_input = None
        else:
            attention_mask_input = torch.cat([attention_mask, attention_mask], dim=0)
        logits = model(model_input, attention_mask=attention_mask_input).logits
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        return un_logits + (cfg_scale + 1.0) * (logits - un_logits)
    return model(x, attention_mask=attention_mask).logits
