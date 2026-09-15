"""Remask-aware warm-up from known dataset corruptions, not physical defect labels."""

from __future__ import annotations
import torch
from dlm_iclr.c2.scope import (
    feasible_modes,
    feasible_counts,
    COUNTS,
    action_positions,
    normalized_choices,
    ordered_site_logprob,
)
from dlm_iclr.c2.warmup_forward import forward_e


def warm_example(row, schema, generator, *, mask_id, available_calls, forced_mode=None):
    n = row["plan_state"]["N"]
    target = list(row["body_token_ids"])
    modes = feasible_modes(n, available_calls)
    mode = (
        forced_mode
        if forced_mode is not None
        else modes[int(torch.randint(len(modes), (), generator=generator))]
    )
    if mode not in modes:
        raise ValueError("Synthetic mode exceeds the registered warm-up budget")
    count_index = None
    if mode == 1:
        counts = feasible_counts(n, available_calls)
        count_index = counts[int(torch.randint(len(counts), (), generator=generator))]
        sites = torch.randperm(n, generator=generator).tolist()[: COUNTS[count_index]]
    elif mode in (2, 3):
        sites = list(range(n))
    else:
        sites = []
    positions = action_positions(n, mode, sites)
    old = target.copy()
    # Coordinate corruption is of known origin, not evidence of phase instability.
    for position in positions:
        if position < 7:
            continue
        choices = [x for x in schema.ids(position) if x != target[position]]
        old[position] = choices[int(torch.randint(len(choices), (), generator=generator))]
    if mode == 3:
        # Changing a positive length preserves representability of the source
        # angle Gram matrix. No physical success label is inferred from this.
        position = 1 + int(torch.randint(3, (), generator=generator))
        choices = [
            token
            for value, token in schema.tables[schema.family(position)].items()
            if value > 0 and token != target[position]
        ]
        if not choices:
            raise ValueError("Vocabulary has no alternative positive lattice length")
        old[position] = choices[int(torch.randint(len(choices), (), generator=generator))]
    scope = {
        "prompt": row["body_prompt"],
        "old": old.copy(),
        "body": old.copy(),
        "n": n,
        "active": [],
        "remaining": available_calls,
        "available_calls": available_calls,
        "reveal": 0.0,
        "temperature": 1.0,
    }
    content = None
    cut = None
    if positions:
        cut = int(torch.randint(len(positions), (), generator=generator))
        current = old.copy()
        for position in positions[:cut]:
            current[position] = target[position]
        for position in positions[cut:]:
            current[position] = mask_id
        content = {
            "prompt": row["body_prompt"],
            "old": old.copy(),
            "body": current,
            "n": n,
            "active": positions.copy(),
            "remaining": available_calls - 1 - cut,
            "available_calls": available_calls,
            "reveal": cut / len(positions),
            "position": positions[cut],
            "temperature": 1.0,
        }
    return {
        "source_id": row["source_id"],
        "target": target,
        "mode": mode,
        "count_index": count_index,
        "sites": sites,
        "positions": positions,
        "scope": scope,
        "content": content,
        "labels_are": "known_corruption_not_physical_optimality",
    }


def edit_warm_loss(
    model, tokenizer, schema, row, generator, *, mask_id, available_calls, forced_mode=None, scope_weight=1.0
):
    example = warm_example(
        row, schema, generator, mask_id=mask_id, available_calls=available_calls, forced_mode=forced_mode
    )
    out, _ = forward_e(model, tokenizer, example["scope"])
    content, prefix = (
        forward_e(model, tokenizer, example["content"]) if example["content"] is not None else (None, None)
    )
    return warm_loss_from_outputs(example, schema, out, content, prefix, scope_weight=scope_weight)


def warm_loss_from_outputs(example, schema, out, content, prefix, *, scope_weight=1.0):
    observation = example["scope"]
    available_calls = observation["available_calls"]
    modes = feasible_modes(observation["n"], available_calls)
    scope_loss = -normalized_choices(out.mode_logits[0], modes)[modes.index(example["mode"])]
    if example["mode"] == 1:
        counts = feasible_counts(observation["n"], available_calls)
        scope_loss = (
            scope_loss - normalized_choices(out.count_logits[0], counts)[counts.index(example["count_index"])]
        )
        scope_loss = scope_loss - ordered_site_logprob(
            out.site_logits[0, : observation["n"]], example["sites"]
        )
    content_loss = scope_loss * 0.0
    if content is not None:
        position = example["content"]["position"]
        actions, lp = schema.log_probabilities(content.logits[0, prefix + position], position)
        content_loss = -lp[actions.index(example["target"][position])]
    return content_loss + scope_weight * scope_loss, {
        "source_id": example["source_id"],
        "mode": example["mode"],
        "labels_are": example["labels_are"],
        "scope_loss": float(scope_loss.detach()),
        "content_loss": float(content_loss.detach()),
    }
