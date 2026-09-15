"""Retained crystal DLM implementation; see docs/method.md for the public workflow."""

import torch
from dlm_iclr._core.expert_edit import inference_view, materialize_edit_batch
from dlm_iclr._core.fixed_slot import MASK_TOKEN_ID
from dlm_iclr._core.ranked_feedback import validate_action_target


def dense_view(row, prefix, *, masked):
    target = row["content_target_tokens"]
    order = row["content_positions"]
    validate_action_target(row["current_tokens"], target, order)
    current = list(row["current_tokens"])
    cut = row.get("training_cut")
    if cut is not None:
        if not 0 <= cut < len(order):
            raise ValueError("training cut must leave at least one masked target")
        for index, pos in enumerate(order):
            current[pos] = target[pos] if index < cut else MASK_TOKEN_ID
    if masked:
        for pos in order if cut is None else []:
            current[pos] = MASK_TOKEN_ID
    view = inference_view(
        prefix,
        row["current_tokens"],
        current,
        row["num_sites"],
        1,
        order,
        remaining=80,
        reveal=0.0 if cut is None else cut / len(order),
    )
    for pos in order if cut is None else order[cut:]:
        view["targets"][pos] = target[pos]
    return view


def dense_vectors(model, tokenizer, rows, objective, *, masked):
    device = next(model.parameters()).device
    views = [
        dense_view(r, tokenizer(r["prompt"], add_special_tokens=False)["input_ids"], masked=masked)
        for r in rows
    ]
    batch = materialize_edit_batch(views, tokenizer, device)
    output = model(
        batch["input_ids"], attention_mask=batch["attention_mask"], edit_context=batch["edit_context"]
    )
    rr, pp = torch.nonzero(batch["targets"] != -100, as_tuple=True)
    relative = pp - batch["edit_context"].prompt_lengths[rr]
    vectors = []
    for kind in range(9):
        if kind < 6:
            selected = relative == kind + 1
            family, axis = ("length", "ABC"[kind]) if kind < 3 else ("angle", "ABG"[kind - 3])
        else:
            selected = (relative >= 8) & ((relative - 8).remainder(4) == kind - 6)
            family, axis = "coord", "XYZ"[kind - 6]
        r, p = rr[selected], pp[selected]
        if not len(r):
            continue
        logits, ids = objective.typed_vector(output.logits, r, p, family, axis)
        target = batch["targets"][r, p]
        if family == "coord":
            original_ids, values = objective.tables[(family, axis)]
            alias, zero = original_ids[values == 1].item(), original_ids[values == 0].item()
            target = torch.where(target == alias, zero, target)
        matches = target[:, None] == ids[None]
        if not bool(matches.any(-1).all()):
            raise ValueError("dense target is outside canonical typed periodic vocabulary")
        vectors.append((r, logits.log_softmax(-1), matches.long().argmax(-1)))
    if sum(len(r) for r, _, _ in vectors) != len(rr):
        raise ValueError("dense supervision omitted or duplicated a target token")
    return vectors


def dense_loss(live, reference, rows):
    device = live[0][1].device
    sums = torch.zeros(len(rows), device=device)
    counts = torch.zeros_like(sums)
    kl_sums = torch.zeros_like(sums)
    for (r, p, target), (qr, q, qtarget) in zip(live, reference, strict=True):
        if not torch.equal(r, qr) or not torch.equal(target, qtarget):
            raise ValueError("reference dense supervision changed")
        ce = -p.gather(1, target[:, None]).squeeze(1)
        kl = (q.exp() * (q - p)).sum(-1)
        sums = sums.scatter_add(0, r, ce)
        kl_sums = kl_sums.scatter_add(0, r, kl)
        counts = counts.scatter_add(0, r, torch.ones_like(ce))
    if bool((counts == 0).any()):
        raise ValueError("empty dense target")
    return (sums / counts).sum(), (kl_sums / counts).sum(), int(counts.sum())
