"""Retained crystal DLM implementation; see docs/reproduction.md for the public workflow."""

import random
import torch
from dlm_iclr._core.expert_edit import inference_view, materialize_edit_batch


def epoch_indices(size, *, batch_size, world, epochs, seed):
    """Every record once per epoch, with at most one padding repeat per record."""
    if min(size, batch_size, world, epochs) < 1:
        raise ValueError("positive data and batch dimensions required")
    rng = random.Random(seed)
    width = batch_size * world
    for epoch in range(epochs):
        indices = list(range(size))
        rng.shuffle(indices)
        # Pad only to world size, not to a full large minibatch. The final
        # minibatch is smaller; increasing capacity must not multiply exposure.
        padding = (-len(indices)) % world
        # Small datasets may need more than one padding copy; expose this in receipts.
        extra = []
        while len(extra) < padding:
            copy = list(range(size))
            rng.shuffle(copy)
            extra.extend(copy)
        indices.extend(extra[:padding])
        for start in range(0, len(indices), width):
            yield epoch, indices[start : start + width]


def editor_head_loss(model, tokenizer, examples, device, *, detach_content=False, site_objective="binary"):
    from dlm_iclr._core.ranked_feedback import COUNTS

    chosen = [x for x in examples if has_head_supervision(x)]
    if not chosen:
        return None, 0
    views, modes, judges = [], [], []
    for row in chosen:
        prefix = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        if row.get("mode_target") is not None:
            modes.append(len(views))
            views.append(
                inference_view(prefix, row["current_tokens"], row["current_tokens"], row["num_sites"], 1)
            )
        else:
            modes.append(None)
        if row.get("proposal_tokens") is not None and row.get("accept_target") is not None:
            judges.append(len(views))
            views.append(
                inference_view(
                    prefix,
                    row["current_tokens"],
                    row["proposal_tokens"],
                    row["num_sites"],
                    1,
                    row.get("action_positions", []),
                    remaining=80,
                    reveal=1.0,
                )
            )
        else:
            judges.append(None)
    batch = materialize_edit_batch(views, tokenizer, device)
    kwargs = {"detach_head_features": True} if detach_content else {}
    out = model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        edit_context=batch["edit_context"],
        **kwargs,
    )
    losses = []
    for row, i, j in zip(chosen, modes, judges):
        weight = 2.0 if row.get("known_sun") else 1.0
        loss = out.quality_logits.new_zeros(())
        if i is not None:
            loss = weight * torch.nn.functional.cross_entropy(
                out.mode_logits[i : i + 1], torch.tensor([row["mode_target"]], device=device)
            )
        if row.get("mode_target") == 1:
            sites = torch.tensor(row["site_targets"], device=device, dtype=torch.float32)
            site_logits = out.site_logits[i, : row["num_sites"]]
            if site_objective == "categorical":
                loss = loss - (site_logits.log_softmax(-1) * sites).sum() / sites.sum().clamp_min(1)
            else:
                loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(site_logits, sites)
            loss = loss + torch.nn.functional.cross_entropy(
                out.count_logits[i : i + 1], torch.tensor([COUNTS.index(int(sites.sum()))], device=device)
            )
        if j is not None:
            loss = loss + weight * torch.nn.functional.binary_cross_entropy_with_logits(
                out.quality_logits[j, 3], torch.tensor(float(row["accept_target"]), device=device)
            )
        losses.append(loss)
    return torch.stack(losses).sum() / len(examples), len(chosen)


def has_head_supervision(row):
    return row.get("mode_target") is not None or (
        row.get("proposal_tokens") is not None and row.get("accept_target") is not None
    )


def decision_head_parameter(name):
    return name.split(".", 1)[0] in ("mode_head", "site_head", "count_head", "quality_head")
