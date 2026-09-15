"""Retained crystal DLM implementation; see docs/method.md for the public workflow."""

import random
import torch
from dlm_iclr._core.fixed_slot import MASK_TOKEN_ID
from dlm_iclr._core.rsi_preference import legal_vector, numeric_order
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


def training_view(example, target, cut, branch, *, mask_seed=0):
    n = example["num_sites"]
    if branch == "G" and example.get("plan_state"):
        groups = example["generation_groups"]
        expected = set(numeric_order(n, "G"))
        positions = [pos for group in groups for pos in group]
        if len(positions) != len(expected) or set(positions) != expected:
            raise ValueError("registered native groups changed numeric support")
        rng = random.Random(mask_seed)
        # Native semantic groups stay in order; cover possible reveal subsets
        # within each confidence-remasked group instead of one site ordering.
        order = []
        for positions in groups:
            positions = list(positions)
            rng.shuffle(positions)
            order.extend(positions)
    else:
        order = (
            example.get("action_positions", numeric_order(n, branch))
            if branch == "E"
            else numeric_order(n, branch)
        )
    if not 0 <= cut < len(order):
        raise ValueError("invalid mask cut")
    if branch == "E":
        from dlm_iclr._core.ranked_feedback import validate_action_target

        validate_action_target(example["current_tokens"], target, order)
    current = list(target)
    for pos in order[cut:]:
        current[pos] = MASK_TOKEN_ID
    return {
        "example": example,
        "target": target,
        "current": current,
        "position": order[cut],
        "order": order,
        "reveal": cut / len(order),
    }


def conditional_batch(model, tokenizer, views, branch, support):
    """One transformer forward for multiple independent conditional views."""
    device = next(model.parameters()).device
    prefixes = [tokenizer(v["example"]["prompt"], add_special_tokens=False)["input_ids"] for v in views]
    if branch == "G":
        width = max(len(p) + len(v["current"]) for p, v in zip(prefixes, views))
        ids = torch.full((len(views), width), int(tokenizer.pad_token_id), device=device, dtype=torch.long)
        attention = torch.zeros_like(ids)
        for i, (prefix, view) in enumerate(zip(prefixes, views)):
            body = prefix + view["current"]
            ids[i, : len(body)] = torch.tensor(body, device=device)
            attention[i, : len(body)] = 1
        output = model(ids, attention_mask=attention)
    else:
        examples = [
            inference_view(
                prefix,
                v["example"]["current_tokens"],
                v["current"],
                v["example"]["num_sites"],
                1,
                v["order"],
                remaining=80,
                reveal=v["reveal"],
            )
            for prefix, v in zip(prefixes, views)
        ]
        batch = materialize_edit_batch(examples, tokenizer, device)
        output = model(
            batch["input_ids"], attention_mask=batch["attention_mask"], edit_context=batch["edit_context"]
        )
    values, distributions = [], []
    for i, (prefix, view) in enumerate(zip(prefixes, views)):
        pos = view["position"]
        target = int(view["target"][pos])
        vector, report = legal_vector(
            output.logits[i, len(prefix) + pos].float(),
            view["current"],
            view["example"]["num_sites"],
            pos,
            support,
        )
        minimum = torch.finfo(vector.dtype).min
        if not report["available"] or vector[target] <= minimum:
            raise ValueError("minibatch target absent from unchanged hard support")
        legal = (vector.detach() > minimum).nonzero().flatten()
        logp = (vector[legal] / 0.7).log_softmax(-1)
        values.append(logp[(legal == target).nonzero().item()])
        distributions.append(logp)
    return torch.stack(values), distributions


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
