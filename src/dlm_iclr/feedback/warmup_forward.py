"""Shared geometric model view for scope and conditional fill supervision."""


def forward_e(model, tokenizer, observation):
    from .._core.expert_edit import inference_view, materialize_edit_batch

    device = next(model.parameters()).device
    prefix = tokenizer(observation["prompt"], add_special_tokens=False)["input_ids"]
    view = inference_view(
        prefix,
        observation["old"],
        observation["body"],
        observation["n"],
        1,
        observation.get("active", []),
        remaining=observation["remaining"],
        reveal=observation.get("reveal", 0.0),
    )
    batch = materialize_edit_batch([view], tokenizer, device)
    out = model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        edit_context=batch["edit_context"],
        detach_head_features=False,
    )
    return out, len(prefix)


def forward_many(model, tokenizer, observations):
    from .._core.expert_edit import inference_view, materialize_edit_batch

    prefixes = [tokenizer(o["prompt"], add_special_tokens=False)["input_ids"] for o in observations]
    views = [
        inference_view(
            p,
            o["old"],
            o["body"],
            o["n"],
            1,
            o.get("active", []),
            remaining=o["remaining"],
            reveal=o.get("reveal", 0.0),
        )
        for p, o in zip(prefixes, observations)
    ]
    batch = materialize_edit_batch(views, tokenizer, next(model.parameters()).device)
    out = model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        edit_context=batch["edit_context"],
        detach_head_features=False,
    )
    return out, [len(p) for p in prefixes]
