"""Retained crystal DLM implementation; see docs/method.md for the public workflow."""

from __future__ import annotations
from typing import Any


def ensure_llada2_rope_parameters(config: Any) -> Any:
    """Populate the LLaDA2 ``rope_parameters`` field expected by remote code."""

    if hasattr(config, "rope_parameters"):
        return config

    rope_scaling = getattr(config, "rope_scaling", None)
    rope_type = "default"
    params: dict[str, Any] = {}
    if isinstance(rope_scaling, dict):
        params.update(rope_scaling)
        rope_type = (
            rope_scaling.get("rope_type")
            or rope_scaling.get("type")
            or rope_scaling.get("rope_type_name")
            or "default"
        )
    params.setdefault("rope_type", rope_type)
    params.setdefault("rope_theta", getattr(config, "rope_theta", 10000.0))
    params.setdefault("partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0))
    config.rope_parameters = params
    return config


def ensure_create_bidirectional_mask() -> bool:
    """Install a small fallback for newer remote-code model imports.

    Some LLaDA2 checkpoints import ``create_bidirectional_mask`` from
    ``transformers.masking_utils``.  The A800 environment currently has a
    Transformers build where that symbol is absent, while the model code only
    needs a full bidirectional padding mask for our fixed-length SFT/sampling
    batches.  Returning ``None`` for fully unmasked batches matches the common
    Transformers convention and keeps the model path unchanged.
    """

    try:
        import torch
        import torch.nn.functional as F
        import transformers.masking_utils as masking_utils
    except Exception:
        return False

    if hasattr(masking_utils, "create_bidirectional_mask"):
        return False

    def create_bidirectional_mask(
        config: Any,
        inputs_embeds,
        attention_mask=None,
        encoder_hidden_states=None,
        past_key_values=None,
        **_: Any,
    ):
        del config
        if attention_mask is not None and getattr(attention_mask, "ndim", None) == 4:
            return attention_mask

        embeds = encoder_hidden_states if encoder_hidden_states is not None else inputs_embeds
        batch_size = int(inputs_embeds.shape[0])
        query_length = int(inputs_embeds.shape[1])
        key_value_length = int(embeds.shape[1])
        if past_key_values is not None:
            try:
                key_value_length = max(
                    key_value_length,
                    int(past_key_values.get_seq_length()) + query_length,
                )
            except Exception:
                pass

        if attention_mask is None:
            return None
        if getattr(attention_mask, "ndim", None) != 2:
            return attention_mask

        padding_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)
        if padding_mask.shape[-1] < key_value_length:
            padding_mask = F.pad(
                padding_mask,
                (0, key_value_length - padding_mask.shape[-1]),
                value=False,
            )
        elif padding_mask.shape[-1] > key_value_length:
            padding_mask = padding_mask[:, :key_value_length]

        if bool(padding_mask.all()):
            return None

        dtype = inputs_embeds.dtype if inputs_embeds.dtype.is_floating_point else torch.float32
        min_value = torch.finfo(dtype).min
        zero = torch.tensor(0.0, device=inputs_embeds.device, dtype=dtype)
        masked = torch.tensor(min_value, device=inputs_embeds.device, dtype=dtype)
        mask = torch.where(padding_mask[:, None, None, :], zero, masked)
        return mask.expand(batch_size, 1, query_length, key_value_length)

    masking_utils.create_bidirectional_mask = create_bidirectional_mask
    return True
