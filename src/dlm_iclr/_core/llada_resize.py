"""Retained crystal DLM implementation; see docs/method.md for the public workflow."""

from __future__ import annotations
from typing import Any, Dict
import torch
from torch import nn


def _mean_init_rows(weight: torch.Tensor, old_size: int) -> None:
    if weight.shape[0] <= old_size or old_size <= 0:
        return
    weight[old_size:] = weight[:old_size].mean(dim=0, keepdim=True)


def _resize_embedding(old: nn.Embedding, new_size: int) -> nn.Embedding:
    new = nn.Embedding(
        new_size,
        old.embedding_dim,
        padding_idx=old.padding_idx,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    copy_size = min(old.num_embeddings, new_size)
    with torch.no_grad():
        new.weight[:copy_size] = old.weight[:copy_size]
        _mean_init_rows(new.weight, copy_size)
    return new


def _resize_linear_output(old: nn.Linear, new_size: int) -> nn.Linear:
    new = nn.Linear(
        old.in_features,
        new_size,
        bias=old.bias is not None,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    copy_size = min(old.out_features, new_size)
    with torch.no_grad():
        new.weight[:copy_size] = old.weight[:copy_size]
        _mean_init_rows(new.weight, copy_size)
        if old.bias is not None and new.bias is not None:
            new.bias[:copy_size] = old.bias[:copy_size]
            if new.bias.shape[0] > copy_size and copy_size > 0:
                new.bias[copy_size:] = old.bias[:copy_size].mean()
    return new


def _update_config_vocab(model: Any, new_size: int) -> None:
    for obj in (model, getattr(model, "model", None)):
        config = getattr(obj, "config", None)
        if config is None:
            continue
        if hasattr(config, "vocab_size"):
            config.vocab_size = new_size
        if getattr(config, "embedding_size", None) is not None:
            config.embedding_size = new_size


def ensure_llada_vocab_size(model: Any, new_size: int) -> Dict[str, Any]:
    """Ensure both input embeddings and output logits cover tokenizer ids."""

    info: Dict[str, Any] = {
        "requested_vocab_size": new_size,
        "input_resized": False,
        "output_resized": False,
    }

    input_embeddings = model.get_input_embeddings()
    info["input_vocab_size_before"] = getattr(input_embeddings, "num_embeddings", None)
    if isinstance(input_embeddings, nn.Embedding) and input_embeddings.num_embeddings != new_size:
        model.set_input_embeddings(_resize_embedding(input_embeddings, new_size))
        info["input_resized"] = True

    output_embeddings = model.get_output_embeddings()
    if isinstance(output_embeddings, nn.Embedding):
        info["output_vocab_size_before"] = output_embeddings.num_embeddings
        if output_embeddings.num_embeddings != new_size:
            model.set_output_embeddings(_resize_embedding(output_embeddings, new_size))
            info["output_resized"] = True
    elif isinstance(output_embeddings, nn.Linear):
        info["output_vocab_size_before"] = output_embeddings.out_features
        if output_embeddings.out_features != new_size:
            model.set_output_embeddings(_resize_linear_output(output_embeddings, new_size))
            info["output_resized"] = True
    else:
        info["output_vocab_size_before"] = None

    _update_config_vocab(model, new_size)
    output_after = model.get_output_embeddings()
    info["input_vocab_size_after"] = getattr(model.get_input_embeddings(), "num_embeddings", None)
    info["output_vocab_size_after"] = (
        getattr(output_after, "num_embeddings", None)
        if isinstance(output_after, nn.Embedding)
        else getattr(output_after, "out_features", None)
    )
    return info
