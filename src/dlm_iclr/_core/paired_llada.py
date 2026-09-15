"""LLaDA exact-plan sampling with method-independent per-attempt noise."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from dlm_iclr._core.llada_generation import (
    _apply_lightweight_decoding_masks,
    _apply_schema_masks,
    _model_logits,
    _prepare_atom_count_grammar,
    _validate_generation_position_groups,
    get_num_transfer_tokens,
)
from dlm_iclr._core.paired_noise import paired_uniform


def _paired_suffix_candidates(
    logits: torch.Tensor,
    *,
    current_tokens: torch.Tensor,
    prompt_length: int,
    gen_length: int,
    temperature: float,
    remasking: str,
    base_seeds: list[int],
    semantic_group: int,
    step_in_group: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(base_seeds) != logits.shape[0]:
        raise ValueError("one body-noise seed is required per batch row")
    suffix = logits[:, prompt_length : prompt_length + gen_length, :]
    if temperature == 0:
        suffix_scores = suffix
    else:
        rows = []
        for row, seed in zip(suffix, base_seeds, strict=True):
            uniform = paired_uniform(
                seed,
                stage=f"body_gumbel_suffix_group_{semantic_group}",
                step=step_in_group,
                shape=row.shape,
                device=row.device,
                dtype=torch.float64,
            )
            rows.append(row.to(torch.float64).exp() / ((-torch.log(uniform)) ** temperature))
        suffix_scores = torch.stack(rows)
    suffix_x0 = torch.argmax(suffix_scores, dim=-1)
    x0 = current_tokens.clone()
    x0[:, prompt_length : prompt_length + gen_length] = suffix_x0
    confidence = torch.full(
        current_tokens.shape,
        -float("inf"),
        dtype=logits.dtype,
        device=logits.device,
    )
    if remasking == "low_confidence":
        probabilities = F.softmax(suffix, dim=-1)
        suffix_confidence = torch.gather(probabilities, dim=-1, index=suffix_x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        random_rows = []
        for row, seed in zip(suffix_x0, base_seeds, strict=True):
            random_rows.append(
                paired_uniform(
                    seed,
                    stage=f"body_random_remask_suffix_group_{semantic_group}",
                    step=step_in_group,
                    shape=row.shape,
                    device=row.device,
                    dtype=logits.dtype,
                )
            )
        suffix_confidence = torch.stack(random_rows)
    else:
        raise NotImplementedError(remasking)
    confidence[:, prompt_length : prompt_length + gen_length] = suffix_confidence
    return x0, confidence


@torch.no_grad()
def generate_paired_exact_plan(
    model: Any,
    prompt: torch.Tensor,
    *,
    base_seeds: list[int],
    attention_mask: torch.Tensor | None,
    gen_length: int,
    temperature: float,
    cfg_scale: float,
    remasking: str,
    mask_id: int,
    allowed_token_ids_by_generation_pos: list[list[int]] | None,
    prefill_token_ids_by_generation_pos: dict[int, int | list[int] | torch.Tensor] | None,
    generation_position_groups: list[list[int]],
    lightweight_decoding_constraints: dict | None,
    candidate_hook=None,
    predictor=None,
    candidate_sampler=None,
) -> torch.Tensor:
    if len(base_seeds) != prompt.shape[0]:
        raise ValueError("base_seeds must match the body batch size")
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    if prefill_token_ids_by_generation_pos:
        for generation_pos, token_id in prefill_token_ids_by_generation_pos.items():
            if not 0 <= int(generation_pos) < gen_length:
                continue
            if isinstance(token_id, torch.Tensor):
                values = token_id.to(device=model.device, dtype=torch.long)
            elif isinstance(token_id, list):
                values = torch.tensor(token_id, dtype=torch.long, device=model.device)
            else:
                values = torch.full(
                    (prompt.shape[0],),
                    int(token_id),
                    dtype=torch.long,
                    device=model.device,
                )
            if values.numel() != prompt.shape[0]:
                raise ValueError("prefill batch size changed")
            x[:, prompt.shape[1] + int(generation_pos)] = values.view(-1)
    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (prompt.shape[0], gen_length),
                    dtype=attention_mask.dtype,
                    device=model.device,
                ),
            ],
            dim=-1,
        )
    prompt_index = x != mask_id
    allowed_mask = None
    prepared_atom_count_grammar = None
    if allowed_token_ids_by_generation_pos is not None:
        if len(allowed_token_ids_by_generation_pos) != gen_length:
            raise ValueError("schema mask length changed")
        vocab_size = model.get_output_embeddings().weight.shape[0]
        allowed_mask = torch.zeros((gen_length, vocab_size), dtype=torch.bool, device=model.device)
        for position, token_ids in enumerate(allowed_token_ids_by_generation_pos):
            if not token_ids:
                raise ValueError(f"schema position {position} has no allowed tokens")
            allowed_mask[
                position,
                torch.tensor(token_ids, dtype=torch.long, device=model.device),
            ] = True
        prepared_atom_count_grammar = _prepare_atom_count_grammar(None, vocab_size, model.device)
    for group_index, group in enumerate(
        _validate_generation_position_groups(generation_position_groups, gen_length)
    ):
        absolute = torch.tensor(
            [prompt.shape[1] + position for position in group],
            dtype=torch.long,
            device=x.device,
        )
        group_allowed = torch.zeros_like(x, dtype=torch.bool)
        group_allowed[:, absolute] = True
        group_mask = (x == mask_id) & group_allowed
        group_steps = int(group_mask.sum(dim=1).max().detach().item())
        if group_steps <= 0:
            continue
        transfers = get_num_transfer_tokens(group_mask, group_steps)
        for step_in_group in range(group_steps):
            mask_index = x == mask_id
            if predictor is None:
                logits = _model_logits(model, x, attention_mask, prompt_index, cfg_scale, mask_id)
                hidden = None
            else:
                logits, hidden = predictor(
                    model, x, attention_mask, prompt_index, cfg_scale, mask_id, group_positions=group
                )
            if allowed_mask is not None:
                _apply_schema_masks(
                    logits,
                    x,
                    prompt.shape[1],
                    gen_length,
                    allowed_mask,
                    prepared_atom_count_grammar,
                )
            _apply_lightweight_decoding_masks(
                logits,
                x,
                prompt.shape[1],
                gen_length,
                lightweight_decoding_constraints,
            )
            if candidate_hook is not None:
                candidate_hook(
                    logits,
                    current_tokens=x,
                    prompt_length=prompt.shape[1],
                    gen_length=gen_length,
                    semantic_group=group_index,
                    step_in_group=step_in_group,
                )
            candidate_arguments = dict(
                current_tokens=x,
                prompt_length=prompt.shape[1],
                gen_length=gen_length,
                temperature=temperature,
                remasking=remasking,
                base_seeds=base_seeds,
                semantic_group=group_index,
                step_in_group=step_in_group,
            )
            if candidate_sampler is None:
                x0, confidence = _paired_suffix_candidates(logits, **candidate_arguments)
            else:
                x0, confidence = candidate_sampler(
                    logits, hidden=hidden, group_positions=group, mask_id=mask_id, **candidate_arguments
                )
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index & group_allowed, confidence, -float("inf"))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for row_index in range(confidence.shape[0]):
                count = int(transfers[row_index, step_in_group].detach().item())
                if count <= 0:
                    continue
                _, selected = torch.topk(confidence[row_index], k=count)
                transfer_index[row_index, selected] = True
            if candidate_sampler is not None and hasattr(candidate_sampler, "record_commit"):
                candidate_sampler.record_commit(
                    current_tokens=x,
                    candidates=x0,
                    transfer_index=transfer_index,
                    prompt_length=prompt.shape[1],
                    semantic_group=group_index,
                    step_in_group=step_in_group,
                )
            x[transfer_index] = x0[transfer_index]
    if bool((x[:, prompt.shape[1] :] == mask_id).any()):
        raise RuntimeError("paired exact-plan generation left masked suffix tokens")
    return x


__all__ = ["generate_paired_exact_plan"]
