"""The retained dynamic-v1, response-only masked likelihood objective."""

import torch
import torch.nn.functional as F
from .._core.fixed_slot import MASK_TOKEN_ID


def corrupt(input_ids, attention_mask, prompt_lengths, *, eps=1e-3):
    batch, length = input_ids.shape
    positions = torch.arange(length, device=input_ids.device).expand(batch, length)
    answer = (positions >= prompt_lengths[:, None]) & attention_mask.bool()
    time = torch.rand(batch, device=input_ids.device)
    probability = ((1 - eps) * time + eps)[:, None].expand(batch, length)
    masked = (torch.rand((batch, length), device=input_ids.device) < probability) & answer
    for row in range(batch):
        if answer[row].any() and not masked[row].any():
            available = answer[row].nonzero().flatten()
            masked[row, available[torch.randint(len(available), (1,), device=input_ids.device)]] = True
    return torch.where(masked, MASK_TOKEN_ID, input_ids), masked, probability, answer


def denoising_loss(model, batch):
    noisy, masked, probability, answer = corrupt(
        batch["input_ids"], batch["attention_mask"], batch["prompt_lengths"]
    )
    logits = model(input_ids=noisy, attention_mask=batch["attention_mask"]).logits
    loss = F.cross_entropy(logits[masked], batch["input_ids"][masked], reduction="none")
    norm = answer.sum(1).clamp_min(1)[:, None].expand_as(noisy)
    loss = loss / probability[masked] / norm[masked]
    rows = torch.arange(len(noisy), device=noisy.device)[:, None].expand_as(noisy)[masked]
    per_row = torch.zeros(len(noisy), device=noisy.device, dtype=loss.dtype).scatter_add_(0, rows, loss)
    return per_row.mean()
