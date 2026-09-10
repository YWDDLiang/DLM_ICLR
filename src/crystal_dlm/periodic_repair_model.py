"""Retained crystal DLM implementation; see docs/method.md for the public workflow."""

from __future__ import annotations
import torch
from torch import nn


class FP32Module(nn.Module):
    def _apply(self, fn, recurse=True):
        def keep(tensor):
            result = fn(tensor)
            if result.is_floating_point() and result.dtype != torch.float32:
                return tensor.to(device=result.device, dtype=torch.float32)
            return result

        return super()._apply(keep, recurse=recurse)
