"""Portable device setup. Processes use only explicitly selected devices."""

import os


def setup_device(device="cuda:0", *, threads=1, deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch

    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    result = torch.device(device)
    if result.type == "cuda":
        torch.cuda.set_device(result)
    return result
