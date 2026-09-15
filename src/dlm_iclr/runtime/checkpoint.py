"""Best/last checkpoint retention shared by the trainable modules."""

import os
import random
from pathlib import Path
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    state = np.random.get_state()
    numpy_state = [state[0], state[1].tolist(), state[2], state[3], state[4]]
    return {
        "python": random.getstate(),
        "numpy": numpy_state,
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save(output, state, *, best=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / ".checkpoint.tmp"
    torch.save(state, temporary)
    temporary.replace(output / "last.pt")
    if best:
        target = output / "best.pt"
        target.unlink(missing_ok=True)
        try:
            os.link(output / "last.pt", target)
        except OSError:
            import shutil

            shutil.copy2(output / "last.pt", target)
