"""Batched refinement over saved endpoints with CPU data-loader workers."""

from pathlib import Path
import time
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from dlm_iclr.runtime.io import read_json, write_json


class SavedGraphs(Dataset):
    def __init__(self, root, plans):
        self.root, self.plans = Path(root), plans

    def __len__(self):
        return len(self.plans)

    def __getitem__(self, index):
        generated = read_json(self.root / "raw" / f"{index:06d}.json")
        generated["graph"] = (torch.load(self.root / "graphs" / f"{index:06d}.pt",
                                         map_location="cpu", weights_only=False)
                              if generated["graph_available"] else None)
        return index, self.plans[index], generated


class PendingBatches(Sampler):
    """Keep original membership on resume, even for partly written batches."""
    def __init__(self, indices, size, root):
        self.batches = [indices[i:i + size] for i in range(0, len(indices), size)]
        self.batches = [group for group in self.batches
                        if any(not (Path(root) / "refined" / f"{i:06d}.json").exists() for i in group)]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def collate_saved_graphs(rows):
    return rows


def refine_saved_batches(refiner, plans, root, rank, world, *, batch_size, loader_workers):
    root = Path(root)
    indices = list(range(rank, len(plans), world))
    pending = PendingBatches(indices, batch_size, root)
    loader = DataLoader(SavedGraphs(root, plans), batch_sampler=pending,
                        num_workers=loader_workers, collate_fn=collate_saved_graphs,
                        **({"multiprocessing_context": "spawn", "prefetch_factor": 1}
                           if loader_workers else {}))
    completed = sum((root / "refined" / f"{i:06d}.json").exists() for i in indices)
    for rows in loader:
        started = time.monotonic()
        if refiner.model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(refiner.model.device)
        outputs = refiner.refine_many([row[1] for row in rows], [row[2] for row in rows])
        for (index, _, _), value in zip(rows, outputs, strict=True):
            path = root / "refined" / f"{index:06d}.json"
            if not path.exists():
                write_json(path, value)
                completed += 1
        statistics = {"stage": "diffusion", "worker": rank, "completed": completed,
                      "assigned": len(indices), "batch_requests": len(rows),
                      "model_batch_size": sum(row[2]["graph"] is not None for row in rows),
                      "steps": refiner.steps, "loader_workers": loader_workers,
                      "batch_seconds": time.monotonic() - started,
                      "peak_allocated_GiB": (torch.cuda.max_memory_allocated(refiner.model.device) / 2**30
                                             if refiner.model.device.type == "cuda" else None)}
        write_json(root / f"diffusion_worker{rank}.json", statistics)
        print(statistics, flush=True)
