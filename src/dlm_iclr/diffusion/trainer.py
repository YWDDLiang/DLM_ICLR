"""CrysLLMGen diffusion training with portable datasets and checkpoint paths.

Adapted from kdmsit/crysllmgen diff_train.py (MIT); see THIRD_PARTY_NOTICES.md.
"""

from functools import partial
from pathlib import Path
import torch
from ..runtime.config import run_root
from ..runtime.io import read_rows, write_json
from ..runtime.device import setup_device
from ..runtime.checkpoint import seed_all, rng_state, restore_rng, save


class CrystalGraphs(torch.utils.data.Dataset):
    def __init__(self, source, cache):
        from .._vendor.crysllmgen.data_utils import process_one
        from pymatgen.core import Structure

        cache = Path(cache)
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            self.graphs = torch.load(cache, map_location="cpu", weights_only=False)
        else:
            self.graphs = []
            for row in read_rows(source):
                cif = Structure.from_dict(row["structure"]).to(fmt="cif")
                *_, graph = process_one(cif, True, False, "crystalnn", False, 0.01)
                self.graphs.append(graph)
            torch.save(self.graphs, cache)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        from .refinement import ProposalDataset

        return ProposalDataset(dict(self.graphs[index], refiner_noise_seed=0))[0]


def train(config, *, device="cuda:0", resume=False):
    try:
        from torch_geometric.loader import DataLoader
    except ImportError:
        from torch_geometric.data import DataLoader
    from .._vendor.crysllmgen.models_ddpm.diffusion import CSPDiffusion

    root, recipe = run_root(config), config["diffusion"]["training"]
    device = setup_device(device, threads=config["runtime"]["threads"])
    seed_all(recipe["seed"])
    train_set = CrystalGraphs(root / "data/structures/train.jsonl", root / "data/graphs/train.pt")
    val_set = CrystalGraphs(root / "data/structures/val.jsonl", root / "data/graphs/val.pt")
    loader = DataLoader(
        train_set, batch_size=recipe["batch_size"], shuffle=True, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(val_set, batch_size=recipe["validation_batch_size"], shuffle=True)
    model = CSPDiffusion(recipe["timesteps"], "train").to(device)
    model.device = device
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=recipe["scheduler_factor"],
        patience=recipe["scheduler_patience"],
        threshold=0.0001,
    )
    output = root / "diffusion"
    start = 0
    best = float("inf")
    if resume and (output / "last.pt").exists():
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, best = saved["epoch"], saved["best_loss"]
        restore_rng(saved["rng"])
    history = []
    for epoch in range(start, recipe["epochs"]):
        model.train()
        total = 0.0
        for batch in loader:
            loss, _, _ = model(batch.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), recipe["clip_value"])
            optimizer.step()
            total += float(loss.detach())
        training_loss = total / len(loader)
        model.eval()
        validation = 0.0
        with torch.no_grad():
            for batch in val_loader:
                validation += float(model(batch.to(device))[0])
        validation /= len(val_loader)
        scheduler.step(training_loss)
        improved = training_loss < best
        best = min(best, training_loss)
        # CrysLLMGen selects best by training loss and exports the last model as final.
        save(
            output,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": rng_state(),
                "epoch": epoch + 1,
                "best_loss": best,
                "settings": recipe,
            },
            best=improved,
        )
        report = {"epoch": epoch + 1, "training_loss": training_loss, "validation_loss": validation}
        history.append(report)
        write_json(output / "progress.json", report)
        print(report, flush=True)
    write_json(output / "training.json", {"status": "complete", "settings": recipe, "history": history})
    return {"checkpoint": str(output / "last.pt")}
