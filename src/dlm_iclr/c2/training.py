"""Fit geometric edit proposals with the retained content and decision objectives."""

from pathlib import Path
import random
import torch
from ..runtime.io import read_rows, write_json
from ..runtime.models import load_editor
from ..runtime.checkpoint import rng_state, restore_rng, save
from .._core.r03_physics_transfer import build_repair_constraints
from .._core.rsi_minibatch import decision_head_parameter
from .._core.editor_minibatch import train_editor_minibatches


def train_actor(
    branch, config, data, output, *, checkpoint=None, device="cuda:0", parameter_scope="all", resume=False
):
    if branch != "E":
        raise ValueError("Use the C1 trainer for geometric generation")
    rows, output = read_rows(data), Path(output)
    if not rows or any(row.get("source_split") != "train" for row in rows):
        raise ValueError("Editor training requires training examples")
    output.mkdir(parents=True, exist_ok=True)
    training = config.training
    torch.manual_seed(training.seed)
    random.seed(training.seed)
    model, tokenizer = load_editor(
        config.assets.base_model, checkpoint or config.assets.editor, device, trainable=True
    )
    model.training_modes = {"G": [0, 1, 2, 3], "S": [0, 1, 2, 3]}
    if parameter_scope == "lora_heads":
        for name, p in model.named_parameters():
            p.requires_grad_(
                p.requires_grad
                and (decision_head_parameter(name) or ".lora_A." in name or ".lora_B." in name)
            )
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    selected = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    for _, p in selected:
        p.data = p.data.float()
    reference = {name: p.detach().clone() for name, p in selected}
    for module in model.base_model.modules():
        setter = getattr(module, "set_activation_checkpointing", None)
        if callable(setter) and hasattr(module, "transformer"):
            setter("whole_layer")
    model.base_model.enable_input_require_grads()
    model.eval()
    groups = [
        {
            "params": [p for n, p in selected if not decision_head_parameter(n)],
            "lr": training.editor_content_lr,
        },
        {"params": [p for n, p in selected if decision_head_parameter(n)], "lr": training.editor_heads_lr},
    ]
    optimizer = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.0)
    recovered = None
    if resume and (output / "last.pt").exists():
        state = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        with torch.no_grad():
            for name, p in selected:
                p.copy_(state["parameters"][name].to(device))
        reference = {name: value.to(device) for name, value in state["reference"].items()}
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        recovered = state["kernel"]
    latest = None

    def checkpoint_state(kernel, force=False):
        nonlocal latest
        latest = kernel
        if force or kernel["cursor"] % 100 == 0:
            save(
                output,
                {
                    "parameters": {n: p.detach().cpu() for n, p in selected},
                    "reference": {n: v.detach().cpu() for n, v in reference.items()},
                    "optimizer": optimizer.state_dict(),
                    "rng": rng_state(),
                    "kernel": kernel,
                    "settings": vars(training),
                },
            )

    spec = {
        "branch": "E",
        "seed": training.seed,
        "batch_size": training.batch_size,
        "epochs": training.epochs,
        "reference_kl_weight": training.reference_kl_weight,
        "permute_atoms": training.permute_atoms,
    }
    _, _, _, history, _ = train_editor_minibatches(
        model,
        tokenizer,
        rows,
        spec,
        selected,
        reference,
        optimizer,
        build_repair_constraints(tokenizer),
        output,
        write_json,
        resume_state=recovered,
        checkpoint_callback=checkpoint_state,
    )
    checkpoint_state(latest, force=True)
    target = output / "checkpoint"
    model.save_pretrained(target, save_embedding_layers=False)
    tokenizer.save_pretrained(target)
    write_json(
        output / "training.json",
        {
            "status": "complete",
            "rows": len(rows),
            "sources": len({r["source_id"] for r in rows}),
            "optimizer_steps": len(history),
            "settings": vars(training),
            "parameter_scope": parameter_scope,
            "history": history,
        },
    )
    return target
