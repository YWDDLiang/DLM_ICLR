"""Generator and editor updates with complete data passes and soft reference KL."""

from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import random
import time
import torch
from crystal_dlm.r03_physics_transfer import build_repair_constraints
from crystal_dlm.rsi_minibatch import epoch_indices, training_view, conditional_batch, decision_head_parameter
from .io import read_rows, write_json
from .models import load_model_and_tokenizer, load_editor


@contextmanager
def reference_parameters(selected, reference):
    live = {name: parameter.detach().clone() for name, parameter in selected}
    try:
        with torch.no_grad():
            for name, parameter in selected:
                parameter.copy_(reference[name])
        yield
    finally:
        with torch.no_grad():
            for name, parameter in selected:
                parameter.copy_(live[name])


def train_generator(model, tokenizer, rows, config, selected, reference, optimizer, output):
    from crystal_dlm.rsi_preference import numeric_order
    import torch.distributed as dist

    world, rank = (dist.get_world_size(), dist.get_rank()) if dist.is_initialized() else (1, 0)
    device = next(model.parameters()).device
    support = build_repair_constraints(tokenizer)
    rng, history, visits = random.Random(config.seed + rank), [], Counter()
    started = time.monotonic()
    for epoch, indices in epoch_indices(
        len(rows),
        batch_size=config.generator_batch_size,
        world=world,
        epochs=config.generator_epochs,
        seed=config.seed,
    ):
        width = len(indices) // world
        batch = [rows[i] for i in indices[rank * width : (rank + 1) * width]]
        views, pairs, anchors = [], [], []
        for row in batch:
            order = numeric_order(row["num_sites"], "G")
            for cut in rng.sample(range(len(order)), min(config.mask_cuts, len(order))):
                seed = rng.randrange(2**63)
                if row.get("chosen_tokens") is not None and row.get("rejected_tokens") is not None:
                    pairs.append((len(views), len(views) + 1))
                    views.extend(
                        training_view(row, row[name], cut, "G", mask_seed=seed)
                        for name in ("chosen_tokens", "rejected_tokens")
                    )
                elif row.get("healthy_anchor_tokens") is not None:
                    anchors.append(len(views))
                    views.append(training_view(row, row["healthy_anchor_tokens"], cut, "G", mask_seed=seed))
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), reference_parameters(selected, reference):
            reference_logp, q = conditional_batch(model, tokenizer, views, "G", support)
        actual, p = conditional_batch(model, tokenizer, views, "G", support)
        terms = [
            -torch.nn.functional.logsigmoid(
                config.generator_beta * ((actual[a] - reference_logp[a]) - (actual[b] - reference_logp[b]))
            )
            - config.generator_anchor_weight * actual[a]
            for a, b in pairs
        ]
        terms.extend(-config.generator_anchor_weight * actual[i] for i in anchors)
        kl = torch.stack([(qi.exp() * (qi - pi)).sum() for qi, pi in zip(q, p, strict=True)]).mean()
        loss = torch.stack(terms).sum() / (width * config.mask_cuts) + config.reference_kl_weight * kl
        loss.backward()
        if world > 1:
            for _, parameter in selected:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(world)
        norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in selected], 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        visits.update(row["source_id"] for row in batch)
        event = {
            "step": len(history) + 1,
            "epoch": epoch + 1,
            "loss": float(loss.detach()),
            "reference_KL": float(kl.detach()),
            "gradient_norm": float(norm),
            "seconds": time.monotonic() - started,
        }
        history.append(event)
        if rank == 0:
            write_json(output / "PROGRESS.json", event)
            print(event, flush=True)
    write_json(
        output / f"exposure_{rank}.json", {"source_visits": dict(visits), "optimizer_steps": len(history)}
    )
    return history


def train_actor(branch, config, data, output, *, checkpoint=None, device="cuda:0"):
    import torch.distributed as dist

    rows, output = read_rows(data), Path(output)
    if not rows or any(row.get("source_split") != "train" for row in rows):
        raise ValueError("Actor training requires nonempty, explicitly TRAIN examples")
    output.mkdir(parents=True, exist_ok=True)
    training = config.training
    torch.manual_seed(training.seed)
    random.seed(training.seed)
    if branch == "G":
        model, tokenizer = load_model_and_tokenizer(
            config.assets.base_model, checkpoint or config.assets.generator, device, mean_resizing=False
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(".lora_A." in name or ".lora_B." in name)
    else:
        model, tokenizer = load_editor(
            config.assets.base_model, checkpoint or config.assets.editor, device, trainable=True
        )
        model.training_modes = {"G": [0, 1, 2, 3], "S": [0, 1, 2, 3]}
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    selected = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not selected:
        raise ValueError("The starting checkpoint contains no trainable LoRA or editor parameters")
    for _, parameter in selected:
        parameter.data = parameter.data.float()
    reference = {name: parameter.detach().clone() for name, parameter in selected}
    base = model.base_model if branch == "E" else model
    for module in base.modules():
        setter = getattr(module, "set_activation_checkpointing", None)
        if callable(setter) and hasattr(module, "transformer"):
            setter("whole_layer")
    base.enable_input_require_grads()
    model.eval()
    groups = [
        {
            "params": [p for n, p in selected if not decision_head_parameter(n)],
            "lr": training.generator_lr if branch == "G" else training.editor_content_lr,
        },
        {"params": [p for n, p in selected if decision_head_parameter(n)], "lr": training.editor_heads_lr},
    ]
    optimizer = torch.optim.AdamW([group for group in groups if group["params"]], weight_decay=0.0)
    if branch == "G":
        history = train_generator(model, tokenizer, rows, training, selected, reference, optimizer, output)
    else:
        from crystal_dlm.editor_minibatch import train_editor_minibatches

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
        )
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        target = output / "checkpoint"
        # The trained IO tables are already part of PEFT's modules_to_save.
        # Avoid exporting a second copy under the base embedding names.
        model.save_pretrained(target, save_embedding_layers=False)
        tokenizer.save_pretrained(target)
        deltas = {
            name: float((parameter.detach() - reference[name]).double().square().sum())
            for name, parameter in selected
        }
        write_json(
            output / "training.json",
            {
                "branch": branch,
                "rows": len(rows),
                "sources": len({r["source_id"] for r in rows}),
                "optimizer_steps": len(history),
                "parameter_delta_squared": deltas,
                "reference_KL": "soft_regularization",
                "settings": vars(training),
                "history": history,
            },
        )
    if dist.is_initialized():
        dist.barrier()
    return output / "checkpoint"
