"""LoRA plus trained input/output tables on the compact crystal vocabulary."""

import math
import os
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from ..runtime.io import read_rows, read_json, write_json
from ..runtime.config import run_root, asset
from ..runtime.device import setup_device
from ..runtime.checkpoint import seed_all, rng_state, restore_rng, save
from .objective import denoising_loss


class CrystalDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.rows = read_rows(path)
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        prompt_text = row["prompt"].rstrip() + "\n"
        prompt = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        sequence = self.tokenizer(prompt_text + row["answer"], add_special_tokens=False)["input_ids"]
        if len(sequence) > self.max_length:
            raise ValueError(f"Increase constructor.max_length for source {row['source_id']}")
        return sequence, len(prompt)


def collate(rows, pad_id):
    width = max(len(row[0]) for row in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    for i, (sequence, _) in enumerate(rows):
        ids[i, : len(sequence)] = torch.tensor(sequence)
        attention[i, : len(sequence)] = 1
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "prompt_lengths": torch.tensor([row[1] for row in rows]),
    }


def train(config, *, device="cuda:0", resume=False):
    from functools import partial
    from transformers import AutoTokenizer, AutoConfig
    from peft import LoraConfig, get_peft_model
    from ..runtime.models import model_class_for
    from .._core.llada_resize import ensure_llada_vocab_size
    from .._core.transformers_compat import ensure_create_bidirectional_mask, ensure_llada2_rope_parameters

    root, recipe = run_root(config), config["constructor"]
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        device = f"cuda:{os.environ['LOCAL_RANK']}"
        dist.init_process_group("nccl")
    device = setup_device(device, threads=config["runtime"]["threads"])
    seed_all(recipe["seed"] + rank)
    tokenizer = AutoTokenizer.from_pretrained(root / "data/tokenizer", trust_remote_code=True)
    ensure_create_bidirectional_mask()
    model_config = AutoConfig.from_pretrained(asset(config, "dlm"), trust_remote_code=True)
    ensure_llada2_rope_parameters(model_config)
    model = model_class_for(model_config).from_pretrained(
        asset(config, "dlm"),
        config=model_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    original_tokenizer = AutoTokenizer.from_pretrained(asset(config, "dlm"), trust_remote_code=True)
    new_tokens = len(tokenizer) - len(original_tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    ensure_llada_vocab_size(model, len(tokenizer))
    if new_tokens:
        with torch.no_grad():
            inputs = model.get_input_embeddings().weight
            outputs = model.get_output_embeddings().weight
            inputs[-new_tokens:] = inputs[:-new_tokens].mean(dim=0, keepdim=True)
            outputs[-new_tokens:] = outputs[:-new_tokens].mean(dim=0, keepdim=True)
    io_modules = {id(model.get_input_embeddings()), id(model.get_output_embeddings())}
    tables = [name for name, module in model.named_modules() if id(module) in io_modules]
    model = get_peft_model(
        model,
        LoraConfig(
            r=recipe["lora_r"],
            lora_alpha=recipe["lora_alpha"],
            lora_dropout=recipe["lora_dropout"],
            target_modules=recipe["target_modules"],
            modules_to_save=tables,
            bias="none",
            task_type=None,
        ),
    ).to(device)
    for module in model.modules():
        setter = getattr(module, "set_activation_checkpointing", None)
        if callable(setter) and hasattr(module, "transformer"):
            setter("whole_layer")
    model.enable_input_require_grads()
    parameters = [p for p in model.parameters() if p.requires_grad]
    max_length = (
        read_json(root / "data/statistics.json")["max_length"]
        if recipe["max_length"] == "auto"
        else recipe["max_length"]
    )
    dataset = CrystalDataset(root / "data/constructor/train.jsonl", tokenizer, max_length)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=0)
    loader = DataLoader(
        dataset,
        batch_size=recipe["batch_size"],
        sampler=sampler,
        collate_fn=partial(collate, pad_id=tokenizer.pad_token_id),
    )
    accum = recipe["effective_batch_size"] // (world * recipe["batch_size"])
    if accum < 1 or accum * world * recipe["batch_size"] != recipe["effective_batch_size"]:
        raise ValueError("effective_batch_size must be divisible by world size times batch_size")
    steps = math.ceil(len(loader) / accum) * recipe["epochs"]
    optimizer = torch.optim.AdamW(parameters, lr=recipe["lr"], weight_decay=recipe["weight_decay"])

    def multiplier(step):
        if step < recipe["warmup_steps"]:
            return max(1e-8, (step + 1) / recipe["warmup_steps"])
        progress = min(1.0, (step - recipe["warmup_steps"]) / max(1, steps - recipe["warmup_steps"]))
        return max(recipe["min_lr_ratio"], 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    output = root / "constructor"
    step = start_epoch = cursor = 0
    best_loss = float("inf")
    resumed_rng = None
    if resume and (output / "last.pt").exists():
        state = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if state["world_size"] != world:
            raise ValueError("Resume with the saved world size")
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, start_epoch, cursor, best_loss = (
            state["step"],
            state["epoch"],
            state["cursor"],
            state["best_loss"],
        )
        resumed_rng = state["rng"][rank]
    train_model = (
        torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index]) if world > 1 else model
    )
    validation = CrystalDataset(root / "data/constructor/val.jsonl", tokenizer, max_length)
    val_sampler = DistributedSampler(validation, num_replicas=world, rank=rank, shuffle=False)
    val_loader = DataLoader(
        validation,
        batch_size=recipe["batch_size"],
        sampler=val_sampler,
        collate_fn=partial(collate, pad_id=tokenizer.pad_token_id),
    )

    def validate():
        train_model.eval()
        losses = []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= recipe["eval_max_batches"]:
                    break
                losses.append(denoising_loss(train_model, {k: v.to(device) for k, v in batch.items()}))
        result = torch.stack(losses).mean()
        if world > 1:
            dist.all_reduce(result)
            result /= world
        train_model.train()
        return float(result)

    def checkpoint(epoch, next_cursor):
        nonlocal best_loss
        val_loss = validate()
        improved = val_loss < best_loss
        best_loss = min(best_loss, val_loss)
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, rng_state())
        else:
            states[0] = rng_state()
        if rank == 0:
            from peft import get_peft_model_state_dict

            save(
                output,
                {
                    "model": get_peft_model_state_dict(model, save_embedding_layers=False),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng": states,
                    "step": step,
                    "epoch": epoch,
                    "cursor": next_cursor,
                    "world_size": world,
                    "best_loss": best_loss,
                    "settings": recipe,
                },
                best=improved,
            )
            write_json(
                output / "progress.json",
                {"updates": step, "total_updates": steps, "validation_loss": val_loss},
            )
        if world > 1:
            dist.barrier()

    train_model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, recipe["epochs"]):
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        if resumed_rng is not None:
            restore_rng(resumed_rng)
            resumed_rng = None
        for micro, batch in enumerate(iterator):
            if epoch == start_epoch and micro < cursor:
                continue
            boundary = (micro + 1) % accum == 0 or micro + 1 == len(loader)
            context = train_model.no_sync() if world > 1 and not boundary else nullcontext()
            with context:
                loss = denoising_loss(train_model, {k: v.to(device) for k, v in batch.items()})
                (loss / accum).backward()
            if boundary:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if rank == 0 and step % 20 == 0:
                    print({"stage": "constructor", "step": step, "total": steps, "loss": float(loss)}, flush=True)
                if step % recipe["save_steps"] == 0 or step == steps:
                    checkpoint(
                        epoch + int(micro + 1 == len(loader)), 0 if micro + 1 == len(loader) else micro + 1
                    )
        cursor = 0
    if rank == 0:
        model.save_pretrained(output / "final", save_embedding_layers=False)
        tokenizer.save_pretrained(output / "final")
        write_json(
            output / "training.json",
            {"status": "complete", "updates": step, "settings": recipe, "max_length": max_length},
        )
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return {"checkpoint": str(output / "final"), "updates": step}
