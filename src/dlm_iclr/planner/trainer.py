#!/usr/bin/env python3
"""LoRA SFT for the H1 Llama formula planner."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from dlm_iclr._core.h1_llm_planner import (
    disable_peft_bnb_autodetect,
    ensure_peft_cache_compat,
    load_llama3_compatible_config,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def format_prompt(tokenizer, record: dict[str, Any]) -> str:
    prompt = record.get("prompt")
    if prompt:
        return str(prompt)
    messages = record.get("messages")
    if (
        isinstance(messages, list)
        and hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None)
    ):
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    if isinstance(messages, list) and len(messages) >= 2:
        return f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"
    raise ValueError("H1 SFT record has neither messages nor prompt")


class FormulaPlanDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int) -> None:
        self.rows = list(iter_jsonl(path))
        if not self.rows:
            raise ValueError(f"No rows found in {path}")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        prompt = format_prompt(self.tokenizer, row)
        answer = str(row["answer"]).strip()
        eos = self.tokenizer.eos_token or ""
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer + eos, add_special_tokens=False)["input_ids"]
        if len(answer_ids) >= self.max_length:
            raise ValueError(
                f"H1 formula answer uses {len(answer_ids)} tokens, which does not fit max_length={self.max_length}"
            )
        max_prompt_tokens = self.max_length - len(answer_ids)
        if len(prompt_ids) > max_prompt_tokens:
            # Preserve the task instruction nearest the answer boundary.  This
            # also guarantees that answer + EOS are always supervised.
            prompt_ids = prompt_ids[-max_prompt_tokens:]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_weight": float(row.get("sample_weight", 1.0) or 1.0),
        }


class TrainingSampler(RandomSampler):
    """Keep the original RandomSampler order so a stage can resume mid-epoch."""

    def __init__(self, dataset: Dataset) -> None:
        super().__init__(dataset)
        self.order: list[int] | None = None
        self.cursor = 0

    def __iter__(self):
        if self.order is None or self.cursor >= len(self.order):
            self.order = list(super().__iter__())
            self.cursor = 0
        while self.cursor < len(self.order):
            index = self.order[self.cursor]
            self.cursor += 1
            yield index


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def link_or_copy_tree(source: Path, target: Path) -> None:
    """Keep best/final adapters as hard links to an immutable saved snapshot."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for file in source.iterdir():
        if not file.is_file():
            continue
        destination = target / file.name
        try:
            os.link(file, destination)
        except OSError:
            shutil.copy2(file, destination)


def save_stage_checkpoint(
    output_dir: Path,
    model,
    tokenizer,
    optimizer,
    scheduler,
    sampler: TrainingSampler,
    *,
    global_step: int,
    micro_step: int,
    running: float,
    history: list[dict[str, Any]],
    run_identity: dict[str, Any],
    best_eval_loss: float | None,
    best_step: int | None,
    evaluated_loss: float | None,
) -> tuple[float | None, int | None]:
    rng = capture_rng_state()
    new_best = evaluated_loss is not None and (best_eval_loss is None or evaluated_loss < best_eval_loss)
    if new_best:
        best_eval_loss, best_step = evaluated_loss, global_step
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    temporary = checkpoints / "last.tmp"
    last = checkpoints / "last"
    previous = checkpoints / "last.previous"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        adapter = temporary / "adapter"
        adapter.mkdir()
        model.save_pretrained(adapter)
        tokenizer.save_pretrained(adapter)
        torch.save(
            {
                "global_step": global_step,
                "micro_step": micro_step,
                "running": running,
                "history": history,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "sampler_order": sampler.order,
                "sampler_cursor": sampler.cursor,
                "rng": rng,
                "best_eval_loss": best_eval_loss,
                "best_step": best_step,
                "run_identity": run_identity,
            },
            temporary / "state.pt",
        )
        if previous.exists():
            shutil.rmtree(previous)
        if last.exists():
            last.rename(previous)
        temporary.rename(last)
        if previous.exists():
            shutil.rmtree(previous)
        if new_best:
            link_or_copy_tree(last / "adapter", checkpoints / "best")
            (checkpoints / "best.json").write_text(
                json.dumps({"step": best_step, "eval_loss": best_eval_loss}, indent=2) + "\n",
                encoding="utf-8",
            )
        return best_eval_loss, best_step
    finally:
        restore_rng_state(rng)


def collate(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.full((len(batch), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    weights = torch.ones((len(batch),), dtype=torch.float32)
    for idx, item in enumerate(batch):
        length = item["input_ids"].numel()
        input_ids[idx, :length] = item["input_ids"]
        attention_mask[idx, :length] = 1
        labels[idx, :length] = item["labels"]
        weights[idx] = float(item["sample_weight"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "sample_weight": weights,
    }


def weighted_token_loss(
    logits: torch.Tensor, labels: torch.Tensor, sample_weight: torch.Tensor
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    losses = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    mask = (shift_labels != -100).float()
    per_sample = (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    weights = sample_weight.to(device=per_sample.device, dtype=per_sample.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, max_batches: int) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = weighted_token_loss(outputs.logits, batch["labels"], batch["sample_weight"])
        total += float(loss.item())
        count += 1
    model.train()
    return total / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional PEFT/LoRA adapter checkpoint to continue training from.",
    )
    parser.add_argument(
        "--resume-from", type=Path, default=None, help="Full state checkpoint within the same stage."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--eval-max-batches", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    args = parser.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adapter_source = args.resume_from / "adapter" if args.resume_from else args.checkpoint_path
    tokenizer_source = (
        adapter_source
        if adapter_source and (Path(adapter_source) / "tokenizer_config.json").exists()
        else args.model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    config = load_llama3_compatible_config(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    ensure_peft_cache_compat()
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    disable_peft_bnb_autodetect()

    if adapter_source:
        model = PeftModel.from_pretrained(model, adapter_source, is_trainable=True)
    else:
        lora = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)
    model.to(device)
    model.train()
    parameter_report = {
        "base_requires_grad": [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_" not in name
        ],
        "trainable_lora_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ),
    }
    (args.output_dir / "parameter_report.json").write_text(
        json.dumps(parameter_report, indent=2) + "\n", encoding="utf-8"
    )

    train_ds = FormulaPlanDataset(args.data_dir / "train.jsonl", tokenizer, args.max_length)
    val_ds = FormulaPlanDataset(args.data_dir / "val.jsonl", tokenizer, args.max_length)
    train_sampler = TrainingSampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        sampler=train_sampler,
        collate_fn=lambda batch: collate(batch, int(tokenizer.pad_token_id)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=lambda batch: collate(batch, int(tokenizer.pad_token_id)),
    )

    updates_per_epoch = math.ceil(len(train_loader) / max(1, int(args.grad_accum)))
    total_updates = max(1, int(math.ceil(float(args.epochs) * updates_per_epoch)))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_steps), num_training_steps=total_updates
    )

    run_identity = {
        "model_path": args.model_path,
        "checkpoint_path": args.checkpoint_path,
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "total_updates": total_updates,
        "seed": int(args.seed),
    }

    resume_state = (
        torch.load(args.resume_from / "state.pt", map_location="cpu", weights_only=False)
        if args.resume_from
        else None
    )
    if resume_state:
        if resume_state["run_identity"] != run_identity:
            raise ValueError("Within-stage checkpoint does not match this stage's run identity")
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        train_sampler.order = resume_state["sampler_order"]
        train_sampler.cursor = int(resume_state["sampler_cursor"])

    write_payload = {
        "model_path": args.model_path,
        "checkpoint_path": args.checkpoint_path,
        "resume_from": str(args.resume_from) if args.resume_from else None,
        "data_dir": str(args.data_dir),
        "max_length": int(args.max_length),
        "epochs": float(args.epochs),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "lr": float(args.lr),
        "total_updates": total_updates,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "lora": {
            "r": int(args.lora_r),
            "alpha": int(args.lora_alpha),
            "dropout": float(args.lora_dropout),
            "continued_from_checkpoint": bool(args.checkpoint_path),
        },
    }
    (args.output_dir / "train_config.json").write_text(
        json.dumps(write_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    start = time.time()
    global_step = int(resume_state["global_step"]) if resume_state else 0
    micro_step = int(resume_state["micro_step"]) if resume_state else 0
    running = float(resume_state["running"]) if resume_state else 0.0
    history: list[dict[str, Any]] = list(resume_state["history"]) if resume_state else []
    best_eval_loss = resume_state["best_eval_loss"] if resume_state else None
    best_step = resume_state["best_step"] if resume_state else None
    model.zero_grad(set_to_none=True)
    progress = tqdm(total=total_updates, initial=global_step, desc="H1 Llama planner SFT")
    while global_step < total_updates:
        train_iter = iter(train_loader)
        if resume_state is not None:
            # The saved RNG already includes the original iterator seed here.
            restore_rng_state(resume_state["rng"])
            resume_state = None
        for batch in train_iter:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = weighted_token_loss(outputs.logits, batch["labels"], batch["sample_weight"])
            (loss / max(1, int(args.grad_accum))).backward()
            running += float(loss.item())
            micro_step += 1
            if micro_step % max(1, int(args.grad_accum)) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)
                if global_step % int(args.logging_steps) == 0 or global_step == 1:
                    event = {
                        "step": global_step,
                        "train_loss_recent": running / max(1, int(args.logging_steps)),
                        "lr": float(scheduler.get_last_lr()[0]),
                        "elapsed_sec": time.time() - start,
                    }
                    running = 0.0
                    print(json.dumps(event, ensure_ascii=False), flush=True)
                    history.append(event)
                evaluated_loss = None
                if int(args.eval_steps) > 0 and global_step % int(args.eval_steps) == 0:
                    eval_loss = evaluate(model, val_loader, device, int(args.eval_max_batches))
                    evaluated_loss = eval_loss
                    event = {"step": global_step, "eval_loss": eval_loss, "elapsed_sec": time.time() - start}
                    print(json.dumps(event, ensure_ascii=False), flush=True)
                    history.append(event)
                if int(args.save_steps) > 0 and global_step % int(args.save_steps) == 0:
                    best_eval_loss, best_step = save_stage_checkpoint(
                        args.output_dir,
                        model,
                        tokenizer,
                        optimizer,
                        scheduler,
                        train_sampler,
                        global_step=global_step,
                        micro_step=micro_step,
                        running=running,
                        history=history,
                        run_identity=run_identity,
                        best_eval_loss=best_eval_loss,
                        best_step=best_step,
                        evaluated_loss=evaluated_loss,
                    )
                if global_step >= total_updates:
                    break
        if global_step >= total_updates:
            break

    final_eval = evaluate(model, val_loader, device, int(args.eval_max_batches))
    best_eval_loss, best_step = save_stage_checkpoint(
        args.output_dir,
        model,
        tokenizer,
        optimizer,
        scheduler,
        train_sampler,
        global_step=global_step,
        micro_step=micro_step,
        running=running,
        history=history,
        run_identity=run_identity,
        best_eval_loss=best_eval_loss,
        best_step=best_step,
        evaluated_loss=final_eval,
    )
    final_dir = args.output_dir / "final"
    link_or_copy_tree(args.output_dir / "checkpoints" / "last" / "adapter", final_dir)
    metrics = {
        "global_step": global_step,
        "final_eval_loss": final_eval,
        "best_eval_loss": best_eval_loss,
        "best_step": best_step,
        "elapsed_sec": time.time() - start,
        "history": history,
    }
    (args.output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
