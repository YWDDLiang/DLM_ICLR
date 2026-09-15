"""Initialize the geometric editor and train conditional reconstruction on crystals."""

import re
import torch
from ..runtime.models import load_editor
from ..runtime.config import asset, run_root
from ..runtime.io import read_rows, write_json
from ..runtime.device import setup_device
from ..runtime.checkpoint import seed_all, save, rng_state, restore_rng
from .._core.fixed_slot import MASK_TOKEN_ID
from .warmup_objective import edit_warm_loss


def batch_loss(model, tokenizer, schema, rows, generators, budget):
    from types import SimpleNamespace
    from .warmup_forward import forward_many
    from .warmup_objective import warm_example, warm_loss_from_outputs

    examples = [
        warm_example(row, schema, rng, mask_id=MASK_TOKEN_ID, available_calls=budget)
        for row, rng in zip(rows, generators)
    ]
    observations = [e["scope"] for e in examples]
    contents = {}
    for i, example in enumerate(examples):
        if example["content"] is not None:
            contents[i] = len(observations)
            observations.append(example["content"])
    out, prefixes = forward_many(model, tokenizer, observations)

    def part(index):
        return SimpleNamespace(
            **{
                k: getattr(out, k)[index : index + 1]
                for k in ("logits", "mode_logits", "count_logits", "site_logits")
            }
        )

    loss = 0.0
    for i, example in enumerate(examples):
        j = contents.get(i)
        term, _ = warm_loss_from_outputs(
            example,
            schema,
            part(i),
            part(j) if j is not None else None,
            prefixes[j] if j is not None else None,
        )
        loss = loss + term
    return loss


class NumericVocabulary:
    def __init__(self, tokenizer):
        self.tables = {family: {} for family in ("LA", "LB", "LC", "AA", "AB", "AG", "X", "Y", "Z")}
        for text, token in tokenizer.get_vocab().items():
            match = re.fullmatch(r"<(LA|LB|LC|AA|AB|AG|X|Y|Z)_(\d+)>", text)
            if match:
                self.tables[match[1]][int(match[2])] = token
        self.tables = {f: dict(sorted(v.items())) for f, v in self.tables.items()}

    @staticmethod
    def family(position):
        return (
            ("LA", "LB", "LC", "AA", "AB", "AG")[position - 1] if position < 7 else "XYZ"[(position - 8) % 4]
        )

    def ids(self, position, *, canonical=True):
        family = self.family(position)
        return [
            token
            for bin_, token in self.tables[family].items()
            if not (canonical and family in "XYZ" and bin_ == 100)
        ]

    def log_probabilities(self, logits, position, temperature=1.0):
        family = self.family(position)
        ids = self.ids(position, canonical=False)
        vector = logits[ids].double()
        if family in "XYZ":
            vector = torch.cat((torch.logaddexp(vector[:1], vector[-1:]), vector[1:-1]))
        return self.ids(position), torch.log_softmax(vector / temperature, -1)


def train(config, *, device="cuda:0", resume=False):
    root, recipe = run_root(config), config["c2"]["training"]
    device = setup_device(device, threads=config["runtime"]["threads"])
    seed_all(recipe["warmup_seed"])
    model, tokenizer = load_editor(asset(config, "dlm"), asset(config, "b0"), device, trainable=True)
    output = root / "c2/warmup"
    extra = tuple(k for k in model.extra_modules() if k != "quality_head")
    selected = {}
    for name, p in model.named_parameters():
        p.requires_grad_(
            "lora_A" in name
            or "lora_B" in name
            or "state_conditioner" in name
            or any(name.startswith(k + ".") for k in extra)
        )
        if p.requires_grad:
            p.data = p.data.float()
            selected[name] = p
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        setter = getattr(module, "set_activation_checkpointing", None)
        if callable(setter) and hasattr(module, "transformer"):
            setter("whole_layer")
    model.base_model.enable_input_require_grads()
    schema = NumericVocabulary(tokenizer)
    rows = read_rows(root / "data/structures/train.jsonl")
    for row in rows:
        for site in range(row["plan_state"]["N"]):
            for axis, family in enumerate("XYZ"):
                pos = 8 + 4 * site + axis
                if row["body_token_ids"][pos] == schema.tables[family][100]:
                    row["body_token_ids"][pos] = schema.tables[family][0]
    optimizer = torch.optim.AdamW(selected.values(), lr=recipe["warmup_lr"], weight_decay=0.0)
    completed = 0
    start_epoch = cursor = 0
    if resume and (output / "last.pt").exists():
        state = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        with torch.no_grad():
            for name, value in state["parameters"].items():
                selected[name].copy_(value.to(device))
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        completed, start_epoch, cursor = state["step"], state["epoch"], state["cursor"]
    batch_size = recipe["warmup_batch_size"]
    for epoch in range(start_epoch, recipe["warmup_epochs"]):
        order = torch.randperm(
            len(rows), generator=torch.Generator().manual_seed(recipe["warmup_seed"] + epoch)
        ).tolist()
        for start in range(cursor if epoch == start_epoch else 0, len(rows), batch_size):
            indices = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            micro = recipe["warmup_micro_batch_size"]
            for offset in range(0, len(indices), micro):
                part = indices[offset : offset + micro]
                generators = [
                    torch.Generator().manual_seed(
                        recipe["warmup_seed"] + epoch * len(rows) + start + offset + i
                    )
                    for i in range(len(part))
                ]
                loss = batch_loss(
                    model,
                    tokenizer,
                    schema,
                    [rows[index] for index in part],
                    generators,
                    config["c2"]["max_calls"],
                )
                (loss / len(indices)).backward()
                total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(selected.values(), 1.0)
            optimizer.step()
            completed += 1
            end = start + len(indices)
            end_epoch = end == len(rows)
            if completed % 100 == 0 or end_epoch:
                save(
                    output,
                    {
                        "parameters": {n: p.detach().cpu() for n, p in selected.items()},
                        "optimizer": optimizer.state_dict(),
                        "rng": rng_state(),
                        "step": completed,
                        "epoch": epoch + int(end_epoch),
                        "cursor": 0 if end_epoch else end,
                        "settings": recipe,
                    },
                )
            if completed % 20 == 0:
                print({"stage": "c2-warmup", "step": completed, "loss": total / len(indices)}, flush=True)
    model.save_pretrained(output / "checkpoint", save_embedding_layers=False)
    tokenizer.save_pretrained(output / "checkpoint")
    write_json(
        output / "training.json",
        {"status": "complete", "updates": completed, "sources": len(rows), "settings": recipe},
    )
    return {"checkpoint": str(output / "checkpoint")}
