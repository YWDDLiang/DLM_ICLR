"""Fit a tractable periodic output distribution from partial crystal contexts."""

import math
import torch
from ..runtime.config import asset, run_root
from ..runtime.io import read_rows, fingerprint, write_json
from ..runtime.models import load_model_and_tokenizer
from ..runtime.device import setup_device
from ..runtime.checkpoint import seed_all, rng_state, restore_rng, save
from .._core.paired_noise import derive_subseed
from .distribution import PeriodicAxisHead
from .objectives import AxisTrainingSchema, axis_view, forward_views, loss_from_prediction


def load_head(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    with torch.random.fork_rng(devices=[]):
        head = PeriodicAxisHead(**saved["head_config"])
    head.load_state_dict(saved["state_dict"])
    return head.to(device).eval().requires_grad_(False)


def training_rows(path, schema):
    rows = read_rows(path)
    for row in rows:
        body = list(row["body_token_ids"])
        for i in range(row["plan_state"]["N"]):
            for axis in range(3):
                pos = 8 + 4 * i + axis
                body[pos] = schema.axis_tokens[axis][schema.coord_map[axis][body[pos]] % 100]
        row["body_token_ids"] = body
        row["_source_hash"] = row["source_id"]
        row["_target_hash"] = fingerprint(body)
    return rows


def train(config, *, device="cuda:0", resume=False):
    root, settings = run_root(config), config["periodic"]["training"]
    device = setup_device(device, threads=config["runtime"]["threads"])
    seed_all(settings["seed"])
    model, tokenizer = load_model_and_tokenizer(
        asset(config, "dlm"), asset(config, "constructor"), device, mean_resizing=False
    )
    model.eval().requires_grad_(False)
    schema = AxisTrainingSchema(tokenizer)
    rows = training_rows(root / "data/structures/train.jsonl", schema)
    validation = training_rows(root / "data/structures/val.jsonl", schema)[: settings["validation_limit"]]
    seed_all(settings["seed"])
    head = PeriodicAxisHead(
        model.get_input_embeddings().weight.shape[1],
        width=settings["width"],
        harmonics=settings["harmonics"],
        potential_bound=settings["potential_bound"],
    ).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=settings["learning_rate"], weight_decay=0.0)
    output = root / "periodic"
    step = epoch0 = cursor = 0
    best = math.inf
    if resume and (output / "last.pt").exists():
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        head.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        step, epoch0, cursor, best = saved["step"], saved["epoch"], saved["cursor"], saved["best_loss"]
        restore_rng(saved["rng"])

    def evaluate():
        head.eval()
        values = []
        validation_seed = derive_subseed(settings["seed"], "fixed_axis_VAL_v1")
        with torch.no_grad():
            for start in range(0, len(validation), settings["micro_batch_size"]):
                views = [
                    axis_view(r, schema, seed=validation_seed, epoch=0)
                    for r in validation[start : start + settings["micro_batch_size"]]
                ]
                predictions, _ = forward_views(model, tokenizer, views, max_length=settings["max_length"])
                values.extend(
                    float(
                        loss_from_prediction(head, schema, v, p, temperature=config["periodic"]["temperature"])[0]
                    )
                    for v, p in zip(views, predictions)
                )
        head.train()
        return sum(values) / len(values)

    if not resume:
        best = evaluate()
        save(
            output,
            {
                "schema": "periodic_axis_head_v1",
                "head_config": head.config(),
                "state_dict": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": rng_state(),
                "step": 0,
                "epoch": 0,
                "cursor": 0,
                "best_loss": best,
                "validation_nll": best,
                "settings": settings,
            },
            best=True,
        )
    for epoch in range(epoch0, settings["epochs"]):
        order = torch.randperm(
            len(rows),
            generator=torch.Generator().manual_seed(derive_subseed(settings["seed"], "axis_epoch", epoch)),
        ).tolist()
        for start in range(cursor if epoch == epoch0 else 0, len(rows), settings["effective_batch_size"]):
            selected = order[start : start + settings["effective_batch_size"]]
            views = [axis_view(rows[i], schema, seed=settings["seed"], epoch=epoch) for i in selected]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for micro in range(0, len(views), settings["micro_batch_size"]):
                part = views[micro : micro + settings["micro_batch_size"]]
                predictions, _ = forward_views(model, tokenizer, part, max_length=settings["max_length"])
                for view, pred in zip(part, predictions):
                    loss, _ = loss_from_prediction(
                        head, schema, view, pred, temperature=config["periodic"]["temperature"]
                    )
                    (loss / len(views)).backward()
                    losses.append(float(loss.detach()))
            optimizer.step()
            step += 1
            if step % 20 == 0:
                print({"stage": "periodic", "step": step, "nll": sum(losses) / len(losses)}, flush=True)
            end = start + len(selected)
            end_epoch = end == len(rows)
            if step % settings["save_every"] == 0 or end_epoch:
                measured = evaluate()
                improved = measured < best
                best = min(best, measured)
                save(
                    output,
                    {
                        "schema": "periodic_axis_head_v1",
                        "head_config": head.config(),
                        "state_dict": head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rng": rng_state(),
                        "step": step,
                        "epoch": epoch + int(end_epoch),
                        "cursor": 0 if end_epoch else end,
                        "best_loss": best,
                        "validation_nll": measured,
                        "settings": settings,
                    },
                    best=improved,
                )
                write_json(
                    output / "progress.json", {"step": step, "validation_nll": measured, "best_nll": best}
                )
    write_json(
        output / "training.json",
        {"status": "complete", "updates": step, "settings": settings, "best_nll": best},
    )
    return {"checkpoint": str(output / "best.pt"), "updates": step}
