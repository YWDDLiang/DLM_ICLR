"""Offline teacher distillation; this module is never imported during inference."""

from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import math
import time
import torch
from dlm_iclr.c2.feedback import INVALID
from dlm_iclr.runtime.io import read_rows, write_json
from dlm_iclr.runtime.models import load_editor
from dlm_iclr.c2.value import ValueNetwork, extract_features


def privileged_training_features(score):
    """The original twelve physical features, used only by the offline teacher."""
    hull = score.get("e_above_hull_eV_atom")
    known = hull is not None and math.isfinite(hull)
    hull = float(hull) if known else 0.0
    reliable = score["terminal_verified"] and known and score["terminal_status"] not in INVALID
    raw = score.get("raw") or {}

    def raw_value(key):
        value = raw.get(key)
        return (
            math.tanh(float(value) / 10) if isinstance(value, (int, float)) and math.isfinite(value) else 0.0
        )

    return [
        float(reliable and score["strict_stable"]),
        float(reliable and score["meta_stable"]),
        float(score.get("novel") is not None),
        float(score.get("novel") is True),
        float(reliable),
        float(score["terminal_status"] in INVALID),
        float(score["terminal_status"] == "not_converged"),
        float(known),
        max(-1.0, min(1.0, hull)),
        math.copysign(math.log1p(abs(hull) * 100), hull) / 5,
        raw_value("force_max_eV_A"),
        raw_value("stress_max_GPa"),
    ]


def nested_probabilities(logits):
    probability = logits.sigmoid()
    return torch.stack((probability[..., 0] * probability[..., 1], probability[..., 1]), dim=-1)


def fit_teacher(x, before, after, groups, config, *, output):
    """Small nested NS/NMS readout with source-balanced supervised loss."""
    device = x.device
    weights = torch.tensor(
        [1 / len(groups) / len(groups[source]) for source in range(len(groups)) for _ in groups[source]],
        device=device,
    )
    # Rows passed to this helper are grouped contiguously by source.
    mean = (x * weights[:, None]).sum(0)
    scale = ((x - mean).square() * weights[:, None]).sum(0).sqrt().clamp_min(0.05)
    z = (x - mean) / scale
    linear = torch.nn.Linear(x.shape[1], 2, device=device)
    torch.nn.init.zeros_(linear.weight)
    torch.nn.init.zeros_(linear.bias)
    optimizer = torch.optim.Adam(linear.parameters(), lr=1e-3)
    edges = []
    gains = after[:, 0] - before[:, 0]
    for group in groups:
        pairs = []
        for offset, a in enumerate([-1, *group]):
            for b in [-1, *group][offset + 1 :]:
                ga, gb = 0.0 if a == -1 else float(gains[a]), 0.0 if b == -1 else float(gains[b])
                if ga != gb:
                    pairs.append((a, b) if ga > gb else (b, a))
        edges.extend((a, b, 1 / max(1, len(pairs))) for a, b in pairs)
    if edges:
        a = torch.tensor([max(0, a) for a, b, w in edges], device=device)
        b = torch.tensor([max(0, b) for a, b, w in edges], device=device)
        am = torch.tensor([a == -1 for a, b, w in edges], device=device)
        bm = torch.tensor([b == -1 for a, b, w in edges], device=device)
        keep = torch.tensor([float(before[a if a >= 0 else b, 0]) for a, b, w in edges], device=device)
        ew = torch.tensor([w for a, b, w in edges], device=device)
        ew /= ew.sum()
    rng = torch.Generator().manual_seed(config.seed)
    for _ in range(config.value_epochs):
        order = torch.randperm(len(x), generator=rng)
        for start in range(0, len(x), 256):
            idx = order[start : start + 256].to(device)
            optimizer.zero_grad(set_to_none=True)
            errors = torch.nn.functional.binary_cross_entropy(
                nested_probabilities(linear(z[idx])), after[idx], reduction="none"
            )
            loss = (errors.mean(1) * weights[idx]).sum() * len(x) / len(
                idx
            ) + 0.1 * linear.weight.square().sum()
            if edges:
                values = nested_probabilities(linear(z))[:, 0]
                rank = torch.nn.functional.softplus(
                    -(torch.where(am, keep, values[a]) - torch.where(bm, keep, values[b]))
                )
                loss = loss + 0.2 * (rank * ew).sum()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        targets = nested_probabilities(linear(z)).detach()
    torch.save(
        {
            "weight": linear.weight.detach().cpu(),
            "bias": linear.bias.detach().cpu(),
            "mean": mean.cpu(),
            "scale": scale.cpu(),
            "training_only": True,
        },
        output / "offline_teacher.pt",
    )
    return targets


def train_value(config, data, output, *, editor_checkpoint=None, device="cuda:0"):
    rows, output = read_rows(data), Path(output)
    if not rows or any(row.get("source_split") != "train" for row in rows):
        raise ValueError("Value training requires nonempty TRAIN candidate comparisons")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_id"]].append(row)
    rows = [row for values in grouped.values() for row in values]
    groups, position = [], 0
    for values in grouped.values():
        groups.append(list(range(position, position + len(values))))
        position += len(values)
    output.mkdir(parents=True, exist_ok=True)
    training = config.training
    torch.manual_seed(training.seed)
    editor, tokenizer = load_editor(
        config.assets.base_model, editor_checkpoint or config.assets.editor, device
    )
    raw = extract_features(editor, tokenizer, rows, device)
    representatives = [rows[group[0]] for group in groups]
    keep_rows = [
        dict(row, proposal_tokens=row["current_tokens"], action_positions=[]) for row in representatives
    ]
    raw_keep = extract_features(editor, tokenizer, keep_rows, device)
    quality = editor.quality_head.state_dict()
    hidden_weight = quality["layers.0.weight"].detach().cpu().clone()
    hidden_bias = quality["layers.0.bias"].detach().cpu().clone()
    del editor
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    raw = raw.to(device)
    baseline = torch.stack(
        [raw_keep[group_index] for group_index, group in enumerate(groups) for _ in group]
    ).to(device)
    geometry = torch.tensor([row["geometry_features"] for row in rows], device=device)
    base_geometry = torch.tensor(
        [[row["num_sites"] / 20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for row in rows], device=device
    )
    before = torch.tensor([row["before_targets"] for row in rows], device=device)
    after = torch.tensor([row["after_targets"] for row in rows], device=device)
    with torch.no_grad():
        hidden = torch.nn.functional.silu(raw @ hidden_weight.to(device).T + hidden_bias.to(device))
        privileged = torch.tensor(
            [privileged_training_features(row["before_score"]) for row in rows], device=device
        )
        teacher_x = torch.cat((hidden, privileged, geometry), dim=1)
    teacher = fit_teacher(teacher_x, before, after, groups, training, output=output)
    soft_after = (1 - training.teacher_mix) * after + training.teacher_mix * teacher
    target, actual_target = soft_after - before, after - before
    model = ValueNetwork(raw.shape[1], hidden_weight.shape[0]).to(device)
    with torch.no_grad():
        model.hidden.weight.copy_(hidden_weight)
        model.hidden.bias.copy_(hidden_bias)
        model.head.weight.zero_()
        model.head.bias.zero_()
        weights = torch.tensor(
            [1 / len(groups) / len(group) for group in groups for _ in group], device=device
        )
        features = torch.cat(
            [model.features(raw[i : i + 256], geometry[i : i + 256]) for i in range(0, len(rows), 256)]
        )
        model.mean.copy_((features * weights[:, None]).sum(0))
        model.scale.copy_(((features - model.mean).square() * weights[:, None]).sum(0).sqrt().clamp_min(0.05))
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(
        [
            {"params": model.head.parameters(), "lr": training.value_head_lr},
            {"params": model.hidden.parameters(), "lr": training.value_hidden_lr},
        ],
        weight_decay=0.0,
    )
    rng, visits, steps, history = (
        torch.Generator().manual_seed(training.seed),
        torch.zeros(len(rows), dtype=torch.int64),
        0,
        [],
    )
    started = time.monotonic()
    for epoch in range(training.value_epochs):
        order = torch.randperm(len(groups), generator=rng).tolist()
        for start in range(0, len(order), training.value_sources_per_batch):
            chosen = [groups[i] for i in order[start : start + training.value_sources_per_batch]]
            flat = [i for group in chosen for i in group]
            idx = torch.tensor(flat, device=device)
            optimizer.zero_grad(set_to_none=True)
            value, keep = model(raw[idx], geometry[idx]), model(baseline[idx], base_geometry[idx])
            gain, terms, offset = value - keep, [], 0
            for group in chosen:
                sl = slice(offset, offset + len(group))
                loss = (gain[sl] - target[idx[sl]]).square().mean()
                loss += 0.1 * (
                    (keep[sl] - before[idx[sl]]).square().mean()
                    + (value[sl] - soft_after[idx[sl]]).square().mean()
                )
                scores = torch.cat((gain.new_zeros(1), gain[sl, 0]))
                gold = torch.cat((gain.new_zeros(1), actual_target[idx[sl], 0]))
                better = gold[:, None] > gold[None, :]
                if better.any():
                    loss += (
                        0.2
                        * torch.nn.functional.softplus(-(scores[:, None] - scores[None, :])[better]).mean()
                    )
                terms.append(loss)
                offset += len(group)
            loss = torch.stack(terms).mean() + 0.1 * model.head.weight.square().sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            visits[flat] += 1
            steps += 1
        event = {
            "epoch": epoch + 1,
            "optimizer_steps": steps,
            "loss": float(loss.detach()),
            "seconds": time.monotonic() - started,
        }
        history.append(event)
        write_json(output / "PROGRESS.json", event)
    deltas = {
        name: float((parameter.detach() - initial[name]).double().square().sum())
        for name, parameter in model.named_parameters()
    }
    model.cpu().save(
        output / "autonomous_value.pt",
        metadata={
            "training_rows": len(rows),
            "training_sources": len(groups),
            "teacher_mix": training.teacher_mix,
            "physical_features_at_inference": False,
        },
    )
    write_json(
        output / "training.json",
        {
            "rows": len(rows),
            "sources": len(groups),
            "optimizer_steps": steps,
            "minimum_visits": int(visits.min()),
            "maximum_visits": int(visits.max()),
            "history": history,
            "parameter_delta_squared": deltas,
            "teacher_inputs": "offline_current_physical_feedback",
            "inference_inputs": "DLM_features_and_nine_geometry_features",
            "settings": vars(training),
        },
    )
    return output / "autonomous_value.pt"
