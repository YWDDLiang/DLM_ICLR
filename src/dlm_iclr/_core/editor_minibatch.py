"""Retained crystal DLM implementation; see docs/reproduction.md for the public workflow."""

from collections import Counter
import copy
import json
import random
import time
import torch
from dlm_iclr._core.editor_t2t import dense_loss, dense_vectors
from dlm_iclr._core.expert_edit import ExpertEditObjective
from dlm_iclr._core.rsi_minibatch import (
    decision_head_parameter,
    editor_head_loss,
    epoch_indices,
    has_head_supervision,
)


def permute_atoms(row, permutation):
    """Change serialization order consistently; preserve the physical pair."""
    n = row["num_sites"]
    if sorted(permutation) != list(range(n)):
        raise ValueError("atom permutation must be a bijection")
    result = copy.deepcopy(row)
    old_to_new = {old: new for new, old in enumerate(permutation)}
    for key in (
        "current_tokens",
        "proposal_tokens",
        "content_target_tokens",
        "chosen_tokens",
        "rejected_tokens",
        "healthy_anchor_tokens",
    ):
        body = result.get(key)
        if body is not None:
            if len(body) != 7 + 4 * n:
                raise ValueError("atom permutation requires exact-length bodies")
            result[key] = body[:7] + [t for old in permutation for t in body[7 + 4 * old : 11 + 4 * old]]
    for key in ("action_positions", "content_positions"):
        if key in result:
            result[key] = sorted(
                p if p < 7 else 7 + 4 * old_to_new[(p - 7) // 4] + (p - 7) % 4 for p in result[key]
            )
    if result.get("site_targets") is not None:
        result["site_targets"] = [result["site_targets"][old] for old in permutation]
    return result


def train_editor_minibatches(
    model,
    tokenizer,
    examples,
    spec,
    selected,
    reference,
    optimizer,
    support,
    output,
    write_json,
    *,
    resume_state=None,
    checkpoint_callback=None,
    schedule=None,
):
    import torch.distributed as dist

    device = next(model.parameters()).device
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    content = [(n, p) for n, p in selected if not decision_head_parameter(n)]
    heads = [(n, p) for n, p in selected if decision_head_parameter(n)]
    if spec["branch"] != "E" or not content or not heads:
        raise ValueError("editor minibatches require content modules and decision heads")
    objective = ExpertEditObjective(tokenizer, device, temperature=0.7)
    rng = random.Random(spec["seed"] + rank)
    started = time.monotonic()
    inspected, content_visits, head_visits, token_visits, site_visits = (Counter() for _ in range(5))
    history = []
    content_steps = head_steps = total_heads = 0
    timed_out = False
    cursor, elapsed = 0, 0.0
    if resume_state is not None:
        rng.setstate(resume_state["augmentation_rng"])
        history, cursor, elapsed = resume_state["history"], resume_state["cursor"], resume_state["elapsed"]
        inspected, content_visits, head_visits, token_visits, site_visits = (
            Counter(resume_state[name])
            for name in ("inspected", "content_visits", "head_visits", "token_visits", "site_visits")
        )
        content_steps, head_steps, total_heads = (
            resume_state[name] for name in ("content_steps", "head_steps", "total_heads")
        )

    def local_state():
        return {
            "cursor": len(history),
            "augmentation_rng": rng.getstate(),
            "history": history,
            "elapsed": elapsed + time.monotonic() - started,
            "inspected": dict(inspected),
            "content_visits": dict(content_visits),
            "head_visits": dict(head_visits),
            "token_visits": dict(token_visits),
            "site_visits": dict(site_visits),
            "content_steps": content_steps,
            "head_steps": head_steps,
            "total_heads": total_heads,
        }

    if checkpoint_callback is not None:
        paused = checkpoint_callback(local_state(), force=True)
        if paused is not None:
            return paused
    if schedule is None:
        schedule = epoch_indices(
            len(examples),
            batch_size=spec["batch_size"],
            world=world,
            epochs=spec["epochs"],
            seed=spec["seed"],
        )
    for batch_index, (epoch, indices) in enumerate(schedule):
        if batch_index < cursor:
            continue
        width = len(indices) // world
        rows = []
        for index in indices[rank * width : (rank + 1) * width]:
            row = examples[index]
            if spec.get("permute_atoms", True):
                permutation = list(range(row["num_sites"]))
                rng.shuffle(permutation)
                row = permute_atoms(row, permutation)
            else:
                row = copy.deepcopy(row)
            if row.get("content_target_tokens"):
                order = row["content_positions"]
                row["training_cut"] = 0 if rng.random() < 0.5 else rng.randrange(len(order))
            rows.append(row)
        inspected.update(r["pair_id"] for r in rows)
        optimizer.zero_grad(set_to_none=True)
        targets = [r for r in rows if r.get("content_target_tokens")]
        ce_value = kl_value = 0.0
        if targets:
            live = {n: p.detach().clone() for n, p in content}
            with torch.no_grad():
                try:
                    for n, p in content:
                        p.copy_(reference[n])
                    q = dense_vectors(model, tokenizer, targets, objective, masked=True)
                finally:
                    for n, p in content:
                        p.copy_(live[n])
            del live
            p_vectors = dense_vectors(model, tokenizer, targets, objective, masked=True)
            ce, kl, _ = dense_loss(p_vectors, q, targets)
            # A source contributes once regardless of how many numeric tokens its edit opens.
            loss = (ce + spec["reference_kl_weight"] * kl) / width
            loss.backward()
            ce_value, kl_value = float(ce.detach()) / len(targets), float(kl.detach()) / len(targets)
            del loss, ce, kl, p_vectors, q
        loss_heads, count = editor_head_loss(
            model, tokenizer, rows, device, detach_content=True, site_objective="categorical"
        )
        head_value = None
        if loss_heads is not None:
            loss_heads.backward()
            head_value = float(loss_heads.detach())
            del loss_heads
        active = torch.tensor([bool(targets), count > 0], device=device, dtype=torch.int64)
        if world > 1:
            dist.all_reduce(active, op=dist.ReduceOp.MAX)
        if not bool(active.any()):
            raise ValueError("minibatch has no supervised content or decisions")
        norms = {}
        for name, parameters, present in zip(("content", "heads"), (content, heads), active.tolist()):
            if not present:
                for _, param in parameters:
                    param.grad = None
                continue
            for _, param in parameters:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                if world > 1:
                    dist.all_reduce(param.grad)
                    param.grad.div_(world)
            norms[name] = float(
                torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1.0, error_if_nonfinite=True)
            )
        optimizer.step()
        if bool(active[0]):
            content_steps += 1
            content_visits.update(r["pair_id"] for r in targets)
            for row in targets:
                token_visits[row["pair_id"]] += len(row["content_positions"]) - row["training_cut"]
        if bool(active[1]):
            head_steps += 1
            total_heads += count
            head_visits.update(r["pair_id"] for r in rows if has_head_supervision(r))
            for row in rows:
                if row.get("mode_target") == 1:
                    site_visits.update(i for i, value in enumerate(row["site_targets"]) if value)
        event = dict(
            step=len(history) + 1,
            epoch=epoch,
            dense_CE=ce_value,
            reference_KL=kl_value,
            decision_loss=head_value,
            gradient_norms=norms,
            source_examples_per_GPU=width,
            content_optimizer_steps=content_steps,
            head_optimizer_steps=head_steps,
            content_rows=len(targets),
            head_rows=count,
            update_kind="minibatch",
            seconds=elapsed + time.monotonic() - started,
            peak_GPU_GB=torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0,
        )
        history.append(event)
        if rank == 0:
            write_json(output / "PROGRESS.json", event)
            print(json.dumps(event), flush=True)
        if checkpoint_callback is not None:
            paused = checkpoint_callback(local_state())
            if paused is not None:
                return paused

    report = dict(
        pair_visits=dict(inspected),
        pair_visits_semantics="actual_minibatches",
        content_updated_pair_visits=dict(content_visits),
        head_updated_pair_visits=dict(head_visits),
        content_updated_token_visits=dict(token_visits),
        local_site_target_visits=dict(site_visits),
        content_optimizer_steps=content_steps,
        head_optimizer_steps=head_steps,
        KL_stop=False,
        KL_trigger=None,
        head_continuation_after_KL=False,
        content_update_unit="minibatch",
        reference_KL_policy="regularization_only",
        consistent_atom_permutations=spec.get("permute_atoms", True),
        epochs_cap=spec["epochs"],
        timed_out=timed_out,
    )
    write_json(output / f"EXPOSURE_rank{rank}.json", report)
    return len(history), 0, total_heads, history, rng
