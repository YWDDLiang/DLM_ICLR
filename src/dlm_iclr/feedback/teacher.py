"""Finite-candidate physical supervision mixed with the fitted editor targets."""

from copy import deepcopy
import math
import random
import torch
from .feedback import improves, INVALID


def utility(score):
    if score["terminal_status"] in INVALID:
        return 0.0
    hull = score.get("e_above_hull_eV_atom")
    if not score["terminal_verified"] or hull is None or not math.isfinite(hull):
        return None
    s, ms = score["strict_stable"], score["meta_stable"]
    if s is False and ms is False:
        return 0.0
    n = score.get("novel")
    if type(s) is not bool or type(ms) is not bool or type(n) is not bool:
        return None
    return (2 * int(n and s) + int(n and ms)) / 3


def probabilities(values, beta=0.1):
    gains = torch.as_tensor(values, dtype=torch.float64)
    return torch.softmax((gains - gains.max()) / beta, 0)


def make_row(reference, candidate, before, after):
    row = deepcopy(reference)
    for key in ("content_target_tokens", "content_positions"):
        row.pop(key, None)
    if candidate is None:
        row.update(
            proposal_tokens=list(row["current_tokens"]),
            action_positions=[],
            mode_target=0,
            site_targets=[0.0] * row["num_sites"],
            accept_target=0.0,
        )
        choice = "KEEP"
    else:
        action = candidate["trace"]["action"]
        tokens = candidate["trace"]["proposal_tokens"]
        row.update(
            proposal_tokens=tokens,
            action_positions=action["positions"],
            mode_target=action["mode"],
            site_targets=[float(i in action["sites"]) for i in range(row["num_sites"])],
            accept_target=float(improves(before, after)),
            content_target_tokens=tokens,
            content_positions=action["positions"],
        )
        choice = str(candidate["rank"])
    row["pair_id"] = reference["pair_id"] + ":physical_mix:" + choice
    return row


def build(original, bundles, scores, *, seed=20260911, beta=0.1, fraction=0.25):
    rng = random.Random(seed)
    indexed = {b["plan"]["source_id"]: b for b in bundles}
    scores = {k: {r["source_id"]: r for r in rows} for k, rows in scores.items()}
    result = []
    changed = 0
    for reference in original:
        source = reference["source_id"]
        before = scores["current"][source]
        base = utility(before)
        candidates = [c for c in indexed[source]["E"]["candidates"] if c["commit"]["applied"]]
        gains = [0.0]
        for c in candidates:
            value = utility(scores[f"candidate_{c['rank']}"][source])
            gains.append(value - base if base is not None and value is not None else None)
        row = deepcopy(reference)
        if base is not None and all(v is not None for v in gains):
            q = probabilities(gains, beta)
            if (
                not torch.allclose(q, torch.full_like(q, 1 / len(q)), atol=1e-12, rtol=1e-10)
                and rng.random() < fraction
            ):
                choice = rng.choices(range(len(gains)), weights=q.tolist(), k=1)[0]
                candidate = None if choice == 0 else candidates[choice - 1]
                after = before if candidate is None else scores[f"candidate_{candidate['rank']}"][source]
                row = make_row(reference, candidate, before, after)
                changed += 1
        result.append(row)
    return result, {
        "sources": len(result),
        "changed_targets": changed,
        "beta": beta,
        "teacher_fraction": fraction,
        "seed": seed,
    }
