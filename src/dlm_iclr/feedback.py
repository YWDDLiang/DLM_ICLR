"""Compile offline supervision from clean TRAIN sources and measured endpoints."""

from __future__ import annotations
import copy
import math
from crystal_dlm.r03_physics_transfer import geometry_support_report
from .io import fingerprint
from .proposals import action_positions, COUNTS, MODES

INVALID = {"generation_failure", "invalid_raw", "invalid_terminal"}


def endpoint_targets(score):
    if score["terminal_status"] in INVALID:
        return (0.0, 0.0)
    if score["terminal_status"] == "not_converged" and score.get("e_above_hull_eV_atom") is not None:
        return (0.0, 0.0)
    if not score["terminal_verified"] or score.get("e_above_hull_eV_atom") is None:
        return None
    if not score["strict_stable"] and not score["meta_stable"]:
        return (0.0, 0.0)
    if score.get("novel") is None:
        return None
    return (float(score["strict_stable"] and score["novel"]), float(score["meta_stable"] and score["novel"]))


def quality(score):
    hull = score.get("e_above_hull_eV_atom")
    if not score["terminal_verified"] or hull is None or not math.isfinite(hull):
        return None
    rank = (
        4
        if score["strict_stable"] and score.get("novel") is True
        else 3
        if hull <= 0
        else 2
        if hull <= 0.1
        else 0
    )
    return rank, -hull


def improves(before, after):
    a, b = quality(before), quality(after)
    if b is None:
        return False
    if a is None:
        return before["terminal_status"] in INVALID and b[0] >= 2
    return b[0] > a[0] or (b[0] == a[0] and b[0] != 4 and b[1] - a[1] > 0.01)


def align_tokens(target, reference):
    if len(target) != len(reference) or target[0] != reference[0]:
        raise ValueError("Teacher changed atom count")
    remaining = [target[i : i + 4] for i in range(7, len(target), 4)]
    ordered = target[:7]
    for i in range(7, len(reference), 4):
        match = next((j for j, block in enumerate(remaining) if block[0] == reference[i]), None)
        if match is None:
            raise ValueError("Teacher changed species counts")
        ordered += remaining.pop(match)
    return ordered


def target_action(before, target):
    n = (len(before) - 7) // 4
    sites = [i for i in range(n) if before[8 + 4 * i : 11 + 4 * i] != target[8 + 4 * i : 11 + 4 * i]]
    if before[1:7] != target[1:7]:
        mode, sites = 3, list(range(n))
    elif not sites:
        mode = 0
    else:
        available = [count for count in COUNTS if len(sites) <= count <= n]
        if available:
            mode = 1
            sites += [i for i in range(n) if i not in sites][: available[0] - len(sites)]
            sites.sort()
        else:
            mode, sites = 2, list(range(n))
    return {
        "mode": mode,
        "name": MODES[mode],
        "sites": sites,
        "positions": action_positions(n, mode, sites) if mode else [],
    }


def generator_target_records(plans, generated, current, edited, scores, tokenizer):
    """Create the exact full-token teacher structures before labelling them."""
    from .generation import make_record

    inverse = {int(value): key for key, value in tokenizer.get_vocab().items()}
    result = []
    for i, plan in enumerate(plans):
        raw_tokens = generated[i]["record"].get("body_token_ids")
        options = []
        if raw_tokens:
            options.append((raw_tokens, scores["G"][i], "G"))
            if current[i]["token_ids"]:
                options.append((current[i]["token_ids"], scores["current"][i], "F"))
            options.extend(
                (
                    candidate["trace"]["proposal_tokens"],
                    scores[f"candidate_{candidate['rank']}"][i],
                    f"candidate_{candidate['rank']}",
                )
                for candidate in edited[i]["candidates"]
                if candidate["commit"]["applied"]
            )
        reliable = [option for option in options if quality(option[1]) is not None]
        if reliable:
            target, _, name = max(reliable, key=lambda option: quality(option[1]))
            target = align_tokens(target, raw_tokens)
            record = (
                copy.deepcopy(generated[i]["record"])
                if target == raw_tokens
                else make_record(plan, stage="G_teacher", body="".join(inverse[token] for token in target))
            )
            record.update(body_token_ids=target, teacher_selected_from=name)
        else:
            record = make_record(plan, stage="G_teacher", reason="no_reliable_full_structure_teacher")
        result.append(record)
    return result


def compile_sources(plans, generated, current, edited, scores, *, tokenizer, support, generator_targets=None):
    """One actor-training row per source; all reliable candidates train values."""
    from .plans import axis_schedule

    g_rows, e_rows, value_rows = [], [], []
    for i, plan in enumerate(plans):
        if plan.get("provenance", {}).get("usage_role") != "train":
            raise ValueError(f"Feedback source is not TRAIN: {plan['source_id']}")
        raw_tokens = generated[i]["record"].get("body_token_ids")
        current_tokens = current[i]["token_ids"]
        before_score = scores["current"][i]
        candidates = edited[i]["candidates"]
        positive = [
            candidate
            for candidate in candidates
            if candidate["commit"]["applied"]
            and improves(before_score, scores[f"candidate_{candidate['rank']}"][i])
        ]
        best = (
            max(
                positive,
                key=lambda candidate: (
                    *quality(scores[f"candidate_{candidate['rank']}"][i]),
                    -len(candidate["trace"]["action"]["positions"]),
                    -candidate["rank"],
                ),
            )
            if positive
            else None
        )
        common = {
            "source_id": plan["source_id"],
            "source_split": "train",
            "prompt": plan["body_prompt"],
            "num_sites": plan["plan_state"]["N"],
            "plan_state": plan["plan_state"],
            "known_sun": bool(before_score["terminal_verified"] and before_score.get("strict_sun") is True),
        }
        if current_tokens and current[i]["continuous_trace"]["editable"]:
            proposal = best or max(
                candidates, key=lambda candidate: (candidate["predicted_gain"] or [-1e6])[0], default=None
            )
            if proposal is not None and endpoint_targets(before_score) is not None:
                after = scores[f"candidate_{proposal['rank']}"][i]
                if endpoint_targets(after) is not None:
                    action = proposal["trace"].get("action", {"positions": [], "sites": []})
                    row = dict(
                        common,
                        pair_id=fingerprint([plan["source_id"], "E", proposal["rank"]]),
                        current_tokens=current_tokens,
                        proposal_tokens=proposal["trace"]["proposal_tokens"],
                        action_positions=action["positions"],
                        mode_target=action["mode"] if best else 0,
                        site_targets=[float(site in action["sites"]) for site in range(common["num_sites"])],
                        accept_target=float(best is not None),
                    )
                    if best:
                        row.update(
                            content_target_tokens=proposal["trace"]["proposal_tokens"],
                            content_positions=action["positions"],
                        )
                    e_rows.append(row)
        if raw_tokens and generator_targets is not None:
            teacher = generator_targets[i]
            target, target_score = teacher.get("body_token_ids"), scores["G_teacher"][i]
            if target and quality(target_score) is not None:
                target_name = teacher["teacher_selected_from"]
                if geometry_support_report(target, constraints=support)["supported"]:
                    row = dict(
                        common,
                        pair_id=fingerprint([plan["source_id"], "G", target_name]),
                        current_tokens=raw_tokens,
                        generation_groups=axis_schedule(plan["plan_state"])[1:],
                        teacher_record_key=target_score["record_key"],
                        teacher_exact_tokens_measured=True,
                    )
                    if (
                        target != raw_tokens
                        and improves(scores["G"][i], target_score)
                        and geometry_support_report(raw_tokens, constraints=support)["supported"]
                    ):
                        row.update(chosen_tokens=target, rejected_tokens=raw_tokens)
                    elif quality(target_score)[0] >= 2:
                        row["healthy_anchor_tokens"] = target
                    if row.get("chosen_tokens") or row.get("healthy_anchor_tokens"):
                        g_rows.append(row)
        before = endpoint_targets(before_score)
        if before is None:
            continue
        for candidate in candidates:
            after_score = scores[f"candidate_{candidate['rank']}"][i]
            after = endpoint_targets(after_score)
            if not candidate["commit"]["applied"] or after is None:
                continue
            value_rows.append(
                {
                    "source_id": plan["source_id"],
                    "ordinal": i,
                    "rank": candidate["rank"],
                    "before_targets": before,
                    "after_targets": after,
                    "before_score": before_score,
                    "current_tokens": current_tokens,
                    "proposal_tokens": candidate["trace"]["proposal_tokens"],
                    "action_positions": candidate["trace"]["action"]["positions"],
                    "prompt": plan["body_prompt"],
                    "num_sites": common["num_sites"],
                    "geometry_features": candidate["geometry_features"],
                    "source_split": "train",
                }
            )
    return g_rows, e_rows, value_rows
