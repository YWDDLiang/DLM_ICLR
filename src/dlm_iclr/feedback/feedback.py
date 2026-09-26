"""Compile offline supervision from clean TRAIN sources and measured endpoints."""

from __future__ import annotations
import math
from dlm_iclr.runtime.io import fingerprint

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


def compile_sources(plans, current, edited, scores):
    """One actor-training row per source; all reliable candidates train values."""

    e_rows, value_rows = [], []
    for i, plan in enumerate(plans):
        if plan.get("provenance", {}).get("usage_role") != "train":
            raise ValueError(f"Feedback source is not TRAIN: {plan['source_id']}")
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
    return e_rows, value_rows
