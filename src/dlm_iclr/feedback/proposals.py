"""Autonomous conditional proposals with per-request random streams and budgets."""

from __future__ import annotations
import torch
from dlm_iclr._core.expert_edit import inference_view, materialize_edit_batch
from dlm_iclr._core.fixed_slot import MASK_TOKEN_ID
from dlm_iclr._core.geometry_constraints import geometry_support_report
from dlm_iclr._core.geometry_risk import legal_vector

MODES = ("none", "local_xyz", "all_xyz", "full_cell")
COUNTS = (1, 2, 4, 8)
STREAMS = (
    "E",
    "keep_edit_focus_repeat1",
    "keep_edit_focus_repeat2",
    "keep_edit_operational_fresh1",
    "keep_edit_scope_rank4",
    "keep_edit_scope_rank5",
    "keep_edit_scope_rank6",
    "keep_edit_scope_rank7",
)


def action_positions(n, mode, sites):
    return (list(range(1, 7)) if mode == 3 else []) + [
        8 + 4 * site + axis for site in sites for axis in range(3)
    ]


@torch.no_grad()
def propose_batch(model, tokenizer, requests, *, support, batch_size=64, temperature=0.7):
    """Generate proposals; the learned value model makes the final KEEP decision.

    Input records contain only prompt, token body, N, seed, candidate rank and
    remaining calls. No measured physical outcome is accepted by this API.
    """
    device = next(model.parameters()).device
    results = []
    for start in range(0, len(requests), batch_size):
        states = []
        for request in requests[start : start + batch_size]:
            before = list(request["body"])
            states.append(
                {
                    "request": request,
                    "before": before,
                    "current": before.copy(),
                    "order": [],
                    "offset": 0,
                    "stage": "inspect",
                    "prefix": tokenizer(request["prompt"], add_special_tokens=False)["input_ids"],
                    "generator": torch.Generator(device=device).manual_seed(request["seed"]),
                    "result": {
                        "proposal_generated": False,
                        "proposal_tokens": before,
                        "forward_calls": 0,
                        "sampling_trace": [],
                        "sampling_seed": request["seed"],
                        "sampling_temperature": temperature,
                    },
                }
            )
        while any(state["stage"] != "done" for state in states):
            active = [state for state in states if state["stage"] != "done"]
            views = [
                inference_view(
                    state["prefix"],
                    state["before"],
                    state["current"],
                    state["request"]["n"],
                    1,
                    state["order"],
                    remaining=80,
                    reveal=state["offset"] / len(state["order"]) if state["order"] else 0.0,
                )
                for state in active
            ]
            batch = materialize_edit_batch(views, tokenizer, device)
            out = model(
                batch["input_ids"], attention_mask=batch["attention_mask"], edit_context=batch["edit_context"]
            )
            for index, state in enumerate(active):
                request, result, n = state["request"], state["result"], state["request"]["n"]
                result["forward_calls"] += 1
                if state["stage"] == "inspect":
                    logits = out.mode_logits[index].float()
                    learned_mode = int(logits.argmax())
                    mode = learned_mode or (int(logits[1:].argmax()) + 1)
                    rank = request.get("candidate_rank", 0)
                    if n == 1:
                        # Moving the sole atom is a rigid translation. A cell proposal
                        # is the meaningful action for this representation.
                        mode, sites = 3, [0]
                    elif rank:
                        mode = 1
                        ranked = (
                            out.site_logits[index, :n].float().argsort(descending=True, stable=True).tolist()
                        )
                        sites = [ranked[rank % n]]
                    elif mode == 1:
                        allowed = [j for j, count in enumerate(COUNTS) if count <= n]
                        count = COUNTS[allowed[int(out.count_logits[index, allowed].argmax())]]
                        sites = sorted(out.site_logits[index, :n].topk(count).indices.tolist())
                    else:
                        sites = list(range(n))
                    order = action_positions(n, mode, sites)
                    result.update(
                        learned_mode=learned_mode,
                        mode_logits=logits.tolist(),
                        site_logits=out.site_logits[index, :n].float().tolist(),
                        count_logits=out.count_logits[index].float().tolist(),
                        action={"mode": mode, "name": MODES[mode], "sites": sites, "positions": order},
                    )
                    if len(order) + 2 > request["available_calls"]:
                        result["proposal_failure"] = "insufficient_calls_for_complete_proposal"
                        state["stage"] = "done"
                        continue
                    state["order"] = order
                    for position in order:
                        state["current"][position] = MASK_TOKEN_ID
                    state["stage"] = "fill"
                elif state["stage"] == "fill":
                    position = state["order"][state["offset"]]
                    vector, report = legal_vector(
                        out.logits[index, len(state["prefix"]) + position].float(),
                        state["current"],
                        n,
                        position,
                        support,
                    )
                    if not report["available"]:
                        result["proposal_failure"] = "empty_geometry_support"
                        state["stage"] = "done"
                        continue
                    probability = (vector / temperature).softmax(-1)
                    token = int(torch.multinomial(probability, 1, generator=state["generator"]))
                    state["current"][position] = token
                    result["sampling_trace"].append(
                        {
                            "position": position,
                            "token_id": token,
                            "scalar_logp": float(probability[token].log()),
                        }
                    )
                    state["offset"] += 1
                    if state["offset"] == len(state["order"]):
                        report = geometry_support_report(state["current"], constraints=support)
                        if report["supported"]:
                            state["stage"] = "judge"
                        else:
                            result["proposal_failure"] = report
                            state["stage"] = "done"
                else:
                    result.update(
                        proposal_generated=True,
                        proposal_tokens=state["current"].copy(),
                        learned_accept_probability=float(out.quality_logits[index, 3].sigmoid()),
                    )
                    state["stage"] = "done"
            del out
        results.extend(state["result"] for state in states)
    return results
