"""Validated replay of the student's stored C1 construction states.

The optional score is a typed-unary imitation surrogate at the *recorded*
committed positions. It is neither the C1 joint proposal probability nor the
probability of the confidence-projected commit, and must not be called exact
trajectory DPO. No existing training entry point imports this module.

Old periodic-axis traces contain coordinates and their real sampled lattice
context, but no lattice commit states. New opt-in all-geometry traces also
record lattice inputs. Missing lattice states are never invented from teachers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from .common import digest


class ReplayTraceError(ValueError):
    """A trace cannot be assigned the completed candidate's outcome safely."""


@dataclass(frozen=True)
class CommitView:
    source_id: str
    candidate_id: str
    prompt: str
    input_body: tuple[int, ...]
    positions: tuple[int, ...]
    tokens: tuple[int, ...]
    axis: int
    semantic_group: int
    step_in_group: int
    temperature: float
    mask_id: int

    def to_dict(self):
        return asdict(self)


def coordinate_replay_views(plan, generated, *, mask_id):
    """Read actual first-pass coordinate inputs; reject ambiguous recoveries.

    Every transition must lead exactly to the next recorded input, and the last
    transition to the saved raw body. Only committed tokens are supervised;
    unused joint candidates and eventual refined coordinates never become
    visible context or labels here. Source IDs preserve grouping by parent Plan.
    """
    return _replay_views(plan, generated, mask_id=mask_id, all_geometry=False)


def construction_replay_views(plan, generated, *, mask_id):
    """Require observed lattice AND coordinate states from the opt-in recorder."""
    return _replay_views(plan, generated, mask_id=mask_id, all_geometry=True)


def _replay_views(plan, generated, *, mask_id, all_geometry):
    record, trace = generated["record"], generated["trace"]
    candidate_id = plan["source_id"]
    if record["source_id"] != candidate_id:
        raise ReplayTraceError("Plan and raw candidate identity differ")
    if not record.get("success") or not record.get("body_token_ids"):
        raise ReplayTraceError("Completed raw candidate required")
    if trace.get("construction_recovery", {}).get("recoveries_used", 0):
        raise ReplayTraceError("Recovery credit assignment is not implemented")
    if trace.get("lattice_gamma_last", False):
        raise ReplayTraceError("Interleaved lattice construction is not supported")
    periodic = trace.get("periodic_axis", {})
    recorded = periodic.get("construction_state_replay", {})
    if all_geometry and recorded.get("schema") != "actual_all_geometry_commits_v1":
        raise ReplayTraceError("Actual lattice commit states were not recorded")
    events = recorded.get("events", []) if all_geometry else periodic.get("events", [])
    if not periodic.get("enabled") or not events:
        raise ReplayTraceError("No actual C1 coordinate events")
    prompt = plan["body_prompt"]
    if record.get("body_prompt") != prompt:
        raise ReplayTraceError("Stored raw prompt differs from Plan prompt")
    n = int(plan["plan_state"]["N"])
    final = tuple(record["body_token_ids"])
    if n < 1 or len(final) != 7 + 4*n or mask_id in final:
        raise ReplayTraceError("Completed raw token layout is invalid")
    coordinates = {8 + 4*i + a for i in range(n) for a in range(3)}
    supervised = coordinates | set(range(1, 7)) if all_geometry else coordinates
    fixed = set(range(len(final))) - supervised
    views, seen = [], set()
    expected = None
    previous_axis = -1
    for event in events:
        body = tuple(event["input_body"])
        axis = event["axis"]
        if event.get("attempt_index") != 0:
            raise ReplayTraceError("Multiple attempts cannot share endpoint credit")
        if type(axis) is not int or axis not in range(-1 if all_geometry else 0, 3) or axis < previous_axis:
            raise ReplayTraceError("Coordinate phase order changed")
        if len(body) != len(final) or any(body[p] != final[p] for p in fixed):
            raise ReplayTraceError("Actual lattice/composition context changed")
        if expected is None:
            if any(body[p] != mask_id for p in supervised):
                raise ReplayTraceError("First construction state is incomplete in the trace")
        elif body != expected:
            raise ReplayTraceError("A recorded transition does not match its next input")
        if any(body[p] not in (mask_id, final[p]) for p in supervised):
            raise ReplayTraceError("Visible geometry differs from retained raw draft")
        positions = tuple(event.get("committed_positions", []))
        tokens = tuple(event.get("committed_tokens", []))
        active = set(event["active_positions"])
        if (not positions or len(positions) != len(tokens)
                or len(set(positions)) != len(positions)
                or not set(positions) <= active
                or not active <= supervised
                or any((not 1 <= p <= 6 if axis == -1 else p < 8 or (p-8) % 4 != axis)
                       or body[p] != mask_id for p in active)):
            raise ReplayTraceError("Invalid actual commit support")
        next_body = list(body)
        candidate = event.get("candidate_body_before_projection")
        if candidate is None or len(candidate) != len(final):
            raise ReplayTraceError("Missing proposal identity for commit validation")
        for p, token in zip(positions, tokens, strict=True):
            if p in seen or token != final[p] or candidate[p] != token:
                raise ReplayTraceError("Commit token differs from proposal or final raw")
            next_body[p] = token
            seen.add(p)
        temperature = float(event["temperature"])
        if not math.isfinite(temperature) or temperature <= 0:
            raise ReplayTraceError("Invalid constructor temperature")
        views.append(CommitView(
            source_id=plan.get("parent_source_id", candidate_id),
            candidate_id=candidate_id, prompt=prompt, input_body=body,
            positions=positions, tokens=tokens, axis=axis,
            semantic_group=int(event["semantic_group"]),
            step_in_group=int(event["step_in_group"]),
            temperature=temperature, mask_id=int(mask_id),
        ))
        expected, previous_axis = tuple(next_body), axis
    if expected != final or seen != supervised:
        raise ReplayTraceError("Trace does not reconstruct the whole retained raw draft")
    return views, {
        "schema": "actual_all_geometry_commit_replay_v1" if all_geometry else "actual_coordinate_commit_replay_v1",
        "candidate_id": candidate_id,
        "source_id": plan.get("parent_source_id", candidate_id),
        "events": len(views), "coordinate_commits": len(seen & coordinates),
        "lattice_commits": len(seen - coordinates),
        "lattice_context": "actual_student_lattice_from_input_body",
        "lattice_training_ready": all_geometry,
        "trace_key": digest([v.to_dict() for v in views]),
        "teacher_visible_values_used": False,
        "exact_C1_transition_likelihood": False,
    }


def forward_replay_scores(model, tokenizer, vocab, views, *, max_length):
    """One batched differentiable forward on unchanged recorded input bodies.

    Return mean typed-unary log score and vectors for reference KL. This support
    omits C1 edge energies, dynamic geometry masks and confidence selection.
    Thus it is explicitly an imitation proxy, even when the state is exact.
    Positions and token labels are fixed recorded actions, not chosen anew.
    """
    if not views:
        raise ValueError("Empty state-replay microbatch")
    if tokenizer.pad_token_id is None:
        raise ValueError("A real tokenizer padding ID is required")
    device = next(model.parameters()).device
    prepared = []
    for view in views:
        prefix = tokenizer(view.prompt, add_special_tokens=False)["input_ids"]
        if len(prefix) + len(view.input_body) > max_length:
            raise ValueError("Truncating the recorded construction state is forbidden")
        if (not view.positions or len(view.positions) != len(view.tokens)
                or len(set(view.positions)) != len(view.positions)
                or view.axis not in range(-1, 3)
                or any((not 1 <= p <= 6 if view.axis == -1 else p < 8 or p >= len(view.input_body) or (p-8) % 4 != view.axis)
                       or view.input_body[p] != view.mask_id for p in view.positions)):
            raise ValueError("State-replay target is not an observed masked geometry commit")
        prepared.append((list(prefix) + list(view.input_body), len(prefix)))
    width = max(len(tokens) for tokens, _ in prepared)
    ids = torch.full((len(views), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    for i, (tokens, _) in enumerate(prepared):
        ids[i, :len(tokens)] = torch.tensor(tokens, device=device)
        attention[i, :len(tokens)] = 1
    result = model(ids, attention_mask=attention)
    scores = []
    for i, (view, (_, offset)) in enumerate(zip(views, prepared, strict=True)):
        terms, vectors = [], []
        for position, token in zip(view.positions, view.tokens, strict=True):
            lp, legal = vocab.log_probs(result.logits[i, offset+position], position, view.temperature)
            family = vocab.family(position)
            # Canonicalize the target alias only; never rewrite the observed input.
            if family in "XYZ" and token == vocab.tables[family].get(100):
                token = vocab.tables[family][0]
            if token not in legal:
                raise ValueError("Commit token is outside typed output support")
            terms.append(lp[legal.index(token)])
            vectors.append(lp)
        scores.append((torch.stack(terms).mean(), vectors))
    return scores
