"""Periodic-axis candidates inside the retained per-commit DLM constructor.

No extra DLM forward, new geometry veto, future-token commit, or F invocation.
Geometry support/recovery and confidence commit remain the caller's legacy
rules. Joint candidate log probability is NOT the probability of the projected
confidence-selected transition (uncommitted candidates are auxiliary draws).
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import torch

from dlm_iclr._core.fixed_slot import SYMBOL_TO_Z
from dlm_iclr._core.llada_generation import _model_logits, _lattice_matrix_from_token_ids
from dlm_iclr._core.paired_llada import _paired_suffix_candidates
from dlm_iclr._core.paired_noise import derive_subseed
from dlm_iclr.c1.distribution import PeriodicAxisLaw, collapse_alias_logits, registered_parents


def _axis(group):
    if not group or any(p < 8 or (p - 8) % 4 > 2 for p in group):
        return None
    axes = {(p - 8) % 4 for p in group}
    if len(axes) != 1:
        raise ValueError("Axis candidates require the original single-axis groups")
    return axes.pop()


def axis_predictor(model, x, attention_mask, prompt_index, cfg_scale, mask_id, *, group_positions):
    """llada_generation._model_logits mask ABI, with hidden from SAME call.

    The original helper forwards x and attention_mask without additional masks
    or position IDs for cfg=0. Hidden extraction preserves these arguments.
    Lattice groups retain its exact original call. CFG hidden mixing is not
    defined for this adaptation and is explicitly unsupported.
    """
    if _axis(group_positions) is None:
        return _model_logits(model, x, attention_mask, prompt_index, cfg_scale, mask_id), None
    if cfg_scale != 0:
        raise ValueError("Periodic-axis predictor requires original cfg_scale=0")
    result = model(x, attention_mask=attention_mask, output_hidden_states=True)
    if not result.hidden_states or result.hidden_states[-1].shape[:2] != x.shape:
        raise ValueError("DLM must return aligned final hidden states from the same forward")
    return result.logits, result.hidden_states[-1]


def neutral_head(head):
    """A zero output affine layer gives identically zero finite-head energy."""
    if not hasattr(head, "output"):
        raise ValueError("Expected PeriodicAxisHead output layer")
    if any(not bool(torch.isfinite(p).all()) for p in head.parameters()):
        raise ValueError("Nonfinite axis head")
    return bool((head.output.weight == 0).all()) and bool((head.output.bias == 0).all())


class AxisCandidateSampler:
    def __init__(self, head, tokenizer, plan, constraints, *, parents=None, neutral=False, controller=None,
                 confidence_policy='legacy_unary'):
        if head.q != 100:
            raise ValueError("Live crystal axis constructor requires Q=100")
        if type(neutral) is not bool:
            raise ValueError("neutral must be explicit boolean")
        self.head, self.plan, self.constraints = head, deepcopy(plan), constraints
        self.controller = controller
        if confidence_policy not in ('legacy_unary','joint_marginal'):
            raise ValueError('Unknown axis confidence policy')
        self.confidence_policy=confidence_policy
        self.n = int(plan["plan_state"]["N"])
        self.parents = registered_parents(self.n, parents)
        self.enabled = not (neutral or neutral_head(head))
        self.neutral_reason = (
            "explicit_bypass" if neutral else "zero_output_energy" if not self.enabled else None
        )
        self.events = []
        self.attempts = []
        self.pending = None
        vocab = tokenizer.get_vocab()
        self.axis_tokens = [[int(vocab[f"<{axis}_{k:03d}>"]) for k in range(101)] for axis in "XYZ"]
        self.coord_map = [{token: k for k, token in enumerate(ids)} for ids in self.axis_tokens]
        self.element_map = {
            int(vocab[f"<E_{symbol}>"]): z for symbol, z in SYMBOL_TO_Z.items() if f"<E_{symbol}>" in vocab
        }

    def begin_attempt(self, schedule, seed):
        self.pending = None
        self.attempts.append(
            {
                "attempt_index": len(self.attempts),
                "base_seed": int(seed),
                "schedule": [list(g) for g in schedule],
            }
        )

    def __call__(self, logits, *, hidden, group_positions, mask_id, **arguments):
        self.pending = None
        axis = _axis(group_positions)
        # Same legacy candidate generation for all non-axis values. Its random
        # streams are stateless; neutral construction never calls this wrapper.
        candidates, confidence = _paired_suffix_candidates(logits, **arguments)
        if not self.enabled or axis is None:
            return candidates, confidence
        if arguments["temperature"] <= 0:
            raise ValueError("Nonneutral joint law requires positive temperature")
        if arguments["remasking"] != "low_confidence":
            raise ValueError("Register the original low_confidence commit policy")
        x = arguments["current_tokens"]
        offset = arguments["prompt_length"]
        length = arguments["gen_length"]
        if x.shape[0] != 1 or length != 7 + 4 * self.n:
            raise ValueError("Axis constructor requires one full original crystal")
        if hidden is None or hidden.shape[:2] != x.shape:
            raise ValueError("Missing same-forward hidden states")
        body = x[0, offset:]
        device = logits.device
        positions = torch.tensor([8 + 4 * i + axis for i in range(self.n)], device=device)
        visible = body[positions] != mask_id
        coords = torch.full((self.n, 3), torch.nan, device=device, dtype=torch.float64)
        known = torch.zeros_like(coords, dtype=torch.bool)
        for site in range(self.n):
            for a in range(3):
                token = int(body[8 + 4 * site + a])
                if token != mask_id:
                    if token not in self.coord_map[a]:
                        raise ValueError("Visible coordinate token ABI changed")
                    coords[site, a] = (self.coord_map[a][token] % 100) / 100.0
                    known[site, a] = True
        values = torch.tensor(
            [self.coord_map[axis][int(body[p])] % 100 if bool(v) else 0 for p, v in zip(positions, visible)],
            device=device,
        )
        tokens = torch.tensor(self.axis_tokens[axis], device=device)
        token_logits = logits[0, offset + positions][:, tokens]
        # Legacy geometry/schema masks use finfo.min, not -inf. Preserve their
        # excluded support explicitly before exact log-domain inference.
        token_logits = torch.where(
            token_logits == torch.finfo(logits.dtype).min,
            torch.full_like(token_logits, -torch.inf),
            token_logits,
        )
        unary = collapse_alias_logits(token_logits, list(range(101)), q=100)
        # Visible-site unary constants cancel after conditioning, and must not
        # be replaced by a newly predicted token or vetoed by duplicate masks.
        unary = torch.where(visible[:, None], torch.zeros_like(unary), unary)
        lattice = _lattice_matrix_from_token_ids(x[0], prompt_length=offset, constraints=self.constraints)
        if lattice is None:
            raise ValueError("Original coordinate phase has no valid lattice context")
        elements = torch.tensor(
            [self.element_map[int(body[7 + 4 * i])] for i in range(self.n)], device=device
        )
        site_hidden = hidden[0, offset + positions]
        edge = self.head(site_hidden, elements, lattice, coords, known, axis=axis, parents=self.parents)
        law = PeriodicAxisLaw(
            unary,
            edge,
            parents=self.parents,
            temperature=arguments["temperature"],
            visible_values=values,
            visible_mask=visible,
        )
        group = arguments["semantic_group"]
        step = arguments["step_in_group"]
        base = arguments["base_seeds"][0]
        seed = derive_subseed(base, "periodic_axis_joint_candidates_v1", group, step)
        sample = law.sample(1, generator=torch.Generator(device=device).manual_seed(seed))[0]
        # Only the current schedule group participates in the legacy top-k.
        # Other sampled axes/sites remain hypothetical and are only logged.
        active = [int(p) for p in group_positions if int(body[p]) == mask_id]
        if self.controller is not None:
            sample = self.controller.select(
                self, law, sample, logits, hidden, body, active, axis, arguments
            )
        joint_confidence=None
        if self.confidence_policy=='joint_marginal':
            from .feedback import sampled_value_log_confidence
            joint_confidence=sampled_value_log_confidence(law,sample)
            confidence=confidence.float()
        for p in positions.tolist():
            if int(body[p]) != mask_id:
                continue
            site = (p - 8) // 4
            token = self.axis_tokens[axis][int(sample[site])]
            candidates[0, offset + p] = token
            if p in active:
                confidence[0, offset + p] = (joint_confidence[site].float() if joint_confidence is not None
                    else torch.softmax(logits[0, offset + p], -1)[token])
        event = {
            "attempt_index": len(self.attempts) - 1,
            "semantic_group": group,
            "step_in_group": step,
            "axis": axis,
            "parents": list(self.parents),
            "base_seed": int(base),
            "joint_sample_seed": seed,
            "input_body": body.tolist(),
            "group_positions": list(group_positions),
            "active_positions": active,
            "visible_axis_mask": visible.tolist(),
            "joint_candidate_bins": sample.tolist(),
            "candidate_joint_log_probability": float(law.log_prob(sample)),
            "conditional_log_partition": float(law.log_partition),
            "temperature": arguments["temperature"],
            "temperature_policy": "whole_unary_and_edge_after_alias_merge",
            "alias_policy": "000_100_logaddexp_to_000_before_temperature",
            "hidden_dtype": str(hidden.dtype),
            "logits_dtype": str(logits.dtype),
            "confidence_source": ("joint_log_marginal_under_current_masked_periodic_law" if joint_confidence is not None
                                  else "legacy_untempered_unary_at_joint_candidate_not_calibrated_joint_confidence"),
            "joint_sample_is_auxiliary_not_projected_transition_log_probability": True,
            "new_geometry_veto": False,
            "extra_DLM_calls": 0,
        }
        self.events.append(event)
        self.pending = event
        return candidates, confidence

    def record_commit(self, *, current_tokens, candidates, transfer_index, prompt_length, **kwargs):
        if self.pending is None:
            return
        selected = transfer_index[0].nonzero(as_tuple=False).flatten().tolist()
        positions = [p - prompt_length for p in selected]
        if not set(positions) <= set(self.pending["active_positions"]):
            raise RuntimeError("Commit escaped original active group")
        self.pending["committed_positions"] = positions
        self.pending["committed_tokens"] = [int(candidates[0, p]) for p in selected]
        self.pending["candidate_body_before_projection"] = candidates[0, prompt_length:].tolist()
        if self.controller is not None:
            self.controller.record_commit(self.pending)
        self.pending = None

    def report(self):
        return {
            "schema": "periodic_axis_constructor_v1",
            "enabled": self.enabled,
            "neutral_reason": self.neutral_reason,
            "head_config": self.head.config(),
            "parents": list(self.parents),
            "same_legacy_commit_rule": self.confidence_policy=='legacy_unary',
            "confidence_policy": self.confidence_policy,
            "same_commit_counts_and_schedule": True,
            "DLM_reforward_each_commit": True,
            "extra_DLM_calls": 0,
            "neutral_is_same_asset_sampler_not_historical_S0_identity": True,
            "attempts": deepcopy(self.attempts),
            "events": deepcopy(self.events),
        }
