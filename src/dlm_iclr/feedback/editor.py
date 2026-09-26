"""Continuous KEEP/EDIT using learned candidate-minus-current values."""

from __future__ import annotations
import copy
import torch
from dlm_iclr._core.continuous_keep_edit import commit_patch
from dlm_iclr._core.post_refine_contract import derived_seed
from dlm_iclr._core.geometry_constraints import build_repair_constraints
from dlm_iclr.feedback.proposals import propose_batch, STREAMS
from dlm_iclr.feedback.value import choose, extract_features, geometry_features


class Editor:
    def __init__(self, model, tokenizer, value_model, policy):
        self.model, self.tokenizer, self.value_model, self.policy = model, tokenizer, value_model, policy
        self.device = next(model.parameters()).device
        self.support = build_repair_constraints(tokenizer)
        self.inverse = {int(value): token for token, value in tokenizer.get_vocab().items()}

    @torch.no_grad()
    def edit(self, plans, current, *, retain_features=False):
        results = [
            {
                "source_id": plan["source_id"],
                "ordinal": plan["ordinal"],
                "record": copy.deepcopy(wrapper["record"]),
                "selected_candidate": None,
                "forward_calls": 0,
                "candidates": [],
                "input_features": "model_and_geometry_only",
            }
            for plan, wrapper in zip(plans, current, strict=True)
        ]
        eligible = [
            i
            for i, wrapper in enumerate(current)
            if wrapper["record"]["success"]
            and wrapper["continuous_trace"]["editable"]
            and wrapper["token_ids"]
        ]
        keep_rows = [
            {
                "prompt": plans[i]["body_prompt"],
                "num_sites": plans[i]["plan_state"]["N"],
                "current_tokens": current[i]["token_ids"],
                "proposal_tokens": current[i]["token_ids"],
                "action_positions": [],
            }
            for i in eligible
        ]
        features = extract_features(self.model, self.tokenizer, keep_rows, self.device)
        keep_features, keep_values = {}, {}
        for index, i in enumerate(eligible):
            geometry = geometry_features(
                current[i]["record"], current[i]["record"], [], plans[i]["plan_state"]["N"]
            )
            keep_features[i] = features[index]
            keep_values[i] = self.value_model(
                features[index : index + 1], torch.tensor([geometry], dtype=torch.float32)
            )[0]
            results[i]["forward_calls"] += 1
        training_features = {"keep": keep_features, "candidates": {}}
        for rank in range(self.policy.editor_candidates):
            # Reserve the scoring view for any completed proposal. KEEP has
            # already been evaluated once, so actual calls stay within budget.
            active = [
                i
                for i in eligible
                if results[i]["forward_calls"] + (12 if plans[i]["plan_state"]["N"] == 1 else 6)
                <= self.policy.editor_max_calls
            ]
            requests = [
                {
                    "prompt": plans[i]["body_prompt"],
                    "body": current[i]["token_ids"],
                    "n": plans[i]["plan_state"]["N"],
                    "seed": derived_seed(str(plans[i]["body_noise_seed"]), STREAMS[rank]),
                    "candidate_rank": rank,
                    "available_calls": self.policy.editor_max_calls - results[i]["forward_calls"] - 1,
                }
                for i in active
            ]
            traces = propose_batch(
                self.model,
                self.tokenizer,
                requests,
                support=self.support,
                batch_size=self.policy.editor_batch_size,
                temperature=self.policy.temperature,
            )
            rows, row_indices = [], []
            for i, trace in zip(active, traces, strict=True):
                results[i]["forward_calls"] += trace["forward_calls"]
                native = current[i]["record"]
                proposed = trace["proposal_tokens"]
                record, binding = commit_patch(
                    native,
                    current[i]["token_ids"],
                    proposed,
                    self.inverse,
                    editable=trace["proposal_generated"],
                )
                positions = trace.get("action", {}).get("positions", [])
                candidate = {
                    "rank": rank,
                    "record": record,
                    "trace": trace,
                    "commit": binding,
                    "predicted_gain": None,
                    "geometry_features": geometry_features(
                        native, record, positions, plans[i]["plan_state"]["N"]
                    ),
                }
                results[i]["candidates"].append(candidate)
                if trace["proposal_generated"]:
                    rows.append(
                        {
                            "prompt": plans[i]["body_prompt"],
                            "num_sites": plans[i]["plan_state"]["N"],
                            "current_tokens": current[i]["token_ids"],
                            "proposal_tokens": proposed,
                            "action_positions": positions,
                        }
                    )
                    row_indices.append(i)
            candidate_features = extract_features(self.model, self.tokenizer, rows, self.device)
            for index, i in enumerate(row_indices):
                candidate = results[i]["candidates"][-1]
                prediction = (
                    self.value_model(
                        candidate_features[index : index + 1],
                        torch.tensor([candidate["geometry_features"]], dtype=torch.float32),
                    )[0]
                    - keep_values[i]
                )
                candidate["predicted_gain"] = prediction.tolist()
                results[i]["forward_calls"] += 1
                if retain_features:
                    training_features["candidates"][(i, rank)] = candidate_features[index]
        for i in eligible:
            result = results[i]
            selected = choose(
                [
                    candidate["predicted_gain"] if candidate["commit"]["applied"] else None
                    for candidate in result["candidates"]
                ],
                sun_weight=self.policy.sun_weight,
                msun_weight=self.policy.msun_weight,
            )
            if selected is not None:
                result["selected_candidate"] = result["candidates"][selected]["rank"]
                result["record"] = copy.deepcopy(result["candidates"][selected]["record"])
            result["record"]["stage"] = "E"
            if result["forward_calls"] > self.policy.editor_max_calls:
                raise RuntimeError("Per-request model forward accounting exceeded the configured budget")
        return results, training_features if retain_features else None
