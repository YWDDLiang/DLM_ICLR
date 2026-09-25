"""Fixed-Plan generation using the retained R03 schedule and paired noise."""

from __future__ import annotations
from collections import Counter
from functools import partial
import torch
from dlm_iclr._core import paired_llada, r03_geometry_bridge
from dlm_iclr._core.construction_recovery import construct_cascade
from dlm_iclr._core.dynamic_crystal import parse_dynamic_answer, arrays_to_structure
from dlm_iclr._core.fixed_slot import FixedSlotConfig, MASK_TOKEN_ID, Z_TO_SYMBOL
from dlm_iclr._core.r03_physics_transfer import build_repair_constraints, geometry_support_report
from dlm_iclr.runtime.models import build_dynamic_lightweight_constraints
from dlm_iclr.data.plans import axis_schedule


def schema_and_prefill(tokenizer, plan):
    vocab, config = tokenizer.get_vocab(), FixedSlotConfig()

    def ids(prefix, lower, upper):
        return [int(vocab[f"<{prefix}_{index:03d}>"]) for index in range(lower, upper + 1)]

    schema = [[int(vocab[f"<N_{plan['N']:03d}>"])]]
    schema += [ids(prefix, config.length_min_bin, config.length_max_bin) for prefix in ("LA", "LB", "LC")]
    schema += [ids(prefix, config.angle_min_bin, config.angle_max_bin) for prefix in ("AA", "AB", "AG")]
    elements = [int(vocab[f"<E_{Z_TO_SYMBOL[z]}>"]) for z in range(1, config.max_atomic_number + 1)]
    coordinates = [ids(axis, config.coord_min_bin, config.coord_max_bin) for axis in "XYZ"]
    for _ in range(plan["N"]):
        schema.extend([elements, *coordinates])
    prefill, site = {0: schema[0]}, 0
    for element, count in zip(plan["elements"], plan["counts"], strict=True):
        for _ in range(count):
            prefill[7 + 4 * site] = [int(vocab[f"<E_{element}>"])]
            site += 1
    return schema, prefill


def construct(
    model,
    tokenizer,
    batch,
    runtime=None,
    *,
    constraints,
    geometry_api=None,
    initial_body=None,
    noise_seed_override=None,
    relax_final_z=False,
    lattice_gamma_last=False,
    temperature=0.7,
    axis_sampler=None,
    geometry_monitor=True,
):
    if len(batch) != 1:
        raise ValueError("Use independent worker processes for construction batch size greater than one")
    task = batch[0]
    n, schedule = task["plan_state"]["N"], task["schedule"]
    if lattice_gamma_last:
        schedule = [
            part
            for group in schedule
            for part in (
                [[p for p in group if p != 6], [6]] if 6 in group and len(group) > 1 else [list(group)]
            )
        ]
    encoded = tokenizer([task["body_prompt"]], add_special_tokens=False, padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    prompt, attention = encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
    schema, prefill = schema_and_prefill(tokenizer, task["plan_state"])
    if initial_body is not None:
        if len(initial_body) != 7 + 4 * n:
            raise ValueError("Recovery canvas length differs from the Plan")
        if any(initial_body[position] != tokens[0] for position, tokens in prefill.items()):
            raise ValueError("Recovery changed fixed composition tokens")
        prefill.update(
            {position: [int(value)] for position, value in enumerate(initial_body) if value != MASK_TOKEN_ID}
        )
    monitor = r03_geometry_bridge.ConstructionGeometryMonitor(
        enabled=geometry_monitor,
        tokenizer=tokenizer,
        generation_position_groups=schedule,
        native_constraints=constraints,
        mask_id=MASK_TOKEN_ID,
        relax_final_z=relax_final_z,
    )
    optional = {}
    if axis_sampler is not None:
        axis_sampler.begin_attempt(
            schedule, task["body_noise_seed"] if noise_seed_override is None else noise_seed_override
        )
        if axis_sampler.enabled:
            from dlm_iclr.c1.sampling import axis_predictor

            optional = {"predictor": axis_predictor, "candidate_sampler": axis_sampler}
    generated = paired_llada.generate_paired_exact_plan(
        model,
        prompt,
        base_seeds=[task["body_noise_seed"] if noise_seed_override is None else noise_seed_override],
        attention_mask=attention,
        gen_length=7 + 4 * n,
        temperature=temperature,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=MASK_TOKEN_ID,
        allowed_token_ids_by_generation_pos=schema,
        prefill_token_ids_by_generation_pos=prefill,
        generation_position_groups=schedule,
        lightweight_decoding_constraints=constraints,
        candidate_hook=monitor.apply_logits if geometry_monitor else None,
        **optional,
    )
    suffix = generated[:, prompt.shape[1] :]
    if any(suffix[:, position].cpu().tolist() != tokens for position, tokens in prefill.items()):
        raise RuntimeError("Generation changed a fixed composition or retained recovery token")
    return suffix.cpu(), {
        "effective_generation_schedule": schedule,
        "construction_geometry": monitor.report(),
        "lattice_gamma_last": lattice_gamma_last,
    }


def make_record(plan, *, stage, structure=None, body=None, reason=None):
    return {
        "source_id": plan["source_id"],
        "ordinal": plan["ordinal"],
        "original_ordinal": plan["original_ordinal"],
        "stage": stage,
        "success": structure is not None or bool(body),
        "structure": structure,
        "body": body,
        "declared_composition": dict(zip(plan["plan_state"]["elements"], plan["plan_state"]["counts"]))
        if plan.get("plan_state")
        else None,
        "reason": reason,
    }


class Constructor:
    def __init__(self, model, tokenizer, policy, *, axis_head=None, axis_parents=None, axis_neutral=False):
        self.model, self.tokenizer, self.policy = model, tokenizer, policy
        self.axis_head, self.axis_parents, self.axis_neutral = axis_head, axis_parents, axis_neutral
        self.constraints = build_dynamic_lightweight_constraints(
            tokenizer, duplicate_coordinate_mask=True, lattice_volume_mask=True, min_lattice_rad=1e-4
        )
        self.support = build_repair_constraints(tokenizer)

    @torch.no_grad()
    def generate(self, plan):
        if not plan["body_eligible"]:
            return {
                "record": make_record(plan, stage="G", reason=plan["ineligible_reason"]),
                "trace": {"forward_calls": 0},
                "graph": None,
            }
        task = dict(plan, schedule=axis_schedule(plan["plan_state"]))
        axis_sampler = None
        if self.axis_head is not None:
            from dlm_iclr.c1.sampling import AxisCandidateSampler

            axis_sampler = AxisCandidateSampler(
                self.axis_head,
                self.tokenizer,
                plan,
                self.constraints,
                parents=self.axis_parents,
                neutral=self.axis_neutral,
            )
        calls = [0]

        def count_forward(_model, _inputs):
            calls[0] += 1

        hook = self.model.register_forward_pre_hook(count_forward)
        graph = None
        try:
            sampler = partial(construct, temperature=self.policy.temperature, axis_sampler=axis_sampler,
                              geometry_monitor=self.policy.geometry_monitor)
            if self.policy.construction_recovery:
                suffix, trace = construct_cascade(
                    self.model,
                    self.tokenizer,
                    task,
                    None,
                    construct=sampler,
                    constraints=self.constraints,
                    repair_constraints=self.support,
                    geometry_api=r03_geometry_bridge,
                    complete_geometry=lambda body: geometry_support_report(body, constraints=self.support),
                    adaptive_lattice=self.policy.adaptive_lattice_recovery,
                )
            else:
                suffix, trace = sampler(self.model, self.tokenizer, [task], constraints=self.constraints)
            ids = suffix[0].tolist()
            body = self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            arrays = parse_dynamic_answer(body, strict=True)
            if Counter(arrays["species"]) != Counter(
                dict(zip(plan["plan_state"]["elements"], plan["plan_state"]["counts"]))
            ):
                raise ValueError("Generated body composition differs from the Plan")
            record = make_record(plan, stage="G", body=body)
            record.update(body_token_ids=ids, body_prompt=plan["body_prompt"])
            from dlm_iclr.diffusion.refinement import graph_from_arrays

            try:
                graph, _ = graph_from_arrays(arrays)
                graph["sample_idx"] = plan["original_ordinal"]
            except (ValueError, FloatingPointError, RuntimeError) as error:
                trace["graph_failure"] = str(error)
        except r03_geometry_bridge.GeometryNoLegalSupport as error:
            record = make_record(plan, stage="G", reason="construction_geometry_exhausted")
            trace = {"construction_failure": error.to_dict()}
        except (ValueError, FloatingPointError) as error:
            record = make_record(plan, stage="G", reason=str(error))
            trace = {"construction_failure": type(error).__name__}
        finally:
            hook.remove()
        trace["forward_calls"] = calls[0]
        if axis_sampler is not None:
            trace["periodic_axis"] = axis_sampler.report()
        return {"record": record, "trace": trace, "graph": graph}
