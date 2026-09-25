"""One local conditional revision using the unchanged original crystal editor."""

from dlm_iclr.runtime.capacity import MAX_ATOMS

from copy import deepcopy
import numpy as np
import torch

from dlm_iclr.c2.risk import structure, scalar_feature_grid


def tilted_probabilities(probability, risk, strength, kl_budget):
    p = np.asarray(probability, dtype=float)
    p = p / p.sum()
    r = np.asarray(risk, dtype=float)
    support = p > 0

    def at(lam):
        log = np.log(p[support]) - lam * r[support]
        w = np.exp(log - log.max())
        w /= w.sum()
        q = np.zeros_like(p)
        q[support] = w
        kl = float(np.sum(w * (np.log(w) - np.log(p[support]))))
        return q, kl

    q, kl = at(strength)
    lam = strength
    if kl > kl_budget:
        lo = 0.0
        hi = strength
        for _ in range(24):
            mid = (lo + hi) / 2
            qm, km = at(mid)
            if km <= kl_budget:
                lo = mid
            else:
                hi = mid
        lam = lo
        q, kl = at(lam)
    return q, {"strength": float(lam), "KL": kl, "reference_risk": float(p @ r), "guided_risk": float(q @ r)}


def selected_candidate(bundle):
    rank = bundle["E"]["selected_candidate"]
    return next((c for c in bundle["E"]["candidates"] if c["rank"] == rank), None)


def choose_result(bundle, penalty, minimum_risk_reduction=0.0):
    """Select among KEEP, original winner and one measured-input repair proposal."""
    trace = bundle["onepass"]
    old = bundle["E_baseline"]
    new = deepcopy(old)
    extra = trace.get("extra_DLM_calls", 0)
    new["forward_calls"] = old["forward_calls"] + extra
    if not trace.get("eligible") or trace.get("expected_risk_reduction", 0.0) < minimum_risk_reduction:
        bundle["E"] = new
        return "baseline"
    options = [("current", None, (0.0, 0.0, True))]
    for name, c in [("baseline", selected_candidate({"E": old})), ("repair", bundle.get("repair_candidate"))]:
        if c is None or not c["commit"]["applied"]:
            continue
        gain = c["predicted_gain"]
        risk = trace["risk_" + name]
        utility = 2 * gain[0] + gain[1] - penalty * (risk - trace["risk_keep"])
        options.append((name, c, (float(utility), float(gain[0]), False)))
    name, c, _ = max(options, key=lambda x: x[2])
    if bundle.get("repair_candidate") is not None:
        new["candidates"].append(deepcopy(bundle["repair_candidate"]))
    if c is None:
        new["record"] = deepcopy(bundle["F"]["record"])
        new["record"]["stage"] = "E"
        new["selected_candidate"] = None
    else:
        new["record"] = deepcopy(c["record"])
        new["record"]["stage"] = "E"
        new["selected_candidate"] = c["rank"]
    new["onepass_selection"] = {"winner": name, "risk_penalty": penalty}
    bundle["E"] = new
    return name


class OnePassRepair:
    def __init__(self, model, tokenizer, value, risk, settings):
        from dlm_iclr._core.r03_physics_transfer import build_repair_constraints

        self.model = model
        self.tokenizer = tokenizer
        self.value = value
        self.risk = risk
        self.settings = settings
        self.support = build_repair_constraints(tokenizer)
        self.inverse = {int(v): k for k, v in tokenizer.get_vocab().items()}
        self.device = next(model.parameters()).device

    @torch.inference_mode()
    def repair(self, bundles, *, keep_features=None):
        from dlm_iclr._core.expert_edit import inference_view, materialize_edit_batch
        from dlm_iclr._core.fixed_slot import MASK_TOKEN_ID
        from dlm_iclr._core.rsi_preference import legal_vector
        from dlm_iclr._core.post_refine_contract import derived_seed
        from dlm_iclr._core.continuous_keep_edit import commit_patch
        from dlm_iclr.c2.value import geometry_features, extract_features

        result = deepcopy(bundles)
        queries = []
        states = {}
        maxcalls = self.settings.get("editor_max_calls", 80)
        for i, b in enumerate(result):
            b["E_baseline"] = deepcopy(b["E"])
            b["repair_candidate"] = None
            meta = {"eligible": False, "extra_DLM_calls": 0}
            b["onepass"] = meta
            c = selected_candidate(b)
            if not self.settings.get("repair_enabled", True):
                meta["reason"] = "disabled"
                continue
            if c is None:
                meta["reason"] = "original_KEEP"
                continue
            action = c["trace"]["action"]
            changed = c["commit"].get("changed", [])
            sites = c["commit"].get("changed_sites", [])
            if (
                action["name"] != "local_xyz"
                or len(action["sites"]) != 1
                or len(sites) != 1
                or c["commit"].get("lattice_changed")
            ):
                meta["reason"] = "original_nonlocal_path_preserved"
                continue
            positions = [p for p in changed if p >= 8 and (p - 8) % 4 < 3][
                : self.settings.get("max_repair_fields", 3)
            ]
            missing_keep = keep_features is None or i not in keep_features
            if not positions or b["E"]["forward_calls"] + len(positions) + 1 + int(missing_keep) > maxcalls:
                meta["reason"] = "no_repair_budget_or_fields"
                continue
            native = b["F"]["record"]
            before = structure(native)
            proposal = structure(c["record"])
            meta.update(
                eligible=True,
                reason="inspecting",
                risk_keep=self.risk.score(native, native, []),
                risk_baseline=self.risk.score(native, c["record"], action["positions"]),
            )
            if meta["risk_baseline"] < self.settings.get("minimum_risk_reduction", 0.0):
                meta.update(eligible=False, reason="risk_benefit_below_intervention_cost")
                continue
            state = {
                "candidate": c,
                "before": before,
                "proposal": proposal,
                "options": [],
                "positions": positions,
                "prefix": self.tokenizer(b["plan"]["body_prompt"], add_special_tokens=False)["input_ids"],
            }
            states[i] = state
            for pos in positions:
                canvas = list(c["trace"]["proposal_tokens"])
                canvas[pos] = MASK_TOKEN_ID
                view = inference_view(
                    state["prefix"],
                    b["F"]["token_ids"],
                    canvas,
                    len(before),
                    1,
                    action["positions"],
                    remaining=80,
                    reveal=max(0, (len(action["positions"]) - 1) / len(action["positions"])),
                )
                queries.append((i, pos, canvas, view))
        width = self.settings.get("query_batch_size", 64)
        for start in range(0, len(queries), width):
            batchq = queries[start : start + width]
            batch = materialize_edit_batch([q[3] for q in batchq], self.tokenizer, self.device)
            output = self.model(
                batch["input_ids"], attention_mask=batch["attention_mask"], edit_context=batch["edit_context"]
            )
            for row, (i, pos, canvas, _) in enumerate(batchq):
                b = result[i]
                state = states[i]
                b["onepass"]["extra_DLM_calls"] += 1
                vector, legal = legal_vector(
                    output.logits[row, len(state["prefix"]) + pos].float(),
                    canvas,
                    len(state["before"]),
                    pos,
                    self.support,
                )
                if not legal["available"]:
                    continue
                axis = (pos - 8) % 4
                site = (pos - 8) // 4
                axis_name = "XYZ"[axis]
                ids = [self.support["coord_bin_to_token_id"][axis_name][k] for k in range(100)]
                logits = vector[ids]
                prob = (
                    torch.softmax(logits / self.settings.get("temperature", 0.7), -1).cpu().double().numpy()
                )
                old_bin = self.support["coord_token_to_bin"][axis_name][b["F"]["token_ids"][pos]] % 100
                values = np.arange(100, dtype=float) / 100
                values[old_bin] = state["before"].frac_coords[site, axis]
                feats = scalar_feature_grid(
                    state["before"],
                    state["proposal"],
                    state["candidate"]["trace"]["action"]["positions"],
                    site,
                    axis,
                    values,
                )
                risk = self.risk.predict(feats)
                q, diagnostic = tilted_probabilities(
                    prob, risk, self.settings.get("risk_strength", 2.0), self.settings.get("kl_budget", 0.1)
                )
                reduction = b["onepass"]["risk_baseline"] - float(q @ risk)
                state["options"].append((reduction, pos, ids, q, diagnostic))
            del output, batch
        score_rows = []
        row_kinds = []
        kept = {}
        for i, state in states.items():
            b = result[i]
            meta = b["onepass"]
            if not state["options"]:
                meta.update(eligible=False, reason="no_legal_repair")
                continue
            reduction, pos, ids, q, diagnostic = max(state["options"], key=lambda x: (x[0], -x[1]))
            meta.update(expected_risk_reduction=float(reduction), field=pos, tilt=diagnostic)
            if reduction <= self.settings.get("minimum_risk_reduction", 0.0):
                meta.update(eligible=False, reason="risk_benefit_below_intervention_cost")
                continue
            seed = derived_seed(str(b["plan"]["body_noise_seed"]), "onepass_repair_v1")
            rng = np.random.default_rng(seed)
            token = int(rng.choice(ids, p=q))
            tokens = list(state["candidate"]["trace"]["proposal_tokens"])
            tokens[pos] = token
            record, commit = commit_patch(b["F"]["record"], b["F"]["token_ids"], tokens, self.inverse)
            if tokens == state["candidate"]["trace"]["proposal_tokens"] or not commit["applied"]:
                meta["reason"] = "repair_sample_unchanged_or_not_committed"
                continue
            action = deepcopy(state["candidate"]["trace"]["action"])
            c = {
                "rank": 8,
                "record": record,
                "commit": commit,
                "predicted_gain": None,
                "trace": {
                    "proposal_generated": True,
                    "proposal_tokens": tokens,
                    "action": action,
                    "parent_rank": state["candidate"]["rank"],
                    "repair_field": pos,
                    "sampling_seed": int(seed),
                    "sampling_probability": float(q[ids.index(token)]),
                },
                "geometry_features": geometry_features(
                    b["F"]["record"], record, action["positions"], len(state["before"])
                ),
            }
            b["repair_candidate"] = c
            meta["risk_repair"] = self.risk.score(b["F"]["record"], record, action["positions"])
            common = {
                "prompt": b["plan"]["body_prompt"],
                "num_sites": len(state["before"]),
                "current_tokens": b["F"]["token_ids"],
            }
            if keep_features is not None and i in keep_features:
                kept[i] = keep_features[i]
            else:
                score_rows.append(dict(common, proposal_tokens=b["F"]["token_ids"], action_positions=[]))
                row_kinds.append((i, "keep"))
            score_rows.append(dict(common, proposal_tokens=tokens, action_positions=action["positions"]))
            row_kinds.append((i, "repair"))
            meta["reason"] = "repair_candidate_generated"
        if score_rows:
            raw = extract_features(self.model, self.tokenizer, score_rows, self.device, batch_size=width)
            new_features = {}
            for h, (i, kind) in zip(raw, row_kinds):
                result[i]["onepass"]["extra_DLM_calls"] += 1
                if kind == "keep":
                    kept[i] = h
                else:
                    new_features[i] = h
            for i, h in new_features.items():
                b = result[i]
                c = b["repair_candidate"]
                n = b["plan"]["plan_state"]["N"]
                keep_g = torch.tensor([[n / MAX_ATOMS, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)
                value_g = torch.tensor([c["geometry_features"]], dtype=torch.float32)
                gain = (self.value(h[None], value_g) - self.value(kept[i][None], keep_g))[0]
                c["predicted_gain"] = gain.tolist()
        for b in result:
            choose_result(
                b,
                self.settings.get("selection_penalty", 0.25),
                self.settings.get("minimum_risk_reduction", 0.0),
            )
            if b["E"]["forward_calls"] > maxcalls:
                raise RuntimeError("Repair exceeded the original per-request budget")
        return result
