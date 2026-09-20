"""Construct-time prefix/action values. Never evaluate an incomplete crystal with CHGNet."""
from __future__ import annotations
from collections import defaultdict
import math
from pathlib import Path
import numpy as np
import torch
from .common import digest, read_json, seed_for, write_json


def positive_advantage_index(scores, margin=0.0):
    """The original sampled action is the exact-zero reference, including ties."""
    scores=np.asarray(scores,dtype=float)
    winner=int(np.argmax(scores))
    return winner if scores[winner]-scores[0] > max(0.0,float(margin)) else 0


def projected_action(sample, sampler, logits, active, axis, offset):
    """Exactly mirror legacy batch-one topk on the full sequence, including ties."""
    confidence = torch.full(logits.shape[:2], -float("inf"), dtype=logits.dtype, device=logits.device)
    for pos in active:
        token = sampler.axis_tokens[axis][int(sample[(pos-8)//4])]
        confidence[0, offset+pos] = torch.softmax(logits[0, offset+pos], -1)[token]
    pos = int(torch.topk(confidence[0], k=1).indices[0])-offset
    token = sampler.axis_tokens[axis][int(sample[(pos-8)//4])]
    return (pos, token), float(confidence[0, offset+pos])


class FeatureBuilder:
    def __init__(self, width=32, *, geometry=False):
        self.width, self.projections = width, {}
        self.geometry=geometry

    def __call__(self, hidden, body, offset, action, probability, axis, sampler):
        size = hidden.shape[-1]
        key = (size, str(hidden.device))
        if key not in self.projections:
            rng = torch.Generator(device="cpu").manual_seed(191823)
            self.projections[key] = (torch.randn(size, self.width, generator=rng) / math.sqrt(size)).to(hidden.device)
        pos, token = action
        local = hidden[0, offset+pos].float()
        global_ = hidden[0, offset:offset+len(body)].float().mean(0)
        def norm(v):
            return (v-v.mean()) / v.std(unbiased=False).clamp_min(1e-5)
        vectors = torch.cat((norm(global_) @ self.projections[key], norm(local) @ self.projections[key]))
        k = sampler.coord_map[axis][token] % 100
        from dlm_iclr._core.fixed_slot import MASK_TOKEN_ID
        known = float((body != MASK_TOKEN_ID).sum()) / len(body)
        numeric = [sampler.n/20, (pos-8)//4/max(sampler.n,1), *[float(a==axis) for a in range(3)],
                   known, math.sin(2*math.pi*k/100), math.cos(2*math.pi*k/100),
                   math.log(max(probability,1e-12))/20]
        extra=[]
        if self.geometry:
            from .prefix_geometry import prefix_geometry_features
            extra=prefix_geometry_features(body,action,sampler.constraints).tolist()
        return np.asarray(vectors.cpu().tolist()+numeric+extra, dtype=float)


class ProcessVerifier:
    def __init__(self, saved: dict, binding: str):
        if saved["binding"] != binding:
            raise ValueError("Verifier belongs to a different DLM/C1 asset pair")
        if not saved["validation"]["deployable"]:
            raise ValueError("Uncalibrated verifier is not deployable")
        self.saved = saved
        self.mean = np.asarray(saved["mean"])
        self.scale = np.asarray(saved["scale"])
        self.weight = np.asarray(saved["weight"])
        self.bias = float(saved["bias"])

    def predict(self, features):
        x = np.asarray(features)
        feature_map=self.saved.get('feature_map')
        if feature_map:
            if feature_map['kind'] not in ('action_context_bilinear_v1','action_context_geometry_v1'):
                raise ValueError('Unknown verifier feature map')
            from .advantage import action_context_features
            x=action_context_features(x,feature_map['projection_width'],
                                      geometry=feature_map['kind']=='action_context_geometry_v1')
        result = ((x-self.mean)/self.scale) @ self.weight + self.bias
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite prefix value")
        if self.saved.get('prediction_kind')=='within_prefix_advantage':
            # Only differences are identified; the controller clips differences.
            # Clipping individual scores would erase gains via a shared offset.
            return result
        return np.clip(result, -12.0, 0.0)


class JointController:
    """Observe, force a counterfactual via exact full-prefix replay, or guide a commit.

    Disabled guidance returns the original joint sample exactly. Alternative
    draws have private RNGs and never advance the baseline's random stream.
    """
    def __init__(self, settings, *, guide=None, forced=None, collect=False):
        self.settings, self.guide, self.forced, self.collect = settings, guide, forced, collect
        self.builder = FeatureBuilder(settings["projection_width"],geometry=settings.get('geometry_features',False))
        self.probes, self.decisions = [], []
        self.pending = None
        self.forced_hit = False
        self.force_verified = False
        self.guided_actions = 0

    def select(self, sampler, law, sample, logits, hidden, body, active, axis, arguments):
        self.pending = None
        from dlm_iclr._core.fixed_slot import MASK_TOKEN_ID
        remaining = int((body == MASK_TOKEN_ID).sum())
        if axis != self.settings["axis"] or remaining > self.settings["max_remaining"] or not active:
            return sample
        state = {"body":body.tolist(), "axis":axis, "group":arguments["semantic_group"],
                 "step":arguments["step_in_group"], "base_seed":arguments["base_seeds"][0]}
        state_key = digest(state)
        count = self.settings["candidate_samples"]
        draws = [sample]
        for index in range(1, count):
            seed = seed_for(state["base_seed"], "loop_joint", state["group"], state["step"], index)
            draws.append(law.sample(1, generator=torch.Generator(device=sample.device).manual_seed(seed))[0])
        actions, features = [], []
        for draw in draws:
            action, prob = projected_action(draw, sampler, logits, active, axis, arguments["prompt_length"])
            actions.append(list(action))
            features.append(self.builder(hidden, body, arguments["prompt_length"], action, prob, axis, sampler).tolist())
        selected = 0
        score_info={}
        if self.forced is not None and state_key == self.forced["state_key"]:
            matches = [i for i,a in enumerate(actions) if a == self.forced["action"]]
            if not matches:
                raise RuntimeError("Counterfactual action is absent from the exactly replayed candidate pool")
            selected = matches[0]
            self.forced_hit = True
        elif (self.guide is not None and self.forced is None and self.guided_actions < 1
              and (not self.settings.get('skip_duplicate_opportunity',False)
                   or any(action!=actions[0] for action in actions[1:]))):
            scores = self.guide.predict(features)
            scores = np.clip(scores-scores[0], -self.settings["utility_clip"], self.settings["utility_clip"])
            eta = self.settings["strength"]
            if eta > 0:
                margin=self.guide.saved.get('validation',{}).get('advantage_margin',0.0)
                selected=positive_advantage_index(scores,margin)
                score_info={'predicted_gain':float(scores[selected]),'advantage_margin':float(margin)}
                self.guided_actions += 1
        probe = {"state_key":state_key, "state":state, "actions":actions, "features":features,
                 "selected":selected, "selected_action":actions[selected]}
        if self.collect:
            self.probes.append(probe)
        self.decisions.append({"state_key":state_key, "selected":selected, "action":actions[selected],
                               "selection_policy":"positive_completion_advantage_keep_ties_v1",
                               "opportunity_policy":('first_distinct_candidate' if self.settings.get('skip_duplicate_opportunity',False)
                                                     else 'first_eligible_prefix'),**score_info})
        self.pending = probe
        return draws[selected]

    def record_commit(self, event):
        if self.pending is None:
            return
        actual = list(zip(event["committed_positions"], event["committed_tokens"]))
        expected = tuple(self.pending["selected_action"])
        if actual != [expected]:
            raise RuntimeError("Verifier predicted a different event from the actual single-token commit")
        if self.forced_hit and self.forced and self.pending["state_key"] == self.forced["state_key"]:
            self.force_verified = True
        self.pending = None


def fit_process_verifier(rows, output, binding, settings, *, seed=17):
    """Source-balanced ridge value; held-out sources validate actual branch ranking."""
    grouped = defaultdict(list)
    for row in rows:
        if row.get("target") is not None:
            grouped[row["source_id"]].append(row)
    sources = sorted(grouped)
    rng = np.random.default_rng(seed)
    rng.shuffle(sources)
    cut = max(1, int(len(sources)*0.8))
    train_sources, validation_sources = sources[:cut], sources[cut:]
    train = [r for s in train_sources for r in grouped[s]]
    val = [r for s in validation_sources for r in grouped[s]]
    result = {"schema":"prefix_action_value_v1", "binding":binding,
              "target":"raw_completed_draft_quality_not_F_endpoint", "settings":settings,
              "train_sources":train_sources, "validation_sources":validation_sources}
    if len(train_sources) < settings["min_train_sources"] or len(validation_sources) < settings["min_validation_sources"]:
        result["validation"] = {"deployable":False, "reason":"insufficient_independent_sources"}
        write_json(output, result)
        return result
    x = np.asarray([r["features"] for r in train]); y = np.asarray([r["target"] for r in train])
    counts = {s:len(grouped[s]) for s in train_sources}
    weights = np.asarray([1/counts[r["source_id"]] for r in train]); weights /= weights.sum()
    mean = (x*weights[:,None]).sum(0)
    scale = np.sqrt(((x-mean)**2*weights[:,None]).sum(0)).clip(min=0.05)
    z = (x-mean)/scale
    bias = float(weights @ y)
    lhs = z.T @ (z*weights[:,None]) + settings["ridge"]*np.eye(z.shape[1])
    coefficient = np.linalg.solve(lhs, z.T @ (weights*(y-bias)))
    predictions = ((np.asarray([r["features"] for r in val])-mean)/scale) @ coefficient+bias
    truth = np.asarray([r["target"] for r in val])
    groups = defaultdict(list)
    for row, pred in zip(val,predictions,strict=True):
        groups[(row["source_id"],row["state_key"])].append((row["target"],float(pred)))
    comparisons = []; overestimates=[]
    for group in groups.values():
        for i,(t,p) in enumerate(group):
            for u,q in group[i+1:]:
                if abs(t-u) >= 0.05:
                    comparisons.append(0.5 if p == q else float((t-u)*(p-q)>0))
                if i==0:
                    overestimates.append(max(0.0,(q-p)-(u-t)))
    accuracy = float(np.mean(comparisons)) if comparisons else None
    mse, baseline = float(np.mean((predictions-truth)**2)), float(np.mean((bias-truth)**2))
    deployable = (accuracy is not None and accuracy >= settings["min_pair_accuracy"]
                  and mse <= max(1e-8,1.2*baseline))
    result.update(mean=mean.tolist(), scale=scale.tolist(), weight=coefficient.tolist(), bias=bias,
                  validation={"deployable":bool(deployable), "pair_accuracy":accuracy,
                              "advantage_margin":float(np.quantile(overestimates,.75)) if overestimates else 0.0,
                              "margin_source":"held_out_source_pair_overestimate_q75_not_a_guarantee",
                              "comparisons":len(comparisons), "mse":mse, "constant_mse":baseline})
    write_json(output,result)
    return result
