"""Exact-token teachers and source-balanced training data; no continuous-label copying."""
from __future__ import annotations
from collections import Counter, defaultdict
from copy import deepcopy
from .common import digest
from .quality import Measurement, prefer, is_anchor, teacher_rank, raw_nonregression


def compile_teachers(bundles: list[dict], settings: dict) -> tuple[list[dict], list[dict], dict]:
    teachers, pairs = [], []
    reasons = Counter()
    seen_sources = set()
    for bundle in bundles:
        plan = bundle["plan"]
        source = plan["source_id"]
        if source in seen_sources:
            raise ValueError("Each generation request must contribute at most once")
        seen_sources.add(source)
        if plan["provenance"]["usage_role"] != "train":
            raise ValueError("Nontraining source in teacher compiler")
        raw = bundle["raw"]
        baseline = Measurement.from_dict(raw["measurement"])
        candidates, anchors = [], []
        for row in bundle["teachers"]:
            if not row.get("body_token_ids") or not row.get("exact_token_remeasured"):
                reasons["no_exact_token_measurement"] += 1
                continue
            if row["measurement"]["record_key"] != row["exact_record_key"]:
                raise ValueError("Teacher label is not bound to the exact decoded tokens")
            if row["body_prompt"] != plan["body_prompt"]:
                raise ValueError("Chosen/rejected conditions differ")
            if row["body_token_ids"] == raw.get("body_token_ids"):
                reasons["no_semantic_change"] += 1
                continue
            m = Measurement.from_dict(row["measurement"])
            ok, reason = prefer(m, baseline, settings["quality"], require_hull=settings["require_hull"])
            reasons[reason] += 1
            if ok:
                candidates.append(row)
            # An unknown/unverified baseline cannot supply a preference target.
            # A separately verified low-force teacher can still teach its Plan.
            elif not baseline.trustworthy(settings['require_hull']) and is_anchor(
                m,settings['quality'],require_hull=settings['require_hull']
            ) and (not baseline.has_raw(settings['require_hull']) or raw_nonregression(m,baseline,settings['quality'])):
                anchors.append(row)
        if candidates:
            # One source, one teacher: extra trajectory frames never multiply source weight.
            chosen = max(candidates, key=lambda x: teacher_rank(Measurement.from_dict(x["measurement"]),
                                                              require_hull=settings["require_hull"]))
            base = {"source_id": source, "source_split": "train", "prompt": plan["body_prompt"],
                    "body_prompt": plan["body_prompt"], "plan_state": deepcopy(plan["plan_state"]),
                    "body_token_ids": chosen["body_token_ids"], "measurement": chosen["measurement"],
                    "origin": chosen["origin"], "exact_record_key": chosen["exact_record_key"],
                    "weight": 1.0}
            teachers.append(base)
            if raw.get("body_token_ids") and baseline.trustworthy(settings["require_hull"]):
                pairs.append({**base, "chosen_tokens": chosen["body_token_ids"],
                              "rejected_tokens": raw["body_token_ids"],
                              "rejected_measurement": raw["measurement"],
                              "pair_id": digest([source, chosen["exact_record_key"], baseline.record_key])})
        elif anchors:
            chosen=max(anchors,key=lambda x:teacher_rank(Measurement.from_dict(x['measurement']),
                                                       require_hull=settings['require_hull']))
            teachers.append({'source_id':source,'source_split':'train','prompt':plan['body_prompt'],
                'body_prompt':plan['body_prompt'],'plan_state':deepcopy(plan['plan_state']),
                'body_token_ids':chosen['body_token_ids'],'measurement':chosen['measurement'],
                'origin':chosen['origin'],'exact_record_key':chosen['exact_record_key'],'weight':1.0,
                'supervision':'verified_absolute_teacher_no_relative_claim'})
            reasons['verified_absolute_teacher_without_pair']+=1
        elif raw.get("body_token_ids") and is_anchor(baseline, settings["quality"],
                                                     require_hull=settings["require_hull"]):
            teachers.append({"source_id": source, "source_split": "train", "prompt": plan["body_prompt"],
                             "body_prompt": plan["body_prompt"], "plan_state": deepcopy(plan["plan_state"]),
                             "body_token_ids": raw["body_token_ids"], "measurement": raw["measurement"],
                             "origin": "healthy_student_anchor", "weight": 1.0})
            reasons["healthy_anchor"] += 1
        else:
            reasons["no_teacher_source"] += 1
    return teachers, pairs, {"policy":"strict_tier_and_verified_absolute_SFT_v1",
                             "requested_sources": len(bundles), "teacher_sources": len(teachers),
                             "preference_pairs": len(pairs), "reasons": dict(reasons)}


def balanced_history(round_teachers: list[list[dict]], *, max_per_condition=1) -> list[dict]:
    """Most recent verified teacher per condition; no unlimited accumulation by frequent chemistry."""
    rows, counts = [], defaultdict(int)
    for batch in reversed(round_teachers):
        for row in batch:
            key = digest(row["plan_state"])
            if counts[key] < max_per_condition:
                rows.append(row)
                counts[key] += 1
    return rows
