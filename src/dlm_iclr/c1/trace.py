"""Optional observation of actual lattice and coordinate commit states.

This wrapper delegates all sampling and projection. It has no random draws,
model calls, geometry rules, or modifications to the returned tensors.
"""
from copy import deepcopy


class RecordingCandidateSampler:
    def __init__(self, sampler):
        self.sampler = sampler
        self.events = []
        self.pending = None
        self.attempt_index = -1

    def __getattr__(self, name):
        return getattr(self.sampler, name)

    def begin_attempt(self, schedule, seed):
        self.attempt_index += 1
        self.pending = None
        return self.sampler.begin_attempt(schedule, seed)

    def __call__(self, logits, **kwargs):
        self.pending = None
        body = kwargs["current_tokens"][0, kwargs["prompt_length"]:].tolist()
        group = list(kwargs["group_positions"])
        active = [p for p in group if body[p] == kwargs["mask_id"]]
        result = self.sampler(logits, **kwargs)
        self.pending = {
            "attempt_index": self.attempt_index,
            "semantic_group": int(kwargs["semantic_group"]),
            "step_in_group": int(kwargs["step_in_group"]),
            "axis": -1 if all(1 <= p <= 6 for p in group) else (group[0]-8) % 4,
            "input_body": body,
            "group_positions": group,
            "active_positions": active,
            "temperature": float(kwargs["temperature"]),
        }
        return result

    def record_commit(self, **kwargs):
        result = self.sampler.record_commit(**kwargs)
        if self.pending is not None:
            offset = kwargs["prompt_length"]
            selected = kwargs["transfer_index"][0].nonzero(as_tuple=False).flatten().tolist()
            event = self.pending
            event["committed_positions"] = [p-offset for p in selected]
            event["committed_tokens"] = [int(kwargs["candidates"][0, p]) for p in selected]
            event["candidate_body_before_projection"] = kwargs["candidates"][0, offset:].tolist()
            self.events.append(event)
            self.pending = None
        return result

    def report(self):
        report = self.sampler.report()
        report["construction_state_replay"] = {
            "schema": "actual_all_geometry_commits_v1",
            "events": deepcopy(self.events),
            "sampling_unchanged": True,
            "extra_DLM_calls": 0,
            "extra_random_draws": 0,
            "commit_probability_recorded": False,
        }
        return report
