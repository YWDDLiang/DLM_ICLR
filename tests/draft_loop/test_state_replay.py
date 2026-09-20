from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from dlm_iclr.draft_loop.learning import TypedVocabulary
from dlm_iclr.draft_loop.state_replay import (
    ReplayTraceError, coordinate_replay_views, construction_replay_views, forward_replay_scores,
)


class Tokenizer:
    pad_token_id = 0

    def __init__(self):
        self.vocab = {}
        for family in ("LA", "LB", "LC", "AA", "AB", "AG", "X", "Y", "Z"):
            for value in (0, 1, 2, 100):
                self.vocab[f"<{family}_{value:03d}>"] = len(self.vocab)+1

    def get_vocab(self):
        return self.vocab

    def __call__(self, prompt, **kwargs):
        return {"input_ids": [1]*len(prompt)}


def fixture():
    tokenizer = Tokenizer()
    vocab = tokenizer.vocab
    mask = 40
    final = [39] + [vocab[f"<{f}_001>"] for f in ("LA", "LB", "LC", "AA", "AB", "AG")]
    for _ in range(2):
        final += [38] + [vocab[f"<{a}_001>"] for a in "XYZ"]
    body = final.copy()
    for i in range(2):
        for a in range(3):
            body[8+4*i+a] = mask
    events = []
    for a in range(3):
        for step, i in enumerate((1, 0)):
            p = 8+4*i+a
            events.append({
                "attempt_index": 0, "axis": a, "semantic_group": a+1,
                "step_in_group": step, "input_body": body.copy(),
                "active_positions": [8+4*j+a for j in range(2) if body[8+4*j+a] == mask],
                "committed_positions": [p], "committed_tokens": [final[p]],
                # Other auxiliary coordinates need not agree with eventual output.
                "candidate_body_before_projection": final.copy(), "temperature": .7,
            })
            body[p] = final[p]
    plan = {"source_id": "parent:candidate0", "parent_source_id": "parent",
            "body_prompt": "plan", "plan_state": {"N": 2}}
    generated = {"record": {"source_id": plan["source_id"], "body_prompt": plan["body_prompt"],
                           "success": True, "body_token_ids": final},
                 "trace": {"construction_recovery": {"recoveries_used": 0},
                           "periodic_axis": {"enabled": True, "events": events}}}
    return plan, generated, tokenizer, mask


def test_actual_student_inputs_and_commits_are_kept_and_grouped_by_plan():
    plan, generated, _, mask = fixture()
    views, report = coordinate_replay_views(plan, generated, mask_id=mask)
    assert len(views) == 6 and report["coordinate_commits"] == 6
    assert {v.source_id for v in views} == {"parent"}
    assert views[0].positions == (12,) and views[1].positions == (8,)
    assert views[0].input_body[12] == mask
    assert views[1].input_body[12] == views[0].tokens[0]
    assert all(v.input_body[1:7] == tuple(generated["record"]["body_token_ids"][1:7]) for v in views)
    assert report["lattice_commits"] == 0 and not report["lattice_training_ready"]
    assert not report["exact_C1_transition_likelihood"]


@pytest.mark.parametrize("change", ["future_leak", "wrong_target", "missing_event", "recovery", "wrong_id"])
def test_untraceable_or_misaligned_states_are_rejected(change):
    plan, generated, _, mask = fixture()
    events = generated["trace"]["periodic_axis"]["events"]
    if change == "future_leak":
        events[0]["input_body"][9] = generated["record"]["body_token_ids"][9]
    elif change == "wrong_target":
        events[0]["committed_tokens"][0] += 1
    elif change == "missing_event":
        del events[2]
    elif change == "recovery":
        generated["trace"]["construction_recovery"]["recoveries_used"] = 1
    else:
        generated["record"]["source_id"] = "other"
    with pytest.raises(ReplayTraceError):
        coordinate_replay_views(plan, generated, mask_id=mask)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(41, 6)
        self.output = torch.nn.Linear(6, 41)
        self.calls = []

    def forward(self, ids, attention_mask):
        self.calls.append((ids.detach().clone(), attention_mask.detach().clone()))
        values = self.embedding(ids)*attention_mask[..., None]
        context = values.sum(1)/attention_mask.sum(1, keepdim=True)
        return SimpleNamespace(logits=self.output(values+context[:, None]))

    @property
    def device(self):
        return self.embedding.weight.device

    def get_output_embeddings(self):
        return self.output


def test_batched_scoring_replays_visible_values_and_has_real_gradients():
    plan, generated, tokenizer, mask = fixture()
    views, _ = coordinate_replay_views(plan, generated, mask_id=mask)
    second = replace(views[1], prompt="p")
    model = TinyModel()
    scores = forward_replay_scores(model, tokenizer, TypedVocabulary(tokenizer),
                                   [views[0], second], max_length=100)
    assert len(model.calls) == 1
    ids, attention = model.calls[0]
    assert ids[0, 4:].tolist() == list(views[0].input_body)
    assert ids[1, 1:16].tolist() == list(second.input_body)
    assert attention[1, 16:].sum() == 0
    loss = -torch.stack([s[0] for s in scores]).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert model.embedding.weight.grad.abs().sum() > 0
    assert model.output.weight.grad.abs().sum() > 0


def test_alias_target_merge_does_not_rewrite_observed_inputs_or_allow_truncation():
    plan, generated, tokenizer, mask = fixture()
    views, _ = coordinate_replay_views(plan, generated, mask_id=mask)
    vocab = TypedVocabulary(tokenizer)
    model = TinyModel()
    body = list(views[1].input_body)
    body[12] = tokenizer.vocab["<X_100>"]
    alias = replace(views[1], input_body=tuple(body), tokens=(tokenizer.vocab["<X_100>"],))
    zero = replace(alias, tokens=(tokenizer.vocab["<X_000>"],))
    scores = forward_replay_scores(model, tokenizer, vocab, [alias, zero], max_length=100)
    assert torch.equal(scores[0][0], scores[1][0])
    assert model.calls[0][0][0, 4+12] == tokenizer.vocab["<X_100>"]
    with pytest.raises(ValueError, match="Truncating"):
        forward_replay_scores(model, tokenizer, vocab, [views[0]], max_length=10)


def test_uncommitted_auxiliary_draws_do_not_change_views():
    plan, generated, _, mask = fixture()
    original, report = coordinate_replay_views(plan, generated, mask_id=mask)
    changed = deepcopy(generated)
    changed["trace"]["periodic_axis"]["events"][0]["candidate_body_before_projection"][8] += 1
    other, report2 = coordinate_replay_views(plan, changed, mask_id=mask)
    assert original == other and report["trace_key"] == report2["trace_key"]


def test_old_trace_cannot_be_declared_lattice_replay_ready():
    plan, generated, _, mask = fixture()
    with pytest.raises(ReplayTraceError, match="lattice commit states were not recorded"):
        construction_replay_views(plan, generated, mask_id=mask)


def test_recording_does_not_change_generation_rng_forwards_or_confidence_commits():
    from dlm_iclr._core.paired_llada import generate_paired_exact_plan, _paired_suffix_candidates
    from dlm_iclr.c1.trace import RecordingCandidateSampler

    class Sampler:
        enabled = True

        def __init__(self):
            self.commits = []

        def begin_attempt(self, schedule, seed):
            pass

        def __call__(self, logits, *, hidden, group_positions, mask_id, **kwargs):
            return _paired_suffix_candidates(logits, **kwargs)

        def record_commit(self, **kwargs):
            self.commits.append(kwargs["transfer_index"].clone())

        def report(self):
            return {"enabled": True, "events": []}

    plan, generated, tokenizer, mask = fixture()
    schema = [[token] for token in generated["record"]["body_token_ids"]]
    vocab = TypedVocabulary(tokenizer)
    for p in list(range(1, 7))+[8, 9, 10, 12, 13, 14]:
        schema[p] = list(vocab.tables[vocab.family(p)].values())
    groups = [[0], list(range(1, 7)), [7, 11], [8, 12], [9, 13], [10, 14]]
    model = TinyModel().eval()
    inner = Sampler()
    recorded_inner = Sampler()
    recorder = RecordingCandidateSampler(recorded_inner)
    for sampler in (inner, recorder):
        sampler.begin_attempt(groups, 17)
    args = dict(base_seeds=[17], attention_mask=torch.ones(1, 4, dtype=torch.long),
                gen_length=15, temperature=.7, cfg_scale=0, remasking="low_confidence",
                mask_id=mask, allowed_token_ids_by_generation_pos=schema,
                prefill_token_ids_by_generation_pos={p: schema[p] for p in (0, 7, 11)},
                generation_position_groups=groups, lightweight_decoding_constraints=None)
    torch.manual_seed(912)
    a = generate_paired_exact_plan(model, torch.ones(1, 4, dtype=torch.long), candidate_sampler=inner, **args)
    rng_a = torch.get_rng_state().clone()
    calls_a = len(model.calls)
    model.calls.clear()
    torch.manual_seed(912)
    b = generate_paired_exact_plan(model, torch.ones(1, 4, dtype=torch.long), candidate_sampler=recorder, **args)
    assert torch.equal(a, b) and torch.equal(rng_a, torch.get_rng_state())
    assert len(model.calls) == calls_a == 12
    assert all(torch.equal(x, y) for x, y in zip(inner.commits, recorded_inner.commits, strict=True))
    generated["record"]["body_token_ids"] = b[0, 4:].tolist()
    generated["trace"]["periodic_axis"] = recorder.report()
    views, report = construction_replay_views(plan, generated, mask_id=mask)
    assert report["lattice_training_ready"]
    assert report["lattice_commits"] == 6 and report["coordinate_commits"] == 6
    assert all(v.axis == -1 for v in views[:6])
    assert all(views[0].input_body[p] == mask for p in list(range(1, 7))+[8, 9, 10, 12, 13, 14])
    scored = forward_replay_scores(model, tokenizer, vocab, views[:2], max_length=100)
    loss = -torch.stack([x[0] for x in scored]).mean()
    assert torch.isfinite(loss)
    loss.backward()
