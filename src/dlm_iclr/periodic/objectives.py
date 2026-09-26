"""Original MP20 axis-prefix supervision; no generated/physical targets.

Training uses typed coordinate support, not the deployment geometry monitor.
All alias mass is merged before whole-law temperature. Repeated-commit
inference selects confidence positions; training uses explicit site prefixes,
so this module does NOT assert identical training/deployment state support.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json

import torch

from dlm_iclr._core.fixed_slot import FixedSlotConfig, SYMBOL_TO_Z, MASK_TOKEN_ID
from dlm_iclr._core.llada_generation import _lattice_matrix_from_token_ids
from dlm_iclr._core.paired_noise import derive_subseed
from dlm_iclr.runtime.io import fingerprint
from dlm_iclr.periodic.distribution import PeriodicAxisLaw, collapse_alias_logits


class AxisTrainingSchema:
    def __init__(self, tokenizer, *, mask_id=MASK_TOKEN_ID):
        self.tokenizer, self.mask_id = tokenizer, int(mask_id)
        self.vocab = tokenizer.get_vocab()
        self.signature = fingerprint(self.vocab)
        self.inverse = {value: name for name, value in self.vocab.items()}
        if len(self.inverse) != len(self.vocab):
            raise ValueError("Tokenizer IDs are not one-to-one")
        self.axis_tokens = [[self.vocab[f"<{axis}_{k:03d}>"] for k in range(101)] for axis in "XYZ"]
        self.coord_map = [{token: k for k, token in enumerate(ids)} for ids in self.axis_tokens]
        self.element_map = {
            self.vocab[f"<E_{s}>"]: z for s, z in SYMBOL_TO_Z.items() if z <= 94 and f"<E_{s}>" in self.vocab
        }
        self.lengths = {f: {self.vocab[f"<{f}_{i:03d}>"]: i for i in range(501)} for f in ("LA", "LB", "LC")}
        self.angles = {
            f: {self.vocab[f"<{f}_{i:03d}>"]: i for i in range(1, 180)} for f in ("AA", "AB", "AG")
        }
        self.constraints = {
            "length_token_to_bin": self.lengths,
            "angle_token_to_bin": self.angles,
            "length_step": FixedSlotConfig().length_step,
        }


def axis_view(row, schema, *, seed, epoch, axis=None, cut=None):
    """One deterministic row view; no dependence on process/global RNG."""
    if type(seed) is not int or not 0 <= seed < 2**63 or type(epoch) is not int or epoch < 0:
        raise ValueError("Explicit nonnegative seed/epoch required")
    view_seed = derive_subseed(seed, "axis_training_v1", epoch, row["source_id"])
    rng = torch.Generator(device="cpu").manual_seed(view_seed)
    selected_axis = int(torch.randint(3, (), generator=rng))
    n = row["plan_state"]["N"]
    selected_cut = int(torch.randint(n, (), generator=rng))
    axis = selected_axis if axis is None else axis
    cut = selected_cut if cut is None else cut
    if type(axis) is not int or axis not in range(3) or type(cut) is not int or not 0 <= cut < n:
        raise ValueError("Axis/cut outside original prefix law")
    target = row["body_token_ids"]
    body = target.copy()
    coords = []
    known = []
    axis_target = []
    for site in range(n):
        site_coord = []
        site_known = []
        for a in range(3):
            position = 8 + 4 * site + a
            value = schema.coord_map[a][target[position]] % 100
            visible = a < axis or (a == axis and site < cut)
            site_coord.append(value / 100.0 if visible else None)
            site_known.append(visible)
            if not visible:
                body[position] = schema.mask_id
            if a == axis:
                axis_target.append(value)
        coords.append(site_coord)
        known.append(site_known)
    lattice = _lattice_matrix_from_token_ids(
        torch.tensor(target), prompt_length=0, constraints=schema.constraints
    )
    view = {
        "source_id": row["source_id"],
        "source_hash": row["_source_hash"],
        "target_hash": row["_target_hash"],
        "seed": view_seed,
        "epoch": epoch,
        "axis": axis,
        "cut": cut,
        "body": body,
        "prompt": row["body_prompt"],
        "plan_state": deepcopy(row["plan_state"]),
        "targets": axis_target,
        "visible_values": [t if i < cut else 0 for i, t in enumerate(axis_target)],
        "visible_mask": [i < cut for i in range(n)],
        "coords": coords,
        "known": known,
        "lattice": lattice.tolist(),
        "elements": [schema.element_map[target[7 + 4 * i]] for i in range(n)],
        "supervision": "original_MP20_typed_axis_prefix_joint_NLL",
        "geometry_support": "typed_support",
    }
    view["view_hash"] = fingerprint(view)
    return view


def forward_views(model, tokenizer, views, *, max_length=1024):
    """One real frozen DLM forward, right pad with true attention, no truncation."""
    if not views:
        raise ValueError("Empty microbatch")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("base constructor must be frozen eval")
    device = next(model.parameters()).device
    sequences = []
    prefix_lengths = []
    for v in views:
        prefix = list(tokenizer(v["prompt"], add_special_tokens=False)["input_ids"])
        seq = prefix + v["body"]
        if len(seq) > max_length:
            raise ValueError("Axis view exceeds maximum sequence length; truncation forbidden")
        sequences.append(seq)
        prefix_lengths.append(len(prefix))
    if tokenizer.pad_token_id is None:
        raise ValueError("Actual tokenizer pad token required")
    width = max(map(len, sequences))
    ids = torch.full((len(views), width), tokenizer.pad_token_id, device=device, dtype=torch.long)
    attention = torch.zeros_like(ids)
    for i, seq in enumerate(sequences):
        ids[i, : len(seq)] = torch.tensor(seq, device=device)
        attention[i, : len(seq)] = 1
    calls = [0]
    hook = model.register_forward_pre_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
    try:
        with torch.no_grad():
            out = model(ids, attention_mask=attention, output_hidden_states=True)
    finally:
        hook.remove()
    if calls[0] != 1:
        raise RuntimeError("Frozen base constructor actual forward count differs from one microbatch call")
    if not out.hidden_states or out.hidden_states[-1].shape[:2] != ids.shape:
        raise ValueError("DLM final hidden layout changed")
    predictions = []
    for i, v in enumerate(views):
        positions = torch.tensor(
            [prefix_lengths[i] + 8 + 4 * j + v["axis"] for j in range(len(v["targets"]))], device=device
        )
        logits = out.logits[i, positions]
        hidden = out.hidden_states[-1][i, positions]
        predictions.append((logits.detach(), hidden.detach()))
    return predictions, {
        "DLM_forwards": calls[0],
        "row_views": len(views),
        "sequence_lengths": list(map(len, sequences)),
        "padded_length": width,
        "padding": "right_attention_zero_no_truncation",
        "hidden_dtype": str(out.hidden_states[-1].dtype),
        "logits_dtype": str(out.logits.dtype),
    }


def loss_from_prediction(head, schema, view, prediction, *, temperature=0.7):
    logits, hidden = prediction
    device = hidden.device
    ids = torch.tensor(schema.axis_tokens[view["axis"]], device=device)
    unary = collapse_alias_logits(logits[:, ids], list(range(101)), q=100)
    visible = torch.tensor(view["visible_mask"], device=device, dtype=torch.bool)
    unary = torch.where(visible[:, None], torch.zeros_like(unary), unary)
    coords = torch.tensor(
        [[float("nan") if x is None else x for x in row] for row in view["coords"]],
        device=device,
        dtype=torch.float64,
    )
    law = head.distribution(
        unary,
        hidden,
        view["elements"],
        view["lattice"],
        coords,
        view["known"],
        axis=view["axis"],
        temperature=temperature,
        visible_values=view["visible_values"],
        visible_mask=visible,
    )
    loss = law.nll(view["targets"])
    # N=1: no edge parameters affect the likelihood. Preserve a zero autograd
    # connection so a singleton-only batch is a legitimate zero-gradient step.
    if not loss.requires_grad:
        loss = loss + head.output.weight.sum() * 0.0
    return loss, {
        "view_hash": view["view_hash"],
        "nll": float(loss.detach()),
        "log_partition": float(law.log_partition.detach()),
        "masked_axis_sites": len(view["targets"]) - view["cut"],
        "edge_count": len(view["targets"]) - 1,
    }


def frozen_versions(model):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Frozen base constructor state changed")
    return tuple(
        (n, id(v), v.data_ptr(), v._version, str(v.dtype), str(v.device))
        for n, v in list(model.named_parameters()) + list(model.named_buffers())
    )
