"""Learned periodic axis tree; rebuild after every DLM commit/parameter update.

This module executes no DLM/F/physics call and imposes no distance/SPD veto on
sampled coordinates. Zero edge energy recovers the SAME unary distribution,
not the complete legacy sampler, RNG stream, support rules or S0 checkpoint.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Sequence

import torch
from torch import Tensor, nn


class AxisSupportError(ValueError):
    """The requested law/evidence has no nonzero finite mass."""


def registered_parents(n: int, parents: Sequence[int] | None = None) -> tuple[int, ...]:
    """Root is site 0; child c uses parents[c-1], strictly smaller than c."""
    if type(n) is not int or n < 1:
        raise ValueError("Positive site count required")
    result = tuple((c - 1) // 2 for c in range(1, n)) if parents is None else tuple(parents)
    if len(result) != n - 1 or any(
        not isinstance(p, Integral) or isinstance(p, bool) or not 0 <= p < c for c, p in enumerate(result, 1)
    ):
        raise ValueError("Registered tree requires one parent < child, rooted at site 0")
    return tuple(map(int, result))


def _integer(values, device=None):
    result = torch.as_tensor(values, device=device)
    if result.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("Coordinates must be integer grid indices, not token IDs/floats")
    return result.long()


def canonical_grid_indices(values, q: int = 100):
    """Explicit periodic bin canonicalization (e.g. Q -> 0); no token decoding."""
    if type(q) is not int or q < 2:
        raise ValueError("Grid size must be at least two")
    return _integer(values).remainder(q)


def _logs(value, name, *, finite=False):
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(name + " requires a floating tensor")
    if (
        bool(torch.isnan(value).any())
        or bool(torch.isposinf(value).any())
        or (finite and not bool(torch.isfinite(value).all()))
    ):
        raise ValueError(name + " has invalid log potentials")
    return value.double()


def _lse(values, dim):
    # Empty structural sums stay exactly -inf, with zero rather than NaN grad.
    valid = torch.isfinite(values).any(dim=dim)
    safe = torch.where(valid.unsqueeze(dim), values, torch.zeros_like(values))
    return torch.where(
        valid, torch.logsumexp(safe, dim=dim), torch.full_like(valid, -torch.inf, dtype=values.dtype)
    )


def collapse_alias_logits(logits: Tensor, token_bins, *, q: int = 100):
    """Sum token mass at periodic aliases, BEFORE temperature by this contract.

    logits[..., K] are untempered token log potentials; token_bins[K] identify
    semantic grid bins modulo Q. Absent bins remain -inf. Merging after applying
    temperature is a different policy and must be called explicitly that way.
    """
    values = _logs(logits, "alias logits")
    if values.ndim < 1:
        raise ValueError("Alias logits need a token dimension")
    bins = canonical_grid_indices(token_bins, q).to(values.device)
    if bins.shape != (values.shape[-1],):
        raise ValueError("One semantic bin per token logit required")
    columns = []
    for k in range(q):
        selected = values[..., bins == k]
        columns.append(
            _lse(selected, -1) if selected.shape[-1] else values.new_full(values.shape[:-1], -torch.inf)
        )
    return torch.stack(columns, -1)


class PeriodicAxisLaw:
    """Exact single-axis tree law on [N,Q], with edge table E(parent-child).

    Whole-law temperature divides both unary and edge log potentials. Unaries
    may have explicit -inf support; learned edges must be finite. Internally
    float64, retaining autograd. O(N Q^2) inference/autograd intermediates.
    Visible values are canonical grid indices; aliases require explicit mapping.
    """

    def __init__(
        self,
        log_unary: Tensor,
        periodic_log_edge: Tensor,
        *,
        parents=None,
        temperature=1.0,
        visible_values=None,
        visible_mask=None,
    ):
        unary = _logs(log_unary, "log_unary")
        edge = _logs(periodic_log_edge, "periodic_log_edge", finite=True)
        if unary.ndim != 2:
            raise ValueError("Unaries require [N,Q]")
        self.n, self.q = unary.shape
        if self.q < 2 or edge.shape != (self.n - 1, self.q) or edge.device != unary.device:
            raise ValueError("Edge shape/device must match [N-1,Q]")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("Finite positive whole-law temperature required")
        self.parents = registered_parents(self.n, parents)
        self.children = tuple(
            tuple(c for c in range(1, self.n) if self.parents[c - 1] == p) for p in range(self.n)
        )
        self.temperature = float(temperature)
        self.raw_log_unary, self.raw_log_edge = unary, edge
        self.log_unary, self.log_edge = unary / temperature, edge / temperature
        if bool((torch.isfinite(unary) & ~torch.isfinite(self.log_unary)).any()) or not bool(
            torch.isfinite(self.log_edge).all()
        ):
            raise FloatingPointError("Temperature scaling overflow")
        if visible_values is None and visible_mask is None:
            visible_values = torch.zeros(self.n, dtype=torch.long, device=unary.device)
            visible_mask = torch.zeros(self.n, dtype=torch.bool, device=unary.device)
        elif visible_values is None or visible_mask is None:
            raise ValueError("Supply both visible values and mask")
        self.values = self._values(visible_values)
        self.visible_mask = torch.as_tensor(visible_mask, device=unary.device)
        if (
            self.values.shape != (self.n,)
            or self.visible_mask.shape != (self.n,)
            or self.visible_mask.dtype != torch.bool
        ):
            raise ValueError("Evidence needs [N] integer values and boolean mask")
        grid = torch.arange(self.q, device=unary.device)
        self.difference = (grid[:, None] - grid[None, :]).remainder(self.q)
        allowed = ~self.visible_mask[:, None] | (grid[None] == self.values[:, None])
        self.unary = torch.where(allowed, self.log_unary, torch.full_like(unary, -torch.inf))
        self.subtree, self.upward = [None] * self.n, [None] * self.n
        for node in reversed(range(self.n)):
            score = self.unary[node]
            for child in self.children[node]:
                score = score + self.upward[child]
            self.subtree[node] = score
            if node:
                self.upward[node] = _lse(self._edge(node) + score[None, :], -1)
        self.log_partition = _lse(self.subtree[0], -1)
        if not bool(torch.isfinite(self.log_partition)):
            raise AxisSupportError("Empty evidence/support or unresolvable partition")

    def _values(self, values):
        result = _integer(values, self.log_unary.device)
        if result.ndim < 1 or result.shape[-1] != self.n or bool(((result < 0) | (result >= self.q)).any()):
            raise ValueError("Values must end in [N] canonical grid bins")
        return result

    def _edge(self, child):
        return self.log_edge[child - 1][self.difference]

    def condition(self, values, mask):
        """Add visible evidence to these fixed potentials; does not re-run DLM."""
        values = self._values(values)
        mask = torch.as_tensor(mask, device=self.unary.device)
        if values.shape != (self.n,) or mask.shape != (self.n,) or mask.dtype != torch.bool:
            raise ValueError("Evidence requires [N] values/mask")
        if bool((mask & self.visible_mask & (values != self.values)).any()):
            raise AxisSupportError("Conflicting evidence")
        values = torch.where(self.visible_mask, self.values, values)
        return PeriodicAxisLaw(
            self.raw_log_unary,
            self.raw_log_edge,
            parents=self.parents,
            temperature=self.temperature,
            visible_values=values,
            visible_mask=mask | self.visible_mask,
        )

    def log_prob(self, values):
        values = self._values(values)
        score = self.unary[torch.arange(self.n, device=values.device), values].sum(-1)
        for child, parent in enumerate(self.parents, 1):
            score = (
                score + self.log_edge[child - 1, (values[..., parent] - values[..., child]).remainder(self.q)]
            )
        return score - self.log_partition

    def nll(self, values, *, reduction="mean"):
        result = -self.log_prob(values)
        if not bool(torch.isfinite(result).all()):
            raise AxisSupportError("Supervision lies outside declared support/evidence")
        if reduction == "none":
            return result
        if reduction == "mean":
            return result.mean()
        if reduction == "sum":
            return result.sum()
        raise ValueError("Unknown NLL reduction")

    def marginals(self, *, return_edges=False):
        """Return log node marginals [N,Q], optionally edge [N-1,Q,Q]."""
        down = [None] * self.n
        down[0] = torch.zeros_like(self.unary[0])
        nodes, edges = [], [None] * (self.n - 1)
        for node in range(self.n):
            nodes.append(self.subtree[node] + down[node] - self.log_partition)
            children = self.children[node]
            # Excluding a child by subtraction would produce -inf - -inf.
            # Prefix/suffix sums avoid that and remain linear in tree degree.
            prefix = [torch.zeros_like(self.unary[node])]
            for child in children:
                prefix.append(prefix[-1] + self.upward[child])
            suffix = [None] * (len(children) + 1)
            suffix[-1] = torch.zeros_like(self.unary[node])
            for index in reversed(range(len(children))):
                suffix[index] = suffix[index + 1] + self.upward[children[index]]
            for index, child in enumerate(children):
                outside = self.unary[node] + down[node] + prefix[index] + suffix[index + 1]
                pair = outside[:, None] + self._edge(child)
                down[child] = _lse(pair, 0)
                if return_edges:
                    edges[child - 1] = pair + self.subtree[child][None, :] - self.log_partition
        node_logs = torch.stack(nodes)
        if not return_edges:
            return node_logs
        return node_logs, torch.stack(edges) if edges else self.unary.new_empty((0, self.q, self.q))

    @torch.no_grad()
    def sample(self, count=1, *, generator):
        """Ancestral JOINT samples [count,N], with explicit same-device RNG.

        Log-domain exponential-race sampling avoids softmax mass underflow.
        Like any floating RNG, probabilities below its resolution are not an
        exact real-arithmetic random oracle. It never adds epsilon to support.
        """
        if type(count) is not int or count < 1 or not isinstance(generator, torch.Generator):
            raise ValueError("Positive sample count and explicit Torch generator required")

        def draw(log_weights):
            noise = torch.empty_like(log_weights).exponential_(generator=generator)
            if not bool(torch.isfinite(noise).all()) or bool((noise <= 0).any()):
                raise FloatingPointError("Nonfinite random exponential variate")
            return (log_weights - noise.log()).argmax(-1)

        result = torch.empty((count, self.n), dtype=torch.long, device=self.unary.device)
        result[:, 0] = draw(self.subtree[0].expand(count, -1))
        for child, parent in enumerate(self.parents, 1):
            result[:, child] = draw(self._edge(child)[result[:, parent]] + self.subtree[child])
        return result


class PeriodicAxisHead(nn.Module):
    """Finite learned Fourier edge energies using this DLM call's context.

    lattice rows are real Cartesian lattice vectors (Angstrom). coords are
    fractional, known is [N,3] bool, elements are atomic numbers 1..94. Only
    transverse coordinates jointly known at an edge enter periodic differences;
    active-axis coordinates NEVER enter features, including visible active-axis
    values (handled as law evidence). Fixed tree is not permutation equivariant.
    """

    def __init__(
        self, hidden_size: int, *, width=64, harmonics=8, q=100, potential_bound=4.0, element_width=8
    ):
        super().__init__()
        if (
            any(type(v) is not int or v < 1 for v in (hidden_size, width, harmonics, element_width))
            or type(q) is not int
            or q < 2
        ):
            raise ValueError("Positive head dimensions and Q>=2 required")
        if not math.isfinite(potential_bound) or potential_bound <= 0:
            raise ValueError("Finite positive energy bound required")
        self.hidden_size, self.width, self.harmonics, self.q = hidden_size, width, harmonics, q
        self.potential_bound, self.element_width = float(potential_bound), element_width
        self.normalize = nn.LayerNorm(hidden_size)
        self.elements = nn.Embedding(95, element_width)
        # Gram6 + log volume1 + axis onehot3 + transverse sin/cos/known4 each.
        self.context = nn.Linear(2 * hidden_size + 2 * element_width + 18, width)
        self.output = nn.Linear(width, 2 * harmonics)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def config(self):
        return dict(
            hidden_size=self.hidden_size,
            width=self.width,
            harmonics=self.harmonics,
            q=self.q,
            potential_bound=self.potential_bound,
            element_width=self.element_width,
        )

    def forward(self, hidden, elements, lattice, coords, known, *, axis, parents=None):
        if hidden.ndim != 2 or hidden.shape[1] != self.hidden_size or not bool(torch.isfinite(hidden).all()):
            raise ValueError("Finite DLM hidden [N,H] required")
        n = hidden.shape[0]
        tree = registered_parents(n, parents)
        if type(axis) is not int or axis not in (0, 1, 2):
            raise ValueError("Axis must be 0,1,2")
        device, dtype = self.output.weight.device, self.output.weight.dtype
        if hidden.device != device:
            raise ValueError("Head and DLM hidden must share a device")
        z = _integer(elements, device)
        if z.shape != (n,) or bool(((z < 1) | (z > 94)).any()):
            raise ValueError("Element atomic numbers must be 1..94")
        lattice = torch.as_tensor(lattice, device=device, dtype=torch.float64)
        coords = torch.as_tensor(coords, device=device, dtype=torch.float64)
        known = torch.as_tensor(known, device=device)
        if lattice.shape != (3, 3) or not bool(torch.isfinite(lattice).all()):
            raise ValueError("Finite real lattice [3,3] required")
        if coords.shape != (n, 3) or known.shape != (n, 3) or known.dtype != torch.bool:
            raise ValueError("Fractional coords and known mask need [N,3]")
        transverse = [k for k in range(3) if k != axis]
        c, k = coords[:, transverse], known[:, transverse]
        if not bool(torch.isfinite(c[k]).all()):
            raise ValueError("Known transverse coordinates must be finite")
        c = torch.where(k, c, torch.zeros_like(c)).remainder(1.0)
        determinant = torch.linalg.det(lattice).abs()
        if not bool(torch.isfinite(determinant)) or float(determinant) == 0.0:
            raise ValueError("Context lattice must be nonsingular")
        gram = lattice @ lattice.T
        scale = gram.trace()
        if not bool(torch.isfinite(scale)) or float(scale) <= 0.0:
            raise ValueError("Lattice scale overflow")
        gram = gram / scale
        g = torch.cat(
            [
                gram[[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]],
                determinant.log().reshape(1),
                torch.nn.functional.one_hot(torch.tensor(axis, device=device), 3).to(gram),
            ]
        )
        h = self.normalize(hidden.to(dtype))
        e = self.elements(z)
        parent = torch.tensor(tree, device=device, dtype=torch.long)
        child = torch.arange(1, n, device=device)
        joint = k[parent] & k[child]
        delta = (c[parent] - c[child]) * 2 * math.pi
        transverse_features = torch.stack(
            [
                torch.where(joint, delta.sin(), 0.0),
                torch.where(joint, delta.cos(), 0.0),
                k[parent].to(delta),
                k[child].to(delta),
            ],
            -1,
        ).reshape(n - 1, 8)
        features = torch.cat(
            [
                h[parent],
                h[child],
                e[parent],
                e[child],
                g.expand(n - 1, -1).to(dtype),
                transverse_features.to(dtype),
            ],
            -1,
        )
        coefficients = self.output(torch.nn.functional.silu(self.context(features)))
        grid = torch.arange(self.q, device=device, dtype=dtype) / self.q
        frequency = torch.arange(1, self.harmonics + 1, device=device, dtype=dtype)
        phase = 2 * math.pi * frequency[:, None] * grid[None, :]
        basis = torch.cat([phase.cos(), phase.sin()], 0)
        raw = coefficients @ basis / math.sqrt(self.harmonics)
        energy = self.potential_bound * torch.tanh(raw)
        if not bool(torch.isfinite(energy).all()):
            raise FloatingPointError("Nonfinite learned periodic energy")
        return energy

    def distribution(
        self,
        log_unary,
        hidden,
        elements,
        lattice,
        coords,
        known,
        *,
        axis,
        parents=None,
        temperature=1.0,
        visible_values=None,
        visible_mask=None,
    ):
        if log_unary.shape != (hidden.shape[0], self.q):
            raise ValueError("Head grid and unary support differ")
        edge = self(hidden, elements, lattice, coords, known, axis=axis, parents=parents)
        return PeriodicAxisLaw(
            log_unary,
            edge,
            parents=parents,
            temperature=temperature,
            visible_values=visible_values,
            visible_mask=visible_mask,
        )
