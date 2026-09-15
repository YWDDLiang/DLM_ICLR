import itertools
import numpy as np
import torch
from dlm_iclr.c1.distribution import PeriodicAxisLaw
from dlm_iclr.c2.revision import tilted_probabilities
from dlm_iclr.b0.objective import corrupt
from dlm_iclr.c2.teacher import probabilities
from dlm_iclr.evaluation.workflow import qualify, summary


def test_tree_partition_marginals_and_gradient_against_enumeration():
    rng = torch.Generator().manual_seed(17)
    unary = torch.randn(3, 4, generator=rng, dtype=torch.float64, requires_grad=True)
    edge = torch.randn(2, 4, generator=rng, dtype=torch.float64, requires_grad=True)
    law = PeriodicAxisLaw(unary, edge, temperature=0.7)
    states = torch.tensor(list(itertools.product(range(4), repeat=3)))
    energy = sum(unary[i, states[:, i]] for i in range(3))
    energy += edge[0, (states[:, 0] - states[:, 1]) % 4] + edge[1, (states[:, 0] - states[:, 2]) % 4]
    expected = torch.logsumexp(energy / 0.7, 0)
    torch.testing.assert_close(law.log_partition, expected)
    mass = (energy / 0.7 - expected).exp()
    marginals = torch.stack(
        [torch.stack([mass[states[:, i] == v].sum() for v in range(4)]) for i in range(3)]
    )
    torch.testing.assert_close(law.marginals().exp(), marginals)
    actual_grad = torch.autograd.grad(law.log_partition, (unary, edge), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (unary, edge))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b)


def test_risk_tilt_keeps_support_and_respects_kl_budget():
    p = np.array([0.0, 0.1, 0.3, 0.6])
    risk = np.array([0.0, 1.0, 0.4, 0.1])
    q, info = tilted_probabilities(p, risk, 2.0, 0.01)
    assert q[0] == 0 and np.isclose(q.sum(), 1.0)
    assert info["KL"] <= 0.01 + 1e-12
    assert q @ risk <= p @ risk
    np.testing.assert_allclose(tilted_probabilities(p, risk, 0.0, 0.1)[0], p)


def test_response_corruption_keeps_prompt_and_padding():
    ids = torch.arange(24).reshape(2, 12)
    attention = torch.ones_like(ids)
    attention[1, 10:] = 0
    prefix = torch.tensor([4, 6])
    torch.manual_seed(17)
    noisy, masked, p, answer = corrupt(ids, attention, prefix)
    assert not masked[0, :4].any() and not masked[1, :6].any()
    assert not masked[1, 10:].any()
    assert masked.sum(1).min() >= 1
    torch.testing.assert_close(noisy[~answer], ids[~answer])
    assert p.min() >= 0.001 and p.max() <= 1


def test_physical_teacher_is_normalized_and_reward_weighted():
    q = probabilities([0.0, -0.4, 0.2], beta=0.1)
    assert q[2] > q[0] > q[1]
    torch.testing.assert_close(q.sum(), torch.tensor(1.0, dtype=q.dtype))


def test_terminal_qualification_and_all_request_metrics():
    common = dict(
        comp_valid=True,
        struct_valid=True,
        novel=True,
        unique_representative=True,
        terminal_energy_eV_atom=-1.0,
        hull_energy_eV_atom=-0.95,
        e_above_hull_eV_atom=-0.05,
    )
    rows = [
        dict(common, terminal_status="verified", terminal_verified=True),
        dict(common, terminal_status="not_converged", terminal_verified=False),
        dict(common, terminal_status="invalid_terminal", terminal_verified=False),
    ]
    result = qualify(rows)
    assert [r["strict_stable"] for r in result] == [True, None, False]
    values = summary(result)
    assert values["metrics"]["SUN"]["count_bounds"] == [1, 2]
    assert values["requests"] == 3
