from itertools import product

import pytest
import torch

from dlm_iclr.c1.distribution import PeriodicAxisLaw, AxisSupportError
from dlm_iclr.c1.feedback import committed_value_log_probability,sampled_value_log_confidence


def make_law():
    unary = torch.tensor([[.3, -.8, .1], [.4, .6, -.2], [-.1, .7, -.4]], dtype=torch.float64, requires_grad=True)
    edge = torch.tensor([[.5, -.3, .2], [-.2, .8, .1]], dtype=torch.float64, requires_grad=True)
    law = PeriodicAxisLaw(unary, edge, parents=[0, 0], temperature=.7,
                          visible_values=[1, 0, 0], visible_mask=[True, False, False])
    return law, unary, edge


@pytest.mark.parametrize('sites,bins', [([1], [2]), ([2], [0]), ([1, 2], [2, 0])])
def test_commit_marginal_matches_enumerated_joint_law_and_gradients(sites, bins):
    law, unary, edge = make_law()
    all_values = torch.tensor(list(product(range(3), repeat=3)))
    keep = all_values[:, 0] == 1
    for i, k in zip(sites, bins, strict=True): keep &= all_values[:, i] == k
    expected = torch.logsumexp(law.log_prob(all_values[keep]), dim=0)
    observed = committed_value_log_probability(law, sites, bins)
    assert torch.allclose(observed, expected, atol=1e-12)
    a = torch.autograd.grad(observed, (unary, edge), retain_graph=True)
    b = torch.autograd.grad(expected, (unary, edge))
    assert all(torch.allclose(x, y, atol=1e-12) for x, y in zip(a, b, strict=True))
    assert a[1].abs().sum() > 0


def test_existing_evidence_cannot_be_used_as_a_new_commit():
    law, _, _ = make_law()
    with pytest.raises(ValueError, match='unresolved'):
        committed_value_log_probability(law, [0], [1])
    with pytest.raises(ValueError):
        committed_value_log_probability(law, [1, 1], [0, 0])


def test_impossible_evidence_is_not_smoothed_into_a_positive_probability():
    law = PeriodicAxisLaw(torch.tensor([[0., -float('inf')], [0., 0.]]), torch.zeros(1, 2))
    with pytest.raises(AxisSupportError):
        committed_value_log_probability(law, [0], [1])


def test_joint_dependence_is_visible_to_the_feedback_score():
    unary = torch.zeros(2, 3, dtype=torch.float64)
    no_edge = PeriodicAxisLaw(unary, torch.zeros(1, 3), visible_values=[0, 0], visible_mask=[True, False])
    aligned = PeriodicAxisLaw(unary, torch.tensor([[3., -1., -1.]]), visible_values=[0, 0], visible_mask=[True, False])
    assert committed_value_log_probability(aligned, [1], [0]) > committed_value_log_probability(no_edge, [1], [0])
    assert committed_value_log_probability(aligned, [1], [1]) < committed_value_log_probability(no_edge, [1], [1])


def test_sample_confidence_matches_enumerated_conditioned_node_probabilities():
    law,_,_=make_law();sample=torch.tensor([1,2,0])
    confidence=sampled_value_log_confidence(law,sample)
    grid=torch.tensor(list(product(range(3),repeat=3)))
    for site,value in enumerate(sample):
        keep=(grid[:,0]==1)&(grid[:,site]==value)
        expected=torch.logsumexp(law.log_prob(grid[keep]),0)
        assert torch.allclose(confidence[site],expected,atol=1e-12)
    with pytest.raises(ValueError):sampled_value_log_confidence(law,sample.float())
