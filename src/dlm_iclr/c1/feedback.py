"""Feedback scores under the same periodic axis law used by C1.

These scores condition on which sites are observed. They do NOT include the
probability that confidence projection selected those sites. Consequently this
primitive is not an exact construction-trajectory likelihood or DPO objective.
"""
import torch


def sampled_value_log_confidence(law, sample):
    """Calibrated node log marginals of the submitted joint sample values."""
    values=torch.as_tensor(sample,device=law.unary.device)
    if values.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8):
        raise ValueError('Integer sampled bins required')
    if values.shape!=(law.n,) or bool(((values<0)|(values>=law.q)).any()):
        raise ValueError('One supported bin per site required')
    return law.marginals().gather(1,values.long()[:,None]).squeeze(1)


def committed_value_log_probability(law, sites, bins):
    """Log P(z[sites]=bins | stored visible state), marginalizing other sites.

    Potentials and geometry support must be constructed from the actual state
    by the caller. Unlike a typed-unary score, gradients include periodic edge
    potentials as well as unary potentials. No uncommitted sample is a target.
    """
    sites, bins = list(sites), list(bins)
    if (not sites or len(sites) != len(bins) or len(set(sites)) != len(sites)
            or any(type(i) is not int or not 0 <= i < law.n for i in sites)
            or any(type(k) is not int or not 0 <= k < law.q for k in bins)):
        raise ValueError('Distinct valid sites and canonical coordinate bins required')
    if any(bool(law.visible_mask[i]) for i in sites):
        raise ValueError('A submitted coordinate must have been unresolved in the input')
    values = law.values.clone()
    observed = torch.zeros_like(law.visible_mask)
    values[sites] = torch.tensor(bins, device=values.device, dtype=values.dtype)
    observed[sites] = True
    conditioned = law.condition(values, observed)
    return conditioned.log_partition-law.log_partition
