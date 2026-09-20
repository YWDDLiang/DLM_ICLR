from types import SimpleNamespace

import pytest
import torch

from dlm_iclr.c1.distribution import PeriodicAxisHead,collapse_alias_logits
from dlm_iclr.c1.objectives import AxisTrainingSchema
from dlm_iclr.c1.sampling import AxisCandidateSampler


def fixture():
    vocab={}
    for family in ('LA','LB','LC'):
        for i in range(501):vocab[f'<{family}_{i:03d}>']=len(vocab)
    for family in ('AA','AB','AG'):
        for i in range(1,180):vocab[f'<{family}_{i:03d}>']=len(vocab)
    for family in 'XYZ':
        for i in range(101):vocab[f'<{family}_{i:03d}>']=len(vocab)
    vocab['<E_O>']=len(vocab);vocab['<N_002>']=len(vocab)
    tokenizer=SimpleNamespace(get_vocab=lambda:vocab);schema=AxisTrainingSchema(tokenizer,mask_id=99999)
    body=[vocab['<N_002>']]+[vocab[f'<{a}_050>'] for a in ('LA','LB','LC')]+[vocab[f'<{a}_090>'] for a in ('AA','AB','AG')]
    body += [vocab['<E_O>'],99999,99999,99999]*2
    torch.manual_seed(91);head=PeriodicAxisHead(6,width=8,harmonics=2)
    with torch.no_grad():head.output.bias[0]=.5
    logits=torch.randn(1,len(body),len(vocab));hidden=torch.randn(1,len(body),6)
    for p in [8,12]:
        saved=logits[0,p,schema.axis_tokens[0]].clone();logits[0,p].fill_(torch.finfo(logits.dtype).min)
        logits[0,p,schema.axis_tokens[0]]=saved
    args={'current_tokens':torch.tensor([body]),'prompt_length':0,'gen_length':len(body),
          'temperature':.7,'remasking':'low_confidence','base_seeds':[17],'semantic_group':2,'step_in_group':0}
    return tokenizer,schema,head,logits,hidden,args


def test_default_policy_preserves_legacy_and_joint_policy_keeps_same_candidate_draw():
    tok,schema,head,logits,hidden,args=fixture();results=[];samplers=[]
    for policy in [None,'legacy_unary','joint_marginal']:
        kw={} if policy is None else {'confidence_policy':policy}
        sampler=AxisCandidateSampler(head,tok,{'plan_state':{'N':2}},schema.constraints,**kw)
        sampler.begin_attempt([[8,12]],17)
        with torch.no_grad():result=sampler(logits,hidden=hidden,group_positions=[8,12],mask_id=99999,**args)
        samplers.append(sampler);results.append(result)
    assert all(torch.equal(x,y) for x,y in zip(results[0],results[1]))
    assert torch.equal(results[0][0],results[2][0])
    assert samplers[0].report()['same_legacy_commit_rule']
    assert not samplers[2].report()['same_legacy_commit_rule']
    assert samplers[2].report()['same_commit_counts_and_schedule']
    pos=torch.tensor([8,12]);sample=torch.tensor(samplers[2].events[-1]['joint_candidate_bins'])
    unary=collapse_alias_logits(logits[0,pos][:,schema.axis_tokens[0]],list(range(101)))
    law=head.distribution(unary,hidden[0,pos],[8,8],torch.eye(3)*5,
        torch.full((2,3),torch.nan),torch.zeros((2,3),dtype=torch.bool),axis=0,temperature=.7,
        visible_values=[0,0],visible_mask=[False,False])
    expected=law.marginals().gather(1,sample[:,None]).squeeze(1).float()
    assert torch.allclose(results[2][1][0,pos],expected,atol=1e-5)


def test_unregistered_confidence_policy_is_rejected():
    tok,schema,head,*_=fixture()
    with pytest.raises(ValueError,match='confidence'):
        AxisCandidateSampler(head,tok,{'plan_state':{'N':2}},schema.constraints,confidence_policy='oracle')
