from copy import deepcopy
import numpy as np
from dlm_iclr.draft_loop.advantage import fit_completion_advantage
from dlm_iclr.draft_loop.verifier import ProcessVerifier,positive_advantage_index
from dlm_iclr.draft_loop.settings import DEFAULTS


def test_paired_gain_learns_context_dependent_action_and_cancels_source_offsets(tmp_path):
    settings=deepcopy(DEFAULTS['verifier']);settings['projection_width']=2
    rows=[]
    for i in range(80):
        context=1. if i%2 else -1.
        for action in [0.,1.]:
            feature=[context,0.,0.,0.,.2,0.,0.,0.,1.,.9,action,0.,-.1]
            rows.append({'source_id':str(i),'state_key':str(i),'features':feature,
                'target':-5.+i/20+context*action,
                'outcome_source':'unrefined_draft' if action==0 else 'exact_replay_counterfactual_unrefined_draft'})
    model=fit_completion_advantage(rows,tmp_path/'q.json','binding',settings)
    assert model['validation']['deployable']
    assert not set(model['train_sources'])&set(model['validation_sources'])
    q=ProcessVerifier(model,'binding')
    for i in [0,1]:
        group=rows[i*2:i*2+2];pred=q.predict([r['features'] for r in group])
        assert positive_advantage_index(pred)==i
    shifted=[dict(r,target=r['target']+float(r['source_id'])**2) for r in rows]
    other=fit_completion_advantage(shifted,tmp_path/'shifted.json','binding',settings)
    np.testing.assert_allclose(model['weight'],other['weight'])
