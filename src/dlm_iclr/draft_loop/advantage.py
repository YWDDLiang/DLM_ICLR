"""Within-prefix physical advantage from completed, matched continuations.

Subtracting two branch labels removes source difficulty. A small fixed bilinear
feature map allows the preferred coordinate action to depend on its context.
No physical evaluator is applied to an incomplete structure.
"""
from collections import defaultdict
import numpy as np
from .common import write_json


def action_context_features(features, projection_width, *, geometry=False):
    x=np.asarray(features,dtype=float)
    width=int(projection_width)
    size=2*width+9
    if x.shape[-1]!=size+(8 if geometry else 0):raise ValueError('Unexpected prefix feature layout')
    base=x[...,:size]
    k=min(width,8)
    context=np.concatenate((base[...,:k],base[...,width:width+k]),axis=-1)
    # Last nine values: size, site, axis one-hot, known fraction, sin, cos, log p.
    action=base[...,[-8,-3,-2,-1]]
    interactions=(context[..., :,None]*action[...,None,:]).reshape(*x.shape[:-1],-1)
    return np.concatenate((x,interactions),axis=-1)


def fit_completion_advantage(rows,output,binding,settings,*,seed=17):
    groups=defaultdict(list)
    for row in rows:
        if row.get('target') is not None:
            groups[(row['source_id'],row['state_key'])].append(row)
    pairs=[]
    for (source,state),group in groups.items():
        refs=[r for r in group if r.get('outcome_source')=='unrefined_draft']
        if len(refs)!=1:continue
        ref=refs[0]
        for row in group:
            if row is ref:continue
            x=action_context_features([ref['features'],row['features']],settings['projection_width'],
                                      geometry=settings.get('geometry_features',False))
            pairs.append({'source_id':source,'state_key':state,'difference':x[1]-x[0],
                'gain':float(row['target']-ref['target'])})
    sources=sorted({p['source_id'] for p in pairs})
    rng=np.random.default_rng(seed);rng.shuffle(sources)
    cut=max(1,int(len(sources)*.8));train_sources=sources[:cut];val_sources=sources[cut:]
    saved={'schema':'prefix_completion_advantage_v1','binding':binding,
        'target':'paired_raw_completed_draft_quality_gain_not_F_endpoint',
        'prediction_kind':'within_prefix_advantage',
        'feature_map':{'kind':('action_context_geometry_v1' if settings.get('geometry_features',False)
                              else 'action_context_bilinear_v1'),'projection_width':settings['projection_width']},
        'settings':settings,'train_sources':train_sources,'validation_sources':val_sources,
        'pairs':len(pairs),'same_prefix_and_same_completion_policy':True}
    if len(train_sources)<settings['min_train_sources'] or len(val_sources)<settings['min_validation_sources']:
        saved['validation']={'deployable':False,'reason':'insufficient_independent_paired_sources'}
        write_json(output,saved);return saved
    train_set=set(train_sources);val_set=set(val_sources)
    train=[p for p in pairs if p['source_id'] in train_set]
    val=[p for p in pairs if p['source_id'] in val_set]
    counts={s:sum(p['source_id']==s for p in train) for s in train_sources}
    weights=np.asarray([1/counts[p['source_id']] for p in train]);weights/=weights.sum()
    dx=np.asarray([p['difference'] for p in train]);dy=np.asarray([p['gain'] for p in train])
    scale=np.sqrt((dx**2*weights[:,None]).sum(0)).clip(min=.05)
    z=dx/scale
    coefficient=np.linalg.solve(z.T@(z*weights[:,None])+settings['ridge']*np.eye(z.shape[1]),
        z.T@(weights*dy))
    pred=np.asarray([p['difference'] for p in val])/scale@coefficient
    truth=np.asarray([p['gain'] for p in val]);eligible=np.abs(truth)>=.05
    accuracy=float(np.mean(np.where(pred[eligible]==0,.5,(pred[eligible]*truth[eligible]>0).astype(float)))) if eligible.any() else None
    mse=float(np.mean((pred-truth)**2));zero_mse=float(np.mean(truth**2))
    margin=float(np.quantile(np.maximum(0,pred-truth),.75))
    deployable=accuracy is not None and accuracy>=settings['min_pair_accuracy'] and mse<=max(1e-8,1.2*zero_mse)
    saved.update(mean=np.zeros(len(scale)).tolist(),scale=scale.tolist(),weight=coefficient.tolist(),bias=0.,
        validation={'deployable':bool(deployable),'pair_accuracy':accuracy,'comparisons':int(eligible.sum()),
            'mse':mse,'zero_gain_mse':zero_mse,'advantage_margin':margin,
            'margin_source':'held_out_source_gain_overestimate_q75_not_a_guarantee'})
    write_json(output,saved);return saved
