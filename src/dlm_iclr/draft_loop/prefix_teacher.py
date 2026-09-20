"""Physically unlabelled completion proposals for recorded student prefixes.

Observed fields never change. A periodic teacher registration only proposes
unresolved coordinates; it does not certify that the resulting crystal is good.
Every resulting exact token structure must be measured before use as feedback.
"""
from collections import Counter
from itertools import product

import numpy as np
from scipy.optimize import linear_sum_assignment


def register_teacher(template, template_species, observed, observed_species, known, lengths):
    template=np.asarray(template,float);observed=np.asarray(observed,float);known=np.asarray(known,bool)
    ts=np.asarray(template_species);zs=np.asarray(observed_species);lengths=np.asarray(lengths,float)
    if template.shape!=observed.shape or known.shape!=observed.shape or template.shape[1]!=3:
        raise ValueError('Consistent N by 3 coordinates and visibility required')
    if Counter(ts)!=Counter(zs) or not np.isfinite(template).all() or not np.isfinite(observed[known]).all():
        raise ValueError('Species inventory or observed coordinates invalid')
    if lengths.shape!=(3,) or not np.isfinite(lengths).all() or (lengths<=0).any():
        raise ValueError('Actual student cell lengths required')
    proposals=[]
    for axis in range(3):
        visible=np.flatnonzero(known[:,axis])
        if not len(visible):proposals.append([0.]);continue
        anchor=min(visible,key=lambda i:(int((ts==zs[i]).sum()),int(i)))
        shifts=(observed[anchor,axis]-template[ts==zs[anchor],axis])%1
        proposals.append(sorted(set(np.round(shifts*100).astype(int)%100/100)))
    starts=list(product(*proposals))
    if len(starts)>64:starts=[starts[i] for i in np.linspace(0,len(starts)-1,64,dtype=int)]
    best=None;visible_values=np.where(known,observed,0.)
    for start in starts:
        shift=np.asarray(start,float)
        for _ in range(4):
            delta=(visible_values[:,None,:]-template[None,:,:]-shift+.5)%1-.5
            costs=((delta*lengths)**2*known[:,None,:]).sum(-1)
            costs=np.where(zs[:,None]==ts[None,:],costs,1e30)
            rows,cols=linear_sum_assignment(costs);score=float(costs[rows,cols].sum())
            if best is None or score<best[0]-1e-12:best=(score,cols.copy(),shift.copy())
            updated=shift.copy()
            for axis in range(3):
                visible=np.flatnonzero(known[:,axis])
                if len(visible):
                    residual=observed[visible,axis]-template[cols[visible],axis]
                    angle=np.angle(np.exp(2j*np.pi*residual).mean())/(2*np.pi)
                    updated[axis]=(np.rint(angle*100)%100)/100
            if np.array_equal(updated,shift):break
            shift=updated
    score,order,shift=best
    return (template[order]+shift)%1,{'atom_mapping':order.tolist(),'fractional_translation':shift.tolist(),
        'known_component_weighted_error_A2':score,'known_scalar_coordinates':int(known.sum()),
        'registration_is_approximate_not_physical_verification':True}


def propose_completion(view, teacher, tables):
    body=list(view.input_body);n=(len(body)-7)//4;target=teacher['body_token_ids']
    if len(target)!=len(body) or n<1 or any(body[p]==view.mask_id for p in range(7)):
        raise ValueError('A complete observed lattice and matching teacher are required')
    reverse={family:{token:value for value,token in mapping.items()} for family,mapping in tables.items()}
    template=[];observed=[];known=[]
    for i in range(n):
        template.append([reverse[a][target[8+4*i+j]]%100/100 for j,a in enumerate('XYZ')])
        visible=[body[8+4*i+j]!=view.mask_id for j in range(3)];known.append(visible)
        observed.append([reverse[a][body[8+4*i+j]]%100/100 if visible[j] else np.nan for j,a in enumerate('XYZ')])
    lengths=[reverse[a][body[p]]*.1 for p,a in enumerate(('LA','LB','LC'),1)]
    coordinates,registration=register_teacher(template,[target[7+4*i] for i in range(n)],
        observed,[body[7+4*i] for i in range(n)],known,lengths)
    result=body.copy()
    for i in range(n):
        for j,a in enumerate('XYZ'):
            p=8+4*i+j
            if result[p]==view.mask_id:result[p]=tables[a][int(np.rint(coordinates[i,j]*100))%100]
    if view.mask_id in result or any(a!=view.mask_id and a!=b for a,b in zip(body,result,strict=True)):
        raise ValueError('Completion changed an observed field or left an unresolved field')
    return result,registration
