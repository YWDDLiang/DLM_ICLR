"""Observable periodic geometry for a proposed coordinate commit.

Only already known atoms and the proposed site's now-complete coordinates are
used. Missing coordinates are never imputed or sent to a physical evaluator.
"""
import numpy as np
import torch
from .common import read_json


def visible_geometry_features(lattice,coords,known,site):
    from pymatgen.core import Lattice
    lattice=np.asarray(lattice,dtype=float);coords=np.asarray(coords,dtype=float)
    known=np.asarray(known,dtype=bool);n=len(coords)
    indices=[i for i in range(n) if i!=site and known[i].all()]
    basic=[float(np.log1p(abs(np.linalg.det(lattice))/n)),len(indices)/max(1,n-1)]
    if not known[site].all() or not indices:return np.asarray(basic+[0.]*6)
    # pymatgen provides the same periodic distance implementation used elsewhere.
    distance=Lattice(lattice).get_all_distances(coords[site:site+1],coords[indices])[0]
    if not np.isfinite(distance).all():raise ValueError('Nonfinite visible periodic distance')
    return np.asarray(basic+[1.,np.log1p(distance.min()),np.log1p(distance.mean()),
        np.log1p(np.exp(-distance).sum()),np.log1p(((distance+.1)**-6).sum()),np.log1p(distance.max())])


def prefix_geometry_features(body,action,constraints):
    from dlm_iclr._core.llada_generation import _lattice_matrix_from_token_ids
    from dlm_iclr._core.fixed_slot import MASK_TOKEN_ID
    values=body.detach().cpu().tolist() if torch.is_tensor(body) else list(body)
    n=(len(values)-7)//4;pos,token=action;site=(pos-8)//4
    if not 8<=pos<len(values) or (pos-8)%4>2:raise ValueError('Expected coordinate action')
    lattice=_lattice_matrix_from_token_ids(torch.tensor(values),prompt_length=0,constraints=constraints)
    if lattice is None:raise ValueError('Prefix has no complete lattice')
    values[pos]=int(token)
    coords=np.full((n,3),np.nan);known=np.zeros((n,3),dtype=bool)
    for i in range(n):
        for a,name in enumerate('XYZ'):
            value=values[8+4*i+a]
            if value!=MASK_TOKEN_ID:
                coords[i,a]=(constraints['coord_token_to_bin'][name][value]%100)/100
                known[i,a]=True
    return visible_geometry_features(lattice.numpy(),coords,known,site)


def augment_branch_rows(rows,base_folder,constraints):
    states={}
    for path in base_folder.glob('[0-9]*.json'):
        generated=read_json(path);source=generated['record']['source_id']
        for probe in generated['process_probes']:
            states[(source,probe['state_key'])]=probe['state']['body']
    result=[]
    for row in rows:
        body=states[(row['source_id'],row['state_key'])]
        geometry=prefix_geometry_features(body,row['action'],constraints)
        result.append(dict(row,features=list(row['features'])+geometry.tolist(),
                           geometry_source='known_prefix_atoms_and_complete_lattice_only'))
    return result
