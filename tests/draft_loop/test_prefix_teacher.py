import numpy as np
import pytest

from dlm_iclr.draft_loop.prefix_teacher import register_teacher,propose_completion
from dlm_iclr.draft_loop.state_replay import CommitView


def test_periodic_translation_and_same_species_assignment_respect_known_geometry():
    teacher=np.asarray([[.1,.2,.3],[.6,.7,.8],[.2,.8,.4]])
    species=[1,1,2];order=[1,0,2];observed=(teacher[order]+[.2,.3,.4])%1
    actual,report=register_teacher(teacher,species,observed,species,np.ones((3,3),bool),[5,6,7])
    assert np.max(np.abs((actual-observed+.5)%1-.5))<1e-9
    assert report['known_component_weighted_error_A2']<1e-12


def test_unobserved_coordinates_are_not_read_and_unknown_axes_are_not_fabricated():
    teacher=np.asarray([[.2,.3,.4],[.7,.8,.9]])
    observed=np.full((2,3),np.nan);observed[0,0]=.3;known=np.isfinite(observed)
    actual,report=register_teacher(teacher,[1,2],observed,[1,2],known,[5,5,5])
    assert np.allclose(actual[:,0],[.3,.8])
    assert np.allclose(actual[:,1:],teacher[:,1:])
    assert report['known_scalar_coordinates']==1


def test_composition_mismatch_is_rejected():
    with pytest.raises(ValueError,match='inventory'):
        register_teacher(np.zeros((1,3)),[1],np.zeros((1,3)),[2],np.zeros((1,3),bool),[5,5,5])


def test_proposal_preserves_prompt_lattice_and_observed_tokens():
    tables={a:{i:1000*j+i for i in range(101)} for j,a in enumerate(('LA','LB','LC','AA','AB','AG','X','Y','Z'),1)}
    mask=99999
    teacher=[21]+[tables[a][50] for a in ('LA','LB','LC')]+[tables[a][90] for a in ('AA','AB','AG')]
    teacher += [31,tables['X'][20],tables['Y'][30],tables['Z'][40],32,tables['X'][70],tables['Y'][80],tables['Z'][90]]
    body=teacher.copy();body[1]=tables['LA'][60];body[8]=tables['X'][30]
    for p in [9,10,12,13,14]:body[p]=mask
    view=CommitView('source','candidate','unaltered original prompt',tuple(body),(12,),(tables['X'][10],),0,2,0,.7,mask)
    result,reg=propose_completion(view,{'body_token_ids':teacher},tables)
    assert result[1]==tables['LA'][60] and result[8]==tables['X'][30]
    assert result[12]==tables['X'][80]
    assert all(a==mask or a==b for a,b in zip(body,result))
    assert view.prompt=='unaltered original prompt' and mask not in result
    assert reg['registration_is_approximate_not_physical_verification']
