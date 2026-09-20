import numpy as np
import pytest

from dlm_iclr.draft_loop.teacher_registration import align_teacher_to_anchor


def tables():
    return {f: {i: offset+i for i in range(size)} for f,offset,size in [
        ('LA',1000,501),('LB',2000,501),('LC',3000,501),
        ('AA',4000,181),('AB',5000,181),('AG',6000,181),
        ('X',7000,101),('Y',8000,101),('Z',9000,101)]}


def body(coords, species=(11,11,12,13)):
    result = [104,1054,2054,3054,4060,5060,6060]
    for z, xyz in zip(species,coords,strict=True):
        result.extend([z,7000+xyz[0],8000+xyz[1],9000+xyz[2]])
    return result


def test_same_geometry_different_origin_and_species_order_returns_anchor_coords():
    coords = np.asarray([[25,25,25],[75,75,75],[50,50,50],[0,0,0]])
    original = body(((coords[[1,0,2,3]]+np.array([54,39,47]))%100).tolist())
    target = body(coords.tolist())
    aligned, info = align_teacher_to_anchor(original,target,tables())
    assert aligned == target
    assert info['anchor_coordinate_RMS_A_in_teacher_cell'] < 1e-9
    assert info['exact_periodic_relative_geometry_preserved']


def test_teacher_cell_preserved_when_anchor_cell_differs():
    original = body([[0,0,0],[50,50,50],[25,25,25],[75,75,75]])
    anchor = body([[1,2,3],[51,52,53],[26,27,28],[76,77,78]])
    anchor[1:7] = [1060,2060,3060,4090,5090,6090]
    aligned, _ = align_teacher_to_anchor(original,anchor,tables())
    assert aligned[:7] == original[:7]
    assert aligned[7:] == anchor[7:]


def test_periodic_alias_and_idempotence():
    original = body([[100,0,0],[50,50,50],[25,25,25],[75,75,75]])
    result, _ = align_teacher_to_anchor(original,original,tables())
    assert result[8] == 7000
    assert align_teacher_to_anchor(result,result,tables())[0] == result


def test_chemical_or_incomplete_target_rejected():
    a = body([[0,0,0],[50,50,50],[25,25,25],[75,75,75]])
    b = a.copy(); b[7] = 99
    with pytest.raises(ValueError, match='inventories'): align_teacher_to_anchor(a,b,tables())
    b = a.copy(); b[8] = -1
    with pytest.raises(ValueError, match='typed'): align_teacher_to_anchor(a,b,tables())
