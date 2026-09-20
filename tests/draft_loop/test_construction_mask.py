from dlm_iclr.draft_loop.learning import mask_view,construction_patterns


def test_constructor_masks_hide_future_phases_and_preserve_composition():
    body=list(range(19));mask=999
    groups=[list(range(1,7)),[8,12,16],[9,13,17],[10,14,18]]
    phases=set()
    for seed in range(100):
        noisy,selected,_=mask_view(body,seed,mask,.15,.85,strategy='construction')
        phase=next(i for i,g in enumerate(groups) if set(selected)<=set(g));phases.add(phase)
        assert selected and all(noisy[p]==mask for p in selected)
        assert all(noisy[p]==mask for g in groups[phase+1:] for p in g)
        assert all(noisy[p]==body[p] for g in groups[:phase] for p in g)
        assert all(noisy[p]==body[p] for p in [0,7,11,15])
        other,selected2,_=mask_view([x+100 for x in body],seed,mask,.15,.85,strategy='construction')
        assert selected==selected2
        assert [i for i,x in enumerate(noisy) if x==mask]==[i for i,x in enumerate(other) if x==mask]
    assert phases=={0,1,2,3}


def test_observed_coordinate_mask_shapes_are_replayed():
    body=list(range(19));mask=999
    patterns=[{'axis':0,'active_positions':[12,16],'masked_positions':[12,16,9,13,17,10,14,18]}]
    hits=0
    for seed in range(100):
        noisy,selected,_=mask_view(body,seed,mask,.15,.85,strategy='construction',patterns=patterns)
        if selected==[12,16]:
            hits+=1
            assert noisy[8]==body[8] and noisy[12]==mask and noisy[18]==mask
    assert hits>0


def test_extracts_cell_and_coordinate_shapes_but_skips_recovery_contexts():
    body=list(range(19));mask=999
    for p in [12,16,9,13,17,10,14,18]:body[p]=mask
    generated={'trace':{'periodic_axis':{'events':[{'axis':0,'input_body':body,'active_positions':[12,16]}]},
        'construction_geometry':{'events':[{'stage':'lattice','active_positions':[2,6]}]},
        'construction_recovery':{'recoveries_used':0}}}
    patterns=construction_patterns(generated,mask)
    cell=next(p for p in patterns if p['axis']==-1)
    assert set(cell['masked_positions'])=={2,6,8,12,16,9,13,17,10,14,18}
    for seed in range(100):
        noisy,selected,_=mask_view(list(range(19)),seed,mask,.15,.85,strategy='construction',patterns=patterns)
        if any(p<7 for p in selected):assert selected==[2,6] and noisy[1]==1
    generated['trace']['construction_recovery']['recoveries_used']=1
    assert all(p['axis']!=-1 for p in construction_patterns(generated,mask))
    body[9]=9  # A future coordinate visible during a recovery is not a first-pass view.
    assert construction_patterns(generated,mask)==[]


def test_lattice_reference_view_uses_source_prefix_without_future_coordinates():
    body=list(range(19));mask=999
    pattern={'axis':-1,'active_positions':[2,6],'masked_positions':[2,6,8,9,10,12,13,14,16,17,18]}
    noisy,selected,_=mask_view(body,4,mask,.15,.85,strategy='lattice_anchor',patterns=[pattern])
    assert selected==[2,6]
    assert all(noisy[p]==mask for p in pattern['masked_positions'])
    assert all(noisy[p]==body[p] for p in [0,1,3,4,5,7,11,15])
    noisy,selected,_=mask_view(body,4,mask,.15,.85,strategy='lattice_anchor')
    assert selected==list(range(1,7))
