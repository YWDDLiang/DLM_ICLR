from copy import deepcopy
import pytest

from dlm_iclr.draft_loop.common import digest,write_json
from dlm_iclr.draft_loop.inference_protocol import bind_inference_protocol
from dlm_iclr.feedback import validate_material,settings_from_profile


def material():
    measurement={'record_key':'teacher-key','protocol_key':'same-physics','composition':'Li:1',
        'status':'singlepoint','verified':False,'raw_energy':-1.,'force_rms':.01,'force_max':.02,
        'stress_max':.1,'raw_hull':-.01,'terminal_hull':None,'actual_steps':None}
    teacher={'source_id':'train1','source_split':'train','body_prompt':'same original Plan',
        'registration':{'physically_reverified':True},'measurement':measurement,'exact_record_key':'teacher-key'}
    body=[1,2,3,4,5,6,7,8,99,10,11]
    view={'source_id':'train1','prompt':teacher['body_prompt'],'input_body':body,'tokens':[9],
          'positions':[8],'axis':0,'mask_id':99}
    parent={**measurement,'record_key':'raw-key','force_rms':1.}
    from dlm_iclr.draft_loop.quality import Measurement,raw_utility
    gain=raw_utility(Measurement.from_dict(measurement))-raw_utility(Measurement.from_dict(parent))
    feedback={'source_id':'train1','positive_view':view,'input_prompt_key':digest(view['prompt']),
        'input_prefix_key':digest(body),'proposal_record_key':'teacher-key','measurement':measurement,
        'parent_measurement':parent,'raw_utility_gain':gain,'weight':gain/(1+gain)}
    rows=[teacher];states=[feedback]
    review={'eligible':True,'teachers_digest':digest(rows),'feedback_digest':digest(states)}
    return rows,states,review


def test_singlepoint_supervision_is_accepted_without_forging_relaxed_status():
    teachers,feedback,review=material()
    assert validate_material(teachers,feedback,review)['sources']==1
    assert teachers[0]['measurement']['verified'] is False


@pytest.mark.parametrize('mutation',['prompt','gain','unknown'])
def test_reviewed_artifacts_still_require_correct_prompt_and_physical_comparison(mutation):
    teachers,feedback,review=material()
    if mutation=='prompt':
        feedback[0]['positive_view']['prompt']='different Plan'
        feedback[0]['input_prompt_key']=digest('different Plan')
    elif mutation=='gain':feedback[0]['raw_utility_gain']+=1
    else:feedback[0]['measurement']['status']='unknown'
    review.update(teachers_digest=digest(teachers),feedback_digest=digest(feedback))
    with pytest.raises(ValueError):validate_material(teachers,feedback,review)


def test_frozen_temperature_and_ordered_protocol_survive_manifest_transfer():
    config={'c1':{'temperature':.7},'diffusion':{'reduction_protocol':'legacy_scatter'}};saved=deepcopy(config)
    assets={'inference_protocol':{'body_temperature':.2,'F_reduction_protocol':'ordered_csr_v1'}}
    effective=bind_inference_protocol(config,assets)
    assert effective['c1']['temperature']==.2
    assert effective['diffusion']['reduction_protocol']=='ordered_csr_v1'
    assert config==saved
    assert bind_inference_protocol(config,{}) is config


def test_profile_materializes_runtime_and_shared_limits():
    profile={'runtime':{'devices':['cuda:0'],'generation_workers_per_device':3},'direct_workers':32,'F_steps':800}
    limits={'max_wall_seconds':100,'max_drafts':64}
    settings=settings_from_profile(profile,limits)
    assert settings['runtime']['devices']==['cuda:0'] and settings['runtime']['generation_workers_per_device']==3
    assert settings['evaluation']['direct_workers']==32 and settings['teacher_steps']==800
    assert settings['budget']==limits


def test_train_entry_checks_real_token_key_before_calling_trainer(tmp_path,monkeypatch):
    """Exercise entry wiring with the real physical key, without loading weights."""
    import dlm_iclr.feedback as entry
    import dlm_iclr.runtime.config as config_module
    import dlm_iclr.draft_loop.teacher_fit as fitting
    from dlm_iclr.draft_loop.common import write_rows
    from dlm_iclr.evaluation.physics import record_key
    from transformers import AutoTokenizer
    teachers,feedback,review=material()
    tokens=list(range(1,12));vocabulary={f'<t{i}>':i for i in tokens}
    real_key=record_key({'success':True,'body':''.join(vocabulary)})
    teachers[0].update(body_token_ids=tokens,exact_record_key=real_key)
    teachers[0]['measurement']['record_key']=real_key
    feedback[0]['proposal_record_key']=real_key
    feedback[0]['measurement']['record_key']=real_key
    review.update(teachers_digest=digest(teachers),feedback_digest=digest(feedback))
    class Tokenizer:
        def get_vocab(self):return vocabulary
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**k:Tokenizer())
    monkeypatch.setattr(config_module,'load',lambda p:{})
    monkeypatch.setattr(entry,'model_identity',lambda *args:'fixed-model-binding')
    calls=[]
    def train(base,adapter,rows,output,recipe,**kwargs):
        calls.append(rows)
        return {'selected_checkpoint':str(output/'best'),'best_step':1}
    monkeypatch.setattr(fitting,'train_existing_teachers',train)
    values={'config':{},'assets':{'base':'base','draft':'draft','head':'head','binding':'fixed-model-binding'},
        'recipe':{'schema':'registered_feedback_recipe_v1','training':{},'seed':1,
            'inference_protocol':{'body_temperature':.2,'F_reduction_protocol':'ordered_csr_v1'}},
        'budget-limits':{},'material-review':review}
    for name,value in values.items():write_json(tmp_path/f'{name}.json',value)
    write_rows(tmp_path/'teachers.jsonl',teachers);write_rows(tmp_path/'prefix-feedback.jsonl',feedback)
    argv=['train']
    for name in values:argv.extend(['--'+name,str(tmp_path/f'{name}.json')])
    for name in ['teachers','prefix-feedback']:argv.extend(['--'+name,str(tmp_path/f'{name}.jsonl')])
    argv.extend(['--budget-ledger',str(tmp_path/'ledger.json'),'--output',str(tmp_path/'trained')])
    entry.main(argv)
    assert calls==[teachers]
    assert (tmp_path/'trained/assets.json').exists()
    teachers[0]['body_token_ids'][0]=2
    review.update(teachers_digest=digest(teachers))
    write_json(tmp_path/'material-review.json',review);write_rows(tmp_path/'teachers.jsonl',teachers)
    argv[-1]=str(tmp_path/'mismatched')
    with pytest.raises(ValueError,match='token/physics identity differs'):entry.main(argv)
    assert len(calls)==1


def test_changed_temperature_or_unversioned_cache_cannot_reuse_drafts(tmp_path):
    from dlm_iclr.draft_loop.backend import sample_drafts
    config={'c1':{'temperature':.7},'diffusion':{}}
    assets={'inference_protocol':{'body_temperature':.2,'F_reduction_protocol':'ordered_csr_v1'}}
    plans=[{'source_id':'one'}];settings={'runtime':{}}
    write_json(tmp_path/'000000.json',{'record':{'source_id':'one'}})
    with pytest.raises(ValueError,match='unversioned'):
        sample_drafts(config,assets,plans,tmp_path,settings,None)
    write_json(tmp_path/'inputs.json',{'assets':assets,'plan_sources':['one'],'ordered_plans':digest(plans),'guide':None,'forced':None,'collect':False})
    assert sample_drafts(config,assets,plans,tmp_path,settings,None)[0]['record']['source_id']=='one'
    other=deepcopy(assets);other['inference_protocol']['body_temperature']=.7
    with pytest.raises(ValueError,match='different inputs'):
        sample_drafts(config,other,plans,tmp_path,settings,None)
    with pytest.raises(ValueError,match='different inputs'):
        sample_drafts(config,assets,[{'source_id':'one','body_noise_seed':123}],tmp_path,settings,None)
