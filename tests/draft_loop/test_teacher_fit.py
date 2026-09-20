from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dlm_iclr.draft_loop.learning import TypedVocabulary
from dlm_iclr.draft_loop.teacher_fit import (
    phase_positions,source_phase,teacher_view,train_existing_teachers,fitting_progress_gate,
)
import importlib.util

_spec=importlib.util.spec_from_file_location('teacher_fit_test_fixture',Path(__file__).with_name('test_state_replay.py'))
_fixture=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_fixture)
fixture,TinyModel=_fixture.fixture,_fixture.TinyModel


def data():
    plan,generated,tokenizer,mask=fixture()
    row={'source_id':'teacher','body_prompt':'plan','plan_state':plan['plan_state'],
         'body_token_ids':generated['record']['body_token_ids']}
    return row,tokenizer,mask


def test_exact_source_phase_coverage_and_resume_cursor():
    rows=[{'source_id':str(i)} for i in range(16)]
    draws=[source_phase(rows,i,17) for i in range(2048)]
    assert set(Counter((i,p) for i,p,_ in draws).values())=={32}
    assert draws[701:]==[source_phase(rows,i,17) for i in range(701,2048)]


@pytest.mark.parametrize('phase',range(4))
def test_views_keep_teacher_lattice_and_hide_future_without_mutation(phase):
    row,tok,mask=data();saved=row['body_token_ids'].copy();vocab=TypedVocabulary(tok)
    for repeat in [0,1,2,3]:
        view=teacher_view(row,phase,repeat,19,mask,vocab);groups=phase_positions(row['plan_state']['N'])
        assert all(view['input_body'][p]==mask for p in view['positions'])
        assert all(view['input_body'][p]==mask for g in groups[phase+1:] for p in g)
        assert all(view['input_body'][p]==vocab.canonical(saved)[p] for g in groups[:phase] for p in g)
        assert view['input_body'][0]==saved[0]
    assert row['body_token_ids']==saved


def test_near_token_fit_allows_physical_check_without_claiming_scientific_success():
    before={'fields':{k:{'NLL':1.,'top1':.8,'within_one_bin':.96} for k in ['lattice_lengths','coordinates']}}
    gate=fitting_progress_gate(before,before)
    assert gate['ready_for_bounded_generation_check']
    assert not gate['scientific_effect_established']


@pytest.mark.parametrize('conditioned',[False,True,'fit_focused'])
def test_real_update_preserves_frozen_tables_and_exact_resume(monkeypatch,tmp_path,conditioned):
    row,tokenizer,mask=data()
    import dlm_iclr.runtime.models as models
    import dlm_iclr._core.fixed_slot as fixed_slot
    monkeypatch.setattr(fixed_slot,'MASK_TOKEN_ID',mask)
    instances=[]
    class Adapter(TinyModel):
        def __init__(self):
            torch.manual_seed(71);super().__init__()
            self.lora_A=torch.nn.Parameter(torch.zeros(41,6))
            self.peft_config={'default':SimpleNamespace(modules_to_save=['embedding','output'])}
        def enable_input_require_grads(self):pass
        def forward(self,ids,attention_mask):
            out=super().forward(ids,attention_mask)
            out.logits=out.logits+self.embedding(ids).sum(1,keepdim=True)@self.lora_A.T
            return out
        def save_pretrained(self,folder,**kw):
            Path(folder).mkdir(parents=True,exist_ok=True);torch.save(self.state_dict(),Path(folder)/'weights.pt')
    def loader(*args,**kwargs):
        instance=Adapter();instances.append(instance);return instance,tokenizer
    monkeypatch.setattr(models,'load_model_and_tokenizer',loader)
    tokenizer.save_pretrained=lambda folder:None
    recipe={'updates':8,'effective_batch_size':4,'micro_batch_size':2,'learning_rate':1e-3,
            'temperature':.7,'max_length':128,'save_every':4,'evaluate_every':4}
    kwargs={}
    if conditioned:
        from dlm_iclr.draft_loop.common import digest
        from dlm_iclr.draft_loop.state_replay import CommitView
        v=teacher_view(row,1,0,7,mask,TypedVocabulary(tokenizer))
        view=CommitView(row['source_id'],'candidate',row['body_prompt'],tuple(v['input_body']),
            tuple(v['positions']),tuple(v['target'][p] for p in v['positions']),0,2,0,.7,mask)
        kwargs={'feedback':[{'positive_view':view.to_dict(),'input_prompt_key':digest(view.prompt),
                  'input_prefix_key':digest(view.input_body),'weight':.5}], 'replay':[row]}
        if conditioned=='fit_focused':
            recipe['microbatch_mixture']=['teacher','feedback','teacher','teacher']
            recipe['fit_selection_teacher_weight']=.75
            kwargs.pop('replay')
    initial=Adapter().state_dict()
    report=train_existing_teachers('base','adapter',[row],tmp_path,recipe,seed=7,device='cpu',**kwargs)
    final=torch.load(tmp_path/'last/weights.pt',weights_only=True)
    assert not torch.equal(final['lora_A'],initial['lora_A'])
    for name,value in initial.items():
        if name!='lora_A':assert torch.equal(final[name],value)
    assert sum(x['views'] for x in report['exposure'])==(24 if conditioned is True else 32)
    if conditioned is True:assert report['mixture_views_total']=={'feedback':16,'teacher':8,'replay':8}
    if conditioned=='fit_focused':assert report['mixture_views_total']=={'teacher':24,'feedback':8}
    resumed=train_existing_teachers('base','adapter',[row],tmp_path,recipe,seed=7,device='cpu',**kwargs)
    assert resumed['exposure']==report['exposure']
    again=torch.load(tmp_path/'last/weights.pt',weights_only=True)
    assert all(torch.equal(final[k],again[k]) for k in final)
    with pytest.raises(ValueError,match='identity changed'):
        train_existing_teachers('base','adapter',[row],tmp_path,{**recipe,'learning_rate':2e-3},seed=7,device='cpu',**kwargs)
