"""End-to-end orchestration tests with declared fake GPU/physics backends.

These tests verify state transitions and restart/stop behavior, not crystal quality.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import sys
import types
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from dlm_iclr.draft_loop import workflow as wf
from dlm_iclr.draft_loop import learning
from dlm_iclr.draft_loop import step_budget
from dlm_iclr.draft_loop.common import write_rows,write_json,read_json,digest
from dlm_iclr.draft_loop.settings import DEFAULTS
from dlm_iclr.draft_loop.quality import Measurement


def make_plan(s):
    return {"source_id":s,"ordinal":0,"original_ordinal":0,"body_prompt":"P\n","body_eligible":True,
            "body_noise_seed":17,"refiner_noise_seed":19,"provenance":{"usage_role":"train"},
            "plan_state":{"N":2,"elements":["Li","O"],"counts":[1,1],"anion_framework":"O",
                          "charge_bucket":"neutral","lattice_system":"triclinic",
                          "spacegroup_bucket":"low","volume_per_atom_bin":"mid"}}


def metric(k,force=.4):
    return Measurement(k,"protocol","Li:1|O:1","verified",True,-2.,force,force*1.5,force*2,.02,.01,20)


@pytest.fixture
def prepared(monkeypatch,tmp_path):
    data=tmp_path/"data_root"; out=tmp_path/"new_loop"
    write_rows(data/"data/plans/val.jsonl",[make_plan("dev"+str(i)) for i in range(4)])
    write_rows(data/"data/structures/train.jsonl",[dict(make_plan("replay"),body_token_ids=[1]*15)])
    rc=types.ModuleType("dlm_iclr.runtime.config")
    rc.asset=lambda c,k:c["models"][k]
    rc.run_root=lambda c:data
    parent=types.ModuleType("dlm_iclr.runtime");parent.__path__=[]
    monkeypatch.setitem(sys.modules,"dlm_iclr.runtime",parent)
    monkeypatch.setitem(sys.modules,"dlm_iclr.runtime.config",rc)
    config={"models":{"dlm":"base","b0":"old_draft","c1":"old_head"}}
    settings=deepcopy(DEFAULTS);settings.update(development_sources=4,plans_per_round=3,min_teachers=2,min_pairs=1)
    settings["gate"]["min_mean_gain"]=0.0001
    monkeypatch.setattr(wf,"preflight",lambda *a: {})
    monkeypatch.setattr(wf,"model_identity",lambda *a:digest(list(map(str,a))))
    calls={"train":0,"fresh":[],"verifier":0}
    def fresh(c,folder,r,*args):
        calls["fresh"].append(r)
        return [make_plan(f"round{r}:source{i}") for i in range(3)]
    monkeypatch.setattr(wf,"fresh_plans",fresh)
    def collect(c,assets,plans,*args):
        result=[]
        for p in plans:
            base=metric(p["source_id"])
            better=Measurement("teacher:"+p["source_id"],"protocol","Li:1|O:1","verified",True,
                               -2.1,.1,.2,.3,.01,.01,10)
            result.append({"plan":p,"raw":{"body_token_ids":[1]*15,"measurement":base.to_dict()},
                           "teachers":[{"body_token_ids":[2]*15,"body_prompt":p["body_prompt"],
                              "exact_record_key":better.record_key,"exact_token_remeasured":True,
                              "measurement":better.to_dict(),"origin":"F_teacher"}]})
        return result
    monkeypatch.setattr(wf,"collect_data",collect)
    def train(base,adapter,teachers,pairs,replay,history,folder,*args,**kwargs):
        calls["train"]+=1
        assert len(teachers)==len(pairs)==3
        if calls["train"]==2: assert history
        p=Path(folder)/"checkpoint";p.mkdir(parents=True,exist_ok=True);return str(p)
    monkeypatch.setattr(learning,"train_draft",train)
    def adapt(base,adapter,head,rows,path,*args,**kwargs):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text("candidate_head");return str(p)
    monkeypatch.setattr(learning,"adapt_c1",adapt)
    def dev(c,assets,plans,*args,**kwargs):
        force=.4 if assets["draft"]=="old_draft" else (.15 if "round_000" in assets["draft"] else .1)
        return {p["source_id"]:metric(p["source_id"],force) for p in plans}
    monkeypatch.setattr(wf,"development",dev)
    def verifier(*a,**k):
        calls["verifier"]+=1
        return {"validation":{"deployable":False}}
    monkeypatch.setattr(wf,"train_verifier",verifier)
    monkeypatch.setattr(step_budget,'probe_shorter_refinement',lambda *a,**k:{'selected_steps':k['current_steps'],'tested':False})
    return config,settings,out,calls


def test_two_fresh_rounds_publish_and_resume_without_retraining(prepared):
    config,settings,out,calls=prepared
    result=wf.run(config,out,settings,device="cpu")
    assert calls["fresh"]==[0,1] and calls["train"]==2 and calls["verifier"]==2
    assert "round_001" in result["draft"]
    assert read_json(out/"round_001/decision.json")["status"]=="accepted"
    result2=wf.run(config,out,settings,device="cpu")
    assert calls["train"]==2 and calls["fresh"]==[0,1]
    assert result2==result==read_json(out/"active_assets.json")


def test_no_teacher_stops_without_training(prepared,monkeypatch):
    config,settings,out,calls=prepared
    def empty(c,assets,plans,*args):
        return [{"plan":p,"raw":{"body_token_ids":[1]*15,"measurement":metric(p["source_id"]).to_dict()},
                 "teachers":[]} for p in plans]
    monkeypatch.setattr(wf,"collect_data",empty)
    result=wf.run(config,out,settings,device="cpu")
    assert calls["train"]==0 and result["draft"]=="old_draft"
    assert read_json(out/"round_000/decision.json")["reason"]=="insufficient_verified_exact_token_teachers"


def test_rejected_student_not_published(prepared,monkeypatch):
    config,settings,out,calls=prepared
    monkeypatch.setattr(wf,"development",lambda c,a,p,*args,**kwargs:{r["source_id"]:metric(r["source_id"],.4) for r in p})
    result=wf.run(config,out,settings,device="cpu")
    assert calls["train"]==1 and calls["verifier"]==0
    assert result["draft"]=="old_draft"
    assert read_json(out/"round_000/decision.json")["status"]=="rejected"


def test_configuration_change_cannot_resume(prepared):
    config,settings,out,calls=prepared
    wf.run(config,out,settings,device="cpu")
    changed=deepcopy(settings);changed["plans_per_round"]+=1
    with pytest.raises(ValueError,match="different settings"):
        wf.run(config,out,changed,device="cpu")


def test_learning_signal_loop_does_not_require_online_verifier(prepared):
    config,settings,out,calls=prepared
    settings['verifier']['enabled']=False
    result=wf.run(config,out,settings,device='cpu')
    assert calls['train']==2 and calls['verifier']==0
    assert result['verifier'] is None and 'round_001' in result['draft']
    assert read_json(out/'round_001/verifier_policy.json')['verified_teacher_and_preference_signal_compilation']
