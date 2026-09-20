from __future__ import annotations
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import types
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
# In the delivery tree tests live at overlay/tests/draft_loop: repository root is parents[2].
from dlm_iclr.draft_loop.common import Budget,read_json,write_json,seed_for
from dlm_iclr.draft_loop.conditions import ConditionLedger,composition_key,condition_key
from dlm_iclr.draft_loop.quality import Measurement,prefer,is_anchor,raw_utility,publication_gate
from dlm_iclr.draft_loop.settings import DEFAULTS,merge,load_loop
from dlm_iclr.draft_loop.data import compile_teachers,balanced_history
from dlm_iclr.draft_loop.learning import (geometry_positions,mask_view,TypedVocabulary,forward_score,
                                         preference_loss,reference_parameters)
from dlm_iclr.draft_loop.verifier import JointController,FeatureBuilder,projected_action,fit_process_verifier,ProcessVerifier


def measurement(key="raw",**kw):
    m=Measurement(key,"protocol","Li:1|O:1","verified",True,-2.0,0.4,0.6,1.5,0.1,0.02,50)
    return replace(m,**kw)


def plan(source="p",**kw):
    state={"N":2,"elements":["Li","O"],"counts":[1,1],"anion_framework":"O",
           "charge_bucket":"neutral","lattice_system":"triclinic","spacegroup_bucket":"low",
           "volume_per_atom_bin":"mid"}
    state.update(kw)
    return {"source_id":source,"body_eligible":True,"plan_state":state,
            "body_prompt":"fixed Plan\n","provenance":{"usage_role":"train"},"ordinal":0,"original_ordinal":0}


def test_atomic_json(tmp_path):
    p=tmp_path/"a/b.json"; write_json(p,{"中文":"value"}); assert read_json(p)=={"中文":"value"}
    with pytest.raises(ValueError):write_json(p,{"x":float("nan")})
    assert read_json(p)=={"中文":"value"}


def test_budget_persists_and_rejects_overrun(tmp_path):
    b=Budget(tmp_path/"budget.json",{"max_wall_seconds":100,"max_drafts":2})
    b.reserve("drafts",2)
    b=Budget(tmp_path/"budget.json",b.limits)
    with pytest.raises(RuntimeError,match="BUDGET_STOP"):b.reserve("drafts")
    assert b.state["counts"]["drafts"]==2


def test_seed_deterministic():
    assert seed_for(17,"round",1)==seed_for(17,"round",1)
    assert seed_for(17,"round",1)!=seed_for(17,"round",2)


def test_condition_ledger_fresh_resume_and_cap(tmp_path):
    ledger=ConditionLedger(tmp_path/"ledger.db"); counts=Counter()
    p=plan(); assert ledger.admit(p,0,1,counts)==(True,"accepted")
    assert ledger.admit(plan("duplicate"),0,1,counts)[1]=="duplicate_condition"
    assert ledger.admit(plan("newcue",charge_bucket="mixed"),0,1,counts)[1]=="per_round_composition_cap"
    ledger.close(); ledger=ConditionLedger(tmp_path/"ledger.db")
    # Resuming the same recorded draw must not lose an accepted condition.
    assert ledger.admit(p,0,1,Counter())==(True,"accepted")
    assert ledger.admit(plan("next",charge_bucket="other"),1,1,Counter())[0]
    ledger.close()


def test_reserved_compositions_and_invalid_plans(tmp_path):
    ledger=ConditionLedger(tmp_path/"ledger.db");ledger.reserve([plan()],"development")
    assert ledger.admit(plan("x",N=4,counts=[2,2]),0,4,Counter())[1]=="reserved_composition"
    bad=plan("bad");bad["body_eligible"]=False
    assert ledger.admit(bad,0,4,Counter())[1]=="invalid_plan"
    ledger.close()


def test_condition_key_contains_soft_conditions():
    assert composition_key(plan()["plan_state"])==composition_key(plan(N=4,counts=[2,2])["plan_state"])
    assert condition_key(plan()["plan_state"])!=condition_key(plan(charge_bucket="x")["plan_state"])


def test_no_reward_for_extreme_negative_hull():
    assert raw_utility(measurement(raw_hull=-0.1))==raw_utility(measurement(raw_hull=-100))


def test_unknown_never_negative():
    assert raw_utility(measurement(status="worker_error",verified=False)) is None
    assert raw_utility(measurement(raw_hull=None)) is None
    assert raw_utility(measurement(status="invalid_raw",verified=False))==-12


def test_same_composition_and_protocol_required():
    with pytest.raises(ValueError):prefer(measurement(composition="Na:1|O:1"),measurement(),DEFAULTS["quality"])
    with pytest.raises(ValueError):prefer(measurement(protocol_key="other"),measurement(),DEFAULTS["quality"])


def test_raw_improvement_not_just_terminal():
    current=measurement()
    cand=measurement("q",terminal_hull=-0.1,force_rms=0.8)
    assert not prefer(cand,current,DEFAULTS["quality"])[0]
    cand=measurement("q",raw_energy=-2.1,force_rms=0.2,force_max=0.3,stress_max=0.8)
    assert prefer(cand,current,DEFAULTS["quality"])[0]


def test_bad_quantized_teacher_not_used():
    p=plan(); raw={"body_token_ids":[1]*15,"measurement":measurement().to_dict()}
    teacher={"body_token_ids":[2]*15,"exact_token_remeasured":True,"exact_record_key":"q",
             "body_prompt":p["body_prompt"],"measurement":measurement("q",force_rms=2,force_max=4).to_dict(),
             "origin":"F_end"}
    t,pairs,report=compile_teachers([{"plan":p,"raw":raw,"teachers":[teacher]}],DEFAULTS)
    assert not pairs and report["teacher_sources"]==0


def test_label_geometry_binding_enforced():
    p=plan(); raw={"body_token_ids":[1]*15,"measurement":measurement().to_dict()}
    t={"body_token_ids":[2]*15,"exact_token_remeasured":True,"exact_record_key":"different",
       "body_prompt":p["body_prompt"],"measurement":measurement("q").to_dict()}
    with pytest.raises(ValueError,match="exact decoded"):
        compile_teachers([{"plan":p,"raw":raw,"teachers":[t]}],DEFAULTS)


def test_one_source_one_teacher_and_pair():
    p=plan(); raw={"body_token_ids":[1]*15,"measurement":measurement().to_dict()}
    teachers=[{"body_token_ids":[2+i]*15,"exact_token_remeasured":True,"exact_record_key":str(i),
               "body_prompt":p["body_prompt"],"measurement":measurement(str(i),raw_energy=-2.1,
                  force_rms=0.1,force_max=0.2,stress_max=0.4).to_dict(),"origin":"F_end"} for i in range(5)]
    t,pairs,report=compile_teachers([{"plan":p,"raw":raw,"teachers":teachers}],DEFAULTS)
    assert len(t)==len(pairs)==1
    assert t[0]["prompt"]==p["body_prompt"] and pairs[0]["rejected_tokens"]==raw["body_token_ids"]


def test_development_gate_includes_failures_and_unknowns():
    before={str(i):measurement(str(i)) for i in range(10)}
    after={k:replace(v,force_rms=.1,force_max=.2,stress_max=.3,raw_hull=.02) for k,v in before.items()}
    assert publication_gate(before,after,DEFAULTS["gate"])["accepted"]
    after["0"]=measurement(status="invalid_raw",verified=False)
    assert "raw_validity_regression" in publication_gate(before,after,DEFAULTS["gate"])["reasons"]
    for k in ["0","1","2","3"]: after[k]=measurement(status="worker_error",verified=False)
    assert not publication_gate(before,after,DEFAULTS["gate"])["accepted"]


def test_development_gate_checks_all_same_ids():
    with pytest.raises(ValueError):publication_gate({"a":measurement()},{"b":measurement()},DEFAULTS["gate"])


def test_settings_reject_silent_typo_and_replay_overflow():
    with pytest.raises(ValueError):merge(DEFAULTS,{"max_ronds":10})


class ToyTokenizer:
    def __init__(self):
        self.vocab={"pad":0,"mask":1}
        for family in ("LA","LB","LC","AA","AB","AG","X","Y","Z"):
            for k in (range(101) if family in "XYZ" else range(5)):
                self.vocab[f"<{family}_{k:03d}>"]=len(self.vocab)
        self.pad_token_id=0
    def get_vocab(self):return self.vocab
    def __call__(self,text,**kwargs):return {"input_ids":[2,3]}


class ToyModel(torch.nn.Module):
    def __init__(self,width):
        super().__init__();self.lora_A=torch.nn.Parameter(torch.randn(width)/100)
    def forward(self,ids,attention_mask=None):
        return types.SimpleNamespace(logits=self.lora_A[None,None,:].expand(ids.shape[0],ids.shape[1],-1))


def toy_body(tok,n=2):
    v=tok.vocab
    return [2]+[v[f"<{f}_001>"] for f in ("LA","LB","LC","AA","AB","AG")]+sum(
        ([3,v["<X_020>"],v["<Y_030>"],v["<Z_040>"]] for _ in range(n)),[])


def test_masks_preserve_composition_and_share_positions():
    body=list(range(15));a,positions,p=mask_view(body,17,999,.15,.85)
    b,other,_=mask_view([x+100 for x in body],17,999,.15,.85)
    assert positions==other and positions
    assert all(a[i]==body[i] for i in (0,7,11))


def test_typed_periodic_alias_mass():
    tok=ToyTokenizer();vocab=TypedVocabulary(tok);v=torch.zeros(len(tok.vocab))
    lp,ids=vocab.log_probs(v,8,1)
    assert len(ids)==100 and torch.isclose(lp.exp()[0],torch.tensor(2/101))
    assert tok.vocab["<X_100>"] not in ids


def test_matched_denoising_forward_and_gradients():
    tok=ToyTokenizer();model=ToyModel(len(tok.vocab));vocab=TypedVocabulary(tok)
    score,vec=forward_score(model,tok,vocab,{"body_prompt":"P"},toy_body(tok),19,DEFAULTS["training"],1)
    (-score).backward()
    assert torch.isfinite(model.lora_A.grad).all() and model.lora_A.grad.abs().sum()>0
    score2,_=forward_score(model,tok,vocab,{"body_prompt":"P"},toy_body(tok),19,DEFAULTS["training"],1)
    assert score.item()==score2.item()


def test_preference_direction_and_reference_restore():
    pos=torch.tensor(-1.,requires_grad=True);neg=torch.tensor(-2.,requires_grad=True)
    loss=preference_loss(pos,neg,torch.tensor(-1.),torch.tensor(-2.),.5);loss.backward()
    assert pos.grad<0 and neg.grad>0
    model=ToyModel(10);named=list(model.named_parameters());old=model.lora_A.detach().clone()
    with pytest.raises(RuntimeError):
        with reference_parameters(named,{"lora_A":torch.zeros(10)}):
            assert not model.lora_A.any();raise RuntimeError()
    assert torch.equal(old,model.lora_A)


def install_mask_stub(monkeypatch):
    module=types.ModuleType("dlm_iclr._core.fixed_slot");module.MASK_TOKEN_ID=1
    monkeypatch.setitem(sys.modules,"dlm_iclr._core.fixed_slot",module)


class ToyLaw:
    def __init__(self):self.called=0
    def sample(self,n,generator):
        self.called+=1
        return torch.randint(0,3,(n,2),generator=generator)


def sampler_inputs():
    sampler=types.SimpleNamespace(n=2,axis_tokens=[list(range(2,103)) for _ in range(3)],
                                  coord_map=[{i+2:i for i in range(101)} for _ in range(3)])
    body=torch.tensor([2]*15);body[10]=1;body[14]=1
    logits=torch.zeros(1,17,103);logits[0,12,2:5]=torch.tensor([1.,2.,3.]);logits[0,16,2:5]=torch.tensor([3.,2.,1.])
    hidden=torch.randn(1,17,8,generator=torch.Generator().manual_seed(3))
    args={"prompt_length":2,"semantic_group":4,"step_in_group":0,"base_seeds":[17]}
    return sampler,body,logits,hidden,args


def test_observation_returns_exact_baseline_and_preserves_global_rng(monkeypatch):
    install_mask_stub(monkeypatch)
    sampler,body,logits,hidden,args=sampler_inputs();sample=torch.tensor([0,1]);law=ToyLaw()
    ctrl=JointController(DEFAULTS["verifier"],collect=True)
    rng=torch.get_rng_state().clone()
    result=ctrl.select(sampler,law,sample,logits,hidden,body,[10,14],2,args)
    assert result is sample and torch.equal(rng,torch.get_rng_state())
    action=ctrl.probes[0]["selected_action"]
    ctrl.record_commit({"committed_positions":[action[0]],"committed_tokens":[action[1]]})


def test_commit_mismatch_rejected(monkeypatch):
    install_mask_stub(monkeypatch)
    sampler,body,logits,hidden,args=sampler_inputs();ctrl=JointController(DEFAULTS["verifier"],collect=True)
    ctrl.select(sampler,ToyLaw(),torch.tensor([0,1]),logits,hidden,body,[10,14],2,args)
    with pytest.raises(RuntimeError,match="actual single"):
        ctrl.record_commit({"committed_positions":[8],"committed_tokens":[99]})


def test_counterfactual_requires_exact_state_and_available_action(monkeypatch):
    install_mask_stub(monkeypatch)
    sampler,body,logits,hidden,args=sampler_inputs();sample=torch.tensor([0,1])
    ctrl=JointController(DEFAULTS["verifier"],collect=True)
    ctrl.select(sampler,ToyLaw(),sample,logits,hidden,body,[10,14],2,args)
    probe=ctrl.probes[0]
    forced={"state_key":probe["state_key"],"action":probe["actions"][-1]}
    other=JointController(DEFAULTS["verifier"],forced=forced,collect=True)
    other.select(sampler,ToyLaw(),sample,logits,hidden,body,[10,14],2,args)
    a=forced["action"];other.record_commit({"committed_positions":[a[0]],"committed_tokens":[a[1]]})
    assert other.forced_hit and other.force_verified
    invalid=JointController(DEFAULTS["verifier"],forced=dict(forced,action=[10,999]))
    with pytest.raises(RuntimeError,match="absent"):
        invalid.select(sampler,ToyLaw(),sample,logits,hidden,body,[10,14],2,args)


def test_late_axis_only(monkeypatch):
    install_mask_stub(monkeypatch)
    s,b,l,h,args=sampler_inputs();sample=torch.tensor([0,1]);law=ToyLaw()
    c=JointController(DEFAULTS["verifier"],collect=True)
    assert c.select(s,law,sample,l,h,b,[8,12],0,args) is sample and not c.probes and law.called==0


def test_verifier_rejects_wrong_asset_binding():
    with pytest.raises(ValueError,match="different"):
        ProcessVerifier({"binding":"A"},"B")


def test_verifier_requires_independent_sources(tmp_path):
    r=fit_process_verifier([],tmp_path/"q.json","binding",DEFAULTS["verifier"])
    assert not r["validation"]["deployable"]


def test_process_value_fits_real_action_targets(tmp_path):
    rows=[]
    for i in range(40):
        for a in (0,1):
            rows.append({"source_id":str(i),"state_key":"same_prefix", "features":[float(a),i/40],
                         "target":-2.+a*.5})
    result=fit_process_verifier(rows,tmp_path/"q.json","binding",DEFAULTS["verifier"])
    assert result["validation"]["deployable"]
    q=ProcessVerifier(result,"binding")
    assert q.predict([[1,.4]])>q.predict([[0,.4]])
    assert not set(result["train_sources"]) & set(result["validation_sources"])


def test_balanced_history_does_not_multiply_repeated_condition():
    a={"plan_state":{"x":1},"source_id":"old"};b={"plan_state":{"x":1},"source_id":"new"}
    assert balanced_history([[a],[b]])==[b]


def test_duplicate_candidates_do_not_consume_calibrated_guidance_opportunity(monkeypatch):
    install_mask_stub(monkeypatch)
    sampler,body,logits,hidden,args=sampler_inputs();sample=torch.tensor([0,1])
    class Law:
        def __init__(self,draw):self.draw=draw
        def sample(self,n,generator):return self.draw[None]
    class Guide:
        saved={'validation':{'advantage_margin':0}}
        calls=0
        def predict(self,features):
            self.calls+=1
            return np.array([0.,1.,1.])
    guide=Guide();settings={**DEFAULTS['verifier'],'skip_duplicate_opportunity':True}
    ctrl=JointController(settings,guide=guide)
    assert ctrl.select(sampler,Law(sample),sample,logits,hidden,body,[10,14],2,args) is sample
    assert ctrl.guided_actions==0 and guide.calls==0
    ctrl.select(sampler,Law(torch.tensor([0,0])),sample,logits,hidden,body,[10,14],2,args)
    assert ctrl.guided_actions==1 and guide.calls==1
