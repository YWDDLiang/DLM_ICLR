"""Lightweight tests for baseline-facing contracts using explicit stand-in interfaces."""
from pathlib import Path
import sys
import types
from copy import deepcopy
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from dlm_iclr.draft_loop.backend import make_tracking_refiner
from dlm_iclr.draft_loop.settings import DEFAULTS
from dlm_iclr.draft_loop.verifier import JointController


def test_trajectory_export_preserves_endpoint_and_rng(monkeypatch):
    configmodule=types.ModuleType("dlm_iclr.runtime.config");configmodule.asset=lambda c,k:"frozen.pt"
    refmodule=types.ModuleType("dlm_iclr.diffusion.refinement")
    class Model:
        def sample(self,*a,**kw):
            value=torch.rand(1)
            # t=2 deliberately has invalid lengths; logging must not invalidate the endpoint.
            matrices=torch.ones(4,1,3,3);matrices[1]=0
            return {"value":value,"atom_types":torch.tensor([8])}, {
                "all_lattices":matrices,"all_frac_coords":torch.zeros(4,1,3)}
    class Refiner:
        def __init__(self,*a,steps=3,**kw):self.steps=steps;self.model=Model()
        def sample(self,graph,seed):
            torch.manual_seed(seed)
            out,_=self.model.sample(graph)
            return out
    refmodule.Refiner=Refiner
    refmodule.lattices_to_parameters=lambda l:(l[:,0,:],torch.full((1,3),90.0))
    pm=types.ModuleType("pymatgen.core")
    class Lattice:
        @staticmethod
        def from_parameters(*args):
            if min(args[:3])<=0:raise ValueError("bad intermediate")
            return args
    class Structure:
        def __init__(self,*args):self.args=args
        def as_dict(self):return {"saved":True}
    pm.Structure=Structure;pm.Lattice=Lattice
    monkeypatch.setitem(sys.modules,"dlm_iclr.runtime.config",configmodule)
    monkeypatch.setitem(sys.modules,"dlm_iclr.diffusion.refinement",refmodule)
    monkeypatch.setitem(sys.modules,"pymatgen.core",pm)
    cfg={"diffusion":{"reuse_fixed_geometry":True}}
    settings={"teacher_steps":3,"teacher_times":[2,1]}
    ref=Refiner(steps=3);a=ref.sample({},17);rng_a=torch.get_rng_state().clone()
    tracked=make_tracking_refiner(cfg,None,settings,"cpu")
    b=tracked.sample({},17);rng_b=torch.get_rng_state().clone()
    assert torch.equal(a["value"],b["value"]) and torch.equal(rng_a,rng_b)
    assert len(tracked.selected_states)==1
    assert tracked.selected_states[0]["origin"]=="F_state_t1_from_3"


def test_recipe_uses_fresh_conditions_without_retraining_planner():
    assert DEFAULTS["max_rounds"]>=2 and DEFAULTS["plans_per_round"]>0
    assert DEFAULTS["training"]["preference_warmup_updates"]<DEFAULTS["training"]["updates"]
    source=(ROOT/"src/dlm_iclr/draft_loop/workflow.py").read_text()
    assert "generate_plans(" in source and "planner_weights_updated=False" in source
    assert "load_editor" not in source and 'sample(config, "c2"' not in source


def test_published_sampling_recomputes_assets_instead_of_using_old_editor():
    source=(ROOT/"src/dlm_iclr/draft_loop/backend.py").read_text()
    assert "load_editor" not in source
    assert "Constructor(model,tokenizer" in source
    assert "axis_head=head" in source
