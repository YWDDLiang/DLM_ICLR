"""Dataset capacity must agree across token, planner, construction and edit views."""
import os
from pathlib import Path
import subprocess
import sys
import pytest

@pytest.mark.parametrize('count',[5,52])
def test_capacity_in_isolated_worker(count):
    code=r'''
import re,torch
from dlm_iclr._core.dynamic_crystal import arrays_to_dynamic_answer,parse_dynamic_answer,build_special_tokens
from dlm_iclr._core.expert_edit import ExpertEditConfig,materialize_edit_batch,inference_view
from dlm_iclr._core.periodic_state_conditioning import PeriodicStateConfig,PeriodicStateConditioner
from dlm_iclr._core.r03_physics_transfer import build_repair_constraints
from dlm_iclr._core.r03_geometry_bridge import _checked_groups
from dlm_iclr._core.r5_plan_state import plan_state_from_arrays
from dlm_iclr.data.plans import axis_schedule
from dlm_iclr.runtime.capacity import MAX_ATOMS
n=MAX_ATOMS
coords=[[((i%4)+.1)/4,(((i//4)%4)+.1)/4,((i//16)+.1)/4] for i in range(n)]
body,_=arrays_to_dynamic_answer([20,20,20],[90,90,90],['Si']*n,coords)
a=parse_dynamic_answer(body,strict=True)
vocab={t:i+1 for i,t in enumerate(build_special_tokens())}
class Tokenizer:
 pad_token_id=0
 def get_vocab(self):return vocab
t=Tokenizer();ids=[vocab[x] for x in re.findall(r'<[^>]+>',body)]
assert len(ids)==7+4*n
assert build_repair_constraints(t)['max_atoms']==n
p=plan_state_from_arrays(a,metadata={'spacegroup.number':1})
assert sum(map(len,_checked_groups(axis_schedule(p))))==len(ids)
v=inference_view([0],ids,ids,n,1,[len(ids)-1])
b=materialize_edit_batch([v],t,torch.device('cpu'))
assert b['site_targets'].shape==(1,n)
assert ExpertEditConfig(hidden_size=8).max_sites==n
module=PeriodicStateConditioner(PeriodicStateConfig(hidden_size=8,width=8,max_sites=n))
out=module(lattice=torch.eye(3)[None]*20,lattice_known=torch.ones(1,dtype=torch.bool),
 fractional=torch.tensor(coords)[None],species=torch.full((1,n),14,dtype=torch.long),
 site_known=torch.ones((1,n),dtype=torch.bool),program_rank=torch.arange(n)[None],
 active_sites=torch.zeros((1,n),dtype=torch.bool))
assert out['site_embeddings'].shape==(1,n,8)
loss=out['site_embeddings'].square().sum()+out['cell_embedding'].square().sum()
loss.backward()
assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
'''
    env=dict(os.environ,DLM_MAX_ATOMS=str(count),DLM_MIN_ATOMS='1',DLM_DATASET_LABEL='capacity-test',
             PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
    result=subprocess.run([sys.executable,'-X','utf8','-c',code],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
