import unittest
import torch
import pytest
scatter = pytest.importorskip('torch_scatter').scatter
from dlm_iclr.diffusion.reduction_audit import OrderedScatterMean


class OrderedReductionTests(unittest.TestCase):
    def test_mean_and_gradient_match_including_empty_groups(self):
        x=torch.tensor([[1.,2.],[3.,7.],[8.,2.],[5.,1.]],dtype=torch.float64,requires_grad=True)
        idx=torch.tensor([2,0,2,0])
        legacy=scatter(x,idx,dim=0,reduce='mean',dim_size=4)
        actual=OrderedScatterMean()(x,idx,dim_size=4)
        torch.testing.assert_close(actual,legacy,rtol=0,atol=0)
        a=torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0]
        b=torch.autograd.grad(legacy.square().sum(),x)[0]
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_changed_index_does_not_reuse_wrong_segments(self):
        kernel=OrderedScatterMean();x=torch.tensor([[1.],[4.],[9.]])
        idx=torch.tensor([0,0,1]);kernel(x,idx)
        idx[1]=1
        torch.testing.assert_close(kernel(x,idx),scatter(x,idx,dim=0,reduce='mean'))


if __name__=='__main__':unittest.main()
