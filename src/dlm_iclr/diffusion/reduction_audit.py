"""Opt-in ordered mean reduction for isolated refiner reproducibility audits.

No training or production sampler switches this on automatically. The mean and
graph stay the same; only the floating-point reduction order is fixed.
"""
from contextlib import contextmanager
import torch


class OrderedScatterMean:
    def __init__(self):
        self.cache={}

    def __call__(self,src,index,dim=0,out=None,dim_size=None,reduce='mean'):
        from torch_scatter import segment_csr
        if dim!=0 or out is not None or reduce!='mean' or index.ndim!=1:
            raise ValueError('Audit only supports the two dim=0 mean CSPNet reductions')
        if src.shape[0]!=index.numel():raise ValueError('Mismatched source and group index')
        key=(index.data_ptr(),index.numel(),str(index.device),index._version,dim_size)
        cached=self.cache.get(key)
        if cached is None:
            size=int(index.max())+1 if dim_size is None and index.numel() else int(dim_size or 0)
            order=torch.argsort(index,stable=True)
            counts=torch.bincount(index,minlength=size)
            ptr=torch.cat((index.new_zeros(1),counts.cumsum(0)))
            # Retain the original tensor so its storage address cannot be reused
            # for a different graph while a cache entry is live.
            cached=(index,order,ptr);self.cache[key]=cached
        return segment_csr(src[cached[1]],cached[2],reduce='mean')


@contextmanager
def ordered_csp_reductions():
    """Scoped to one isolated audit process; caller restores the legacy kernel."""
    from dlm_iclr._vendor.crysllmgen.models_ddpm import cspnet
    original=cspnet.scatter
    cspnet.scatter=OrderedScatterMean()
    try:
        yield
    finally:
        cspnet.scatter=original
