"""Bind the validated generator's inference settings to its model manifest."""
from copy import deepcopy
import math


def bind_inference_protocol(config, assets):
    protocol=assets.get('inference_protocol')
    if protocol is None:return config
    if set(protocol)!={'body_temperature','F_reduction_protocol'}:
        raise ValueError('Unknown or incomplete frozen inference protocol')
    temperature=protocol['body_temperature']
    if isinstance(temperature,bool) or not isinstance(temperature,(int,float)) or not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Finite positive frozen body temperature required')
    if protocol['F_reduction_protocol']!='ordered_csr_v1':
        raise ValueError('Verified feedback evaluation requires its ordered F protocol')
    result=deepcopy(config)
    result['c1']['temperature']=float(temperature)
    result['diffusion']['reduction_protocol']=protocol['F_reduction_protocol']
    return result
