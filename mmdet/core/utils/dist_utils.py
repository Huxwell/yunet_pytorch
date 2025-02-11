import torch.distributed as dist
from mmcv.runner import OptimizerHook


class DistOptimizerHook(OptimizerHook):
    """Deprecated optimizer hook for distributed training."""


def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor
