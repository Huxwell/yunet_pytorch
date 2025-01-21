import math
import torch
from torch.utils.data import DistributedSampler as _DistributedSampler
from mmdet.utils.util_distribution import get_device


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True,
        seed=0):
        print('Filip YuNet Minify: Function fidx=0 __init__ called in mmdet/datasets/samplers/distributed_sampler.py:L13 ')
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
            shuffle=shuffle)
        device = get_device()
        self.seed = seed

    def __iter__(self):
        print('Filip YuNet Minify: Function fidx=1 __iter__ called in mmdet/datasets/samplers/distributed_sampler.py:L31 ')
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()
        indices = (indices * math.ceil(self.total_size / len(indices)))[:
            self.total_size]
        assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)
